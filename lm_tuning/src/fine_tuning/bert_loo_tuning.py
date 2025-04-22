import os
import json
import shutil
import argparse
import datetime

import torch
import wandb
import pandas as pd
import scipy.stats as stats
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from peft import (
    LoraConfig,
    get_peft_model,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

# Model params
parser.add_argument('--model_name', type=str, default='FacebookAI/roberta-base')

# Data params
parser.add_argument('--data_path', type=str)
parser.add_argument('--eval_task', type=str, default='cifar10')
parser.add_argument('--train_size', type=int, default=None)
parser.add_argument('--standardize', type=str, default='no', choices=['no', 'ecdf', 'minmax'])
parser.add_argument('--encoding', type=str, default='arch', choices=['arch', 'arch_long', 'arch_pytorch'])
parser.add_argument('--discard_zero', type=bool, default=True)
parser.add_argument('--seed', type=int, default=42)

# Train params
parser.add_argument('--mode', type=str, default='fft', choices=['fft', 'head_only', 'lora'])
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--r', type=int, default=16) # LoRA params
parser.add_argument('--lora_alpha', type=int, default=64) # LoRA params
parser.add_argument('--wandb_project', type=str, default='Surrogate_loo_tuning')

# Output params
parser.add_argument('--output_path', type=str, default=None)
parser.add_argument('--save', type=bool, default=True)
parser.add_argument('--save_pred', type=bool, default=True)

args = parser.parse_args()

# Set up wandb
run_name = f"{args.eval_task}_{args.model_name.split('/')[-1]}_{args.mode}_{args.encoding}_{args.standardize}"
wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

# Load data
df = pd.read_csv(args.data_path)

val_df = df[df['dataset'] == args.eval_task].reset_index(drop=True)
train_df = df[df['dataset'] != args.eval_task]
if args.discard_zero:
    train_df = train_df[train_df['acc'] != 0]
if args.train_size:
    train_df = train_df.sample(args.train_size, random_state=args.seed)
train_df = train_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
# Standardize data  
if args.standardize == 'no':
    pass
else:
    datasets = train_df['dataset'].unique()
    for dataset in datasets:
        if args.standardize == 'minmax':
            scaler = MinMaxScaler(feature_range=(0, 1))
            train_df.loc[train_df['dataset'] == dataset, 'acc'] = scaler.fit_transform(train_df[train_df['dataset'] == dataset]['acc'].values.reshape(-1, 1))
        elif args.standardize == 'ecdf':
            ecdf = stats.ecdf(train_df[train_df['dataset'] == dataset]['acc'])
            train_df.loc[train_df['dataset'] == dataset, 'acc'] = ecdf.cdf.evaluate(train_df[train_df['dataset'] == dataset]['acc'])
        else:
            raise ValueError(f"Standardization method {args.standardize} not implemented")
train_ds = Dataset.from_pandas(train_df).rename_column("acc", "labels")
val_ds = Dataset.from_pandas(val_df).rename_column("acc", "labels")
print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=1,
    problem_type="regression"
).to(device)

# Tokenize data
def tokenize_function(examples):
    return tokenizer(examples[args.encoding], padding='longest', truncation=True)
train_ds = train_ds.map(tokenize_function, batched=True)
val_ds = val_ds.map(tokenize_function, batched=True)

# Train mode
if args.mode == 'head_only':
    for param in model.base_model.parameters():
        param.requires_grad = False
elif args.mode == 'lora':
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        target_modules=["query","value"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)

# Define Metrics
def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    predictions = predictions.flatten()
    mae = mean_absolute_error(labels, predictions)
    spearman_corr = stats.spearmanr(labels, predictions)[0]
    kendall_corr = stats.kendalltau(labels, predictions)[0]
    return {"mae": mae, "spearman_corr": spearman_corr, "kendall_corr": kendall_corr}

# Train Setup
date = datetime.datetime.now().strftime("%Y-%m-%d")
output_dir = os.path.join(args.output_path, date, f'{run_name}_{datetime.datetime.now().strftime("%H%M%S")}')
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'args.json'), 'w') as f:
    json.dump(vars(args), f)

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=args.weight_decay,
    warmup_ratio=0.06,
    logging_steps=200,
    lr_scheduler_type="cosine",
    push_to_hub=False,
    report_to="wandb",
    save_strategy="epoch",
    run_name=run_name,
    load_best_model_at_end=True,
    metric_for_best_model="kendall_corr",
    greater_is_better=True,
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()

# Pred & Save
if args.save_pred:
    preds = trainer.predict(val_ds)
    preds = pd.DataFrame(preds.predictions.flatten(), columns=['pred'])
    preds['true'] = val_df['acc']
    preds.to_csv(os.path.join(output_dir, 'preds.csv'), index=False)
if args.save:
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    # remove saved checkpoints
    for file in os.listdir(output_dir):
        if file.startswith('checkpoint'):
            shutil.rmtree(os.path.join(output_dir, file))