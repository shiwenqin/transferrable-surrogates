import os
import json
import shutil
import argparse
import datetime

import torch
import wandb
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from sklearn.metrics import mean_absolute_error
from peft import (
    LoraConfig,
    get_peft_model,
)
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

# Model params
parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B')

# Data params
parser.add_argument('--data_path', type=str)
parser.add_argument('--task', type=str, default='cifar10')
parser.add_argument('--train_size', type=int, default=None)
parser.add_argument('--encoding', type=str, default='arch_long', choices=['arch', 'arch_long', 'arch_pytorch'])
parser.add_argument('--discard_zero', type=bool, default=True)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--target_col', type=str, default='acc')

# Train params
parser.add_argument('--mode', type=str, default='fft', choices=['fft', 'head_only', 'lora'])
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--r', type=int, default=32) # LoRA params
parser.add_argument('--lora_alpha', type=int, default=128) # LoRA params
parser.add_argument('--wandb_project', type=str, default='LLM_performance')

# Prompt params
parser.add_argument('--prompt_path', type=str, default='../prompts')
parser.add_argument('--role_prompt', type=str, default='acc.txt')
parser.add_argument('--instruction_prompt', type=str, default='cifar10_acc.txt')
parser.add_argument('--definition_prompt', type=str, default='einspace_simple.txt')
parser.add_argument('--num_shots', type=int, default=3)

# Output params
parser.add_argument('--output_path', type=str, default=None)
parser.add_argument('--save', type=bool, default=True)
parser.add_argument('--save_pred', type=bool, default=True)

args = parser.parse_args()

run_name = f'{args.task}_{args.mode}_{args.model_name.split("/")[-1]}_{args.encoding}_{args.num_shots}'

wandb.init(
    project = args.wandb_project,
    config = vars(args),
    name = run_name,
)

if not os.path.exists(args.data_path):
    raise FileNotFoundError(f"Data path {args.data_path} does not exist")

# Load data
train_data = os.path.join(args.data_path, 'train.csv')
val_data = os.path.join(args.data_path, 'val.csv')
train_df = pd.read_csv(train_data)
val_df = pd.read_csv(val_data)
if args.discard_zero:
    train_df = train_df[train_df['acc'] != 0]
    val_df = val_df[val_df['acc'] != 0].reset_index(drop=True)
if args.train_size is not None:
    train_df = train_df.sample(n=args.train_size, random_state=args.seed)
train_df = train_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
train_ds = Dataset.from_pandas(train_df).rename_column(args.target_col, "labels")
val_ds = Dataset.from_pandas(val_df).rename_column(args.target_col, "labels")
print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

role_prompt = open(os.path.join(args.prompt_path, 'role/' , args.role_prompt)).read()
instruction_prompt = open(os.path.join(args.prompt_path, 'instruction/' , args.instruction_prompt)).read() if args.instruction_prompt is not None else ''
definition_prompt = open(os.path.join(args.prompt_path, 'definition/' , args.definition_prompt)).read() if args.definition_prompt is not None else ''

base_prompt = role_prompt + '\n' + instruction_prompt + '\n' + definition_prompt

if args.num_shots != 0:
    shots = train_df.sample(args.num_shots, random_state=args.seed)
    train_df = train_df.drop(shots.index)
    train_df = train_df.reset_index(drop=True)
    shots = shots.reset_index(drop=True)

if args.num_shots != 0:
    example_str = 'Examples: \n'
    for i, row in shots.iterrows():
        example_str += f"Example {i}: \n{row[args.encoding]} \nAccuracy: \n{row['acc']} \n"
    base_prompt += example_str

def tokenize_function(examples):
    txts = [base_prompt + '\nTest Architecture: \n' + example for example in examples[args.encoding]]
    return tokenizer(txts, padding='longest', truncation=True)

train_ds = train_ds.map(tokenize_function, batched=True)
val_ds = val_ds.map(tokenize_function, batched=True)

print(f'Length of a sample: {len(train_ds[0]["input_ids"])}')

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=1,
    problem_type="regression",
    device_map="auto",
)#.to(device)
model.config.pad_token_id = tokenizer.pad_token_id

if args.mode == 'head_only':
    for param in model.base_model.parameters():
        param.requires_grad = False
elif args.mode == 'lora':
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj","v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)

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
with open(os.path.join(output_dir, 'info.json'), 'w') as f:
    json.dump(vars(args), f, indent=4)

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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

if args.save:
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    # remove saved checkpoints
    for file in os.listdir(output_dir):
        if file.startswith('checkpoint'):
            shutil.rmtree(os.path.join(output_dir, file))

# Pred & Save
if args.save_pred:
    preds = trainer.predict(val_ds)
    preds = pd.DataFrame(preds.predictions.flatten(), columns=['pred'])
    preds['true'] = val_df['acc']
    preds.to_csv(os.path.join(output_dir, 'preds.csv'), index=False)

    mae = mean_absolute_error(val_df['acc'], preds['pred'])
    spearman = stats.spearmanr(val_df['acc'], preds['pred'])[0]
    kendall = stats.kendalltau(val_df['acc'], preds['pred'])[0]

    info = json.load(open(os.path.join(output_dir, 'info.json')))
    res = {
        'mae': mae,
        'spearman': spearman,
        'kendall': kendall,
    }
    info.update(res)
    json.dump(info, open(os.path.join(output_dir, 'info.json'), 'w'), indent=4)

    target = val_df['acc']
    target_argsort = np.argsort(target)
    target = target[target_argsort]
    preds = preds['pred'][target_argsort]
    sns.scatterplot(x=range(len(target)), y=target, label='targets')
    sns.scatterplot(x=range(len(preds)), y=preds, label='tuned roberta_large predictions')

    plt.title('Predictions vs targets')
    plt.ylabel('Accuracy')
    plt.xlabel('Indexes (sorted by target accuracy)')
    plt.savefig(os.path.join(output_dir, 'preds.pdf'))
