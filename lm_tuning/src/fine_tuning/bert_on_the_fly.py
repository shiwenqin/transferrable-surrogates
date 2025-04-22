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
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

# Model params
parser.add_argument('--model_name', type=str, default='FacebookAI/roberta-base')

# on-the-fly finetuning params
parser.add_argument('--finetune', type=bool, default=False)
parser.add_argument('--target_seed', type=int, default=0)

# Data params
parser.add_argument('--data_path', type=str)
parser.add_argument('--task', type=str, default='cifar10')
parser.add_argument('--pop_size', type=int, default=100)
parser.add_argument('--encoding', type=str, default='arch_long', choices=['arch', 'arch_long', 'arch_pytorch'])
parser.add_argument('--discard_zero', type=bool, default=True)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--target_col', type=str, default='acc')

# Train params
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--wandb_project', type=str, default='Bert_on_the_fly')

# Output params
parser.add_argument('--output_path', type=str, default=None)
parser.add_argument('--save', type=bool, default=False)
parser.add_argument('--save_pred', type=bool, default=True)

args = parser.parse_args()

if not os.path.exists(args.data_path):
    raise FileNotFoundError(f"Data path {args.data_path} does not exist")

# Load data
df = pd.read_csv(args.data_path)
df = df[df['dataset'] == args.task]
df = df[df['seed'] == f'seed={args.target_seed}']

# split data into chunks of size pop_size
chunks = [df[i:i + args.pop_size] for i in range(0, df.shape[0], args.pop_size)]

# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=1,
    problem_type="regression"
).to(device)

for i, chunk in enumerate(chunks):
    
    if i == len(chunks) - 1:
        # no evaluation for the last chunk
        break

    # Set up wandb
    run_name = f"{args.task}_{args.model_name.split('/')[-1]}_{args.encoding}_chunk{i}_{args.target_seed}_{args.finetune}"
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args), reinit=True)

    train_df = chunk
    val_df = chunks[i + 1]

    if args.discard_zero:
        train_df = train_df[train_df[args.target_col] != 0]
        #val_df = val_df[val_df[args.target_col] != 0]

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_ds = Dataset.from_pandas(train_df).rename_column(args.target_col, "labels")
    val_ds = Dataset.from_pandas(val_df).rename_column(args.target_col, "labels")
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

    # Tokenize data
    def tokenize_function(examples):
        return tokenizer(examples[args.encoding], padding='longest', truncation=True)
    train_ds = train_ds.map(tokenize_function, batched=True)
    val_ds = val_ds.map(tokenize_function, batched=True)

    # Define Metrics
    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        predictions = predictions.flatten()
        mae = mean_absolute_error(labels, predictions)
        spearman_corr = stats.spearmanr(labels, predictions)[0]
        kendall_corr = stats.kendalltau(labels, predictions)[0]
        return {"mae": mae, "spearman_corr": spearman_corr, "kendall_corr": kendall_corr}
    
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
        save_strategy="no",
        run_name=run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    if args.finetune:
        trainer.train()

    if args.save:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

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
        # clear plot
        plt.clf()

    wandb.finish()