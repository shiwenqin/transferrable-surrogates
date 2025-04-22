
import os
import json
import argparse
import datetime

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from datasets import Dataset
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as stats

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, default='FacebookAI/roberta-base')
parser.add_argument('--data_path', type=str)
parser.add_argument('--eval_task', type=str, default='cifar10')
parser.add_argument('--output_path', type=str, default=None)
parser.add_argument('--encoding', type=str, default='arch_long', choices=['arch', 'arch_long', 'arch_pytorch'])

args = parser.parse_args()

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

output_file = os.path.join(args.output_path, f"{args.model_path.split('/')[-1]}_acc_inference.json")

df = pd.read_csv(args.data_path)
val_df = df[df['dataset'] == args.eval_task].reset_index(drop=True)

val_ds = Dataset.from_pandas(val_df)
val_ds = val_ds.rename_column("acc", "labels")

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
#tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_path,
    num_labels=1,
    problem_type="regression"
).to(device)
model.eval()

info = {
    'params': vars(args),
}
json.dump(info, open(output_file, 'w'), indent=4)

def tokenize_function(examples):
    return tokenizer(examples[args.encoding], 
                     padding="longest", 
                     truncation=True, 
                     return_tensors="pt")

val_ds = val_ds.map(tokenize_function, batched=True)

val_ds.set_format(
    type='torch', 
    columns=["input_ids", "attention_mask"]  # or omit "label" if not available
)

dataloader = torch.utils.data.DataLoader(val_ds, batch_size=8)

# batch inference
results = []
with torch.no_grad():
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # For regression, outputs.logits is shape (batch_size, 1)
        logits = outputs.logits.squeeze(-1)  # shape: (batch_size)
        predictions = logits.cpu().numpy().tolist()
        results.extend(predictions)

mae = mean_absolute_error(val_df['acc'], results)
spearman = stats.spearmanr(val_df['acc'], results)[0]
kendall = stats.kendalltau(val_df['acc'], results)[0]

res = {
    'mae': mae,
    'spearman': spearman,
    'kendall': kendall,
}
info.update(res)
json.dump(info, open(output_file, 'w'), indent=4)

fig = plt.figure(figsize=(10, 6))

targets = val_df['acc']
targets_argsort = np.argsort(targets)
targets = targets[targets_argsort]
bert_preds = np.array(results)[targets_argsort]
sns.scatterplot(x=range(len(targets)), y=targets, label='targets')
sns.scatterplot(x=range(len(bert_preds)), y=bert_preds, label='tuned roberta_large predictions')

plt.title('Predictions vs targets')
plt.ylabel('Accuracy')
plt.xlabel('Indexes (sorted by target accuracy)')
plt.savefig(output_file.replace('.json', '.pdf'))

val_df['pred'] = results
val_df.to_csv(output_file.replace('.json', '.csv'), index=False)