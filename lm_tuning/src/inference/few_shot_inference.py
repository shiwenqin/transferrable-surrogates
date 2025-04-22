'''
This document is used to evaluate the few-shot performance of LLMs.

The task is to predict the accuracy of a given architecture on a given dataset.
'''
import os
import re
import json
import argparse
import datetime

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate


# set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default='gpt2')
parser.add_argument('--data_path', type=str)
parser.add_argument('--output_path', type=str, default=None)
parser.add_argument('--num_shots', type=int, default=3)
parser.add_argument('--prompt_path', type=str, default='../prompts')
parser.add_argument('--role_prompt', type=str, default='acc.txt')
parser.add_argument('--instruction_prompt', type=str, default='cifar10_acc.txt')
parser.add_argument('--definition_prompt', type=str, default='einspace.txt')
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--adapter_path', type=str, default=None)

args = parser.parse_args()

# Set the random seed
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
transformers.set_seed(args.random_seed)

# Output file
if args.output_path is not None:
    os.makedirs(args.output_path, exist_ok=True)
    adapter_str = 'finetuned' if args.adapter_path is not None else 'base'
    output_file = os.path.join(args.output_path, f"{args.model_name.split('/')[-1]}_{adapter_str}_{args.random_seed}_few_shot_inference_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.json")

# Load the model
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")

# set pad token
model.config.pad_token_id = model.config.eos_token_id

if args.adapter_path is not None:
    model.load_adapter(args.adapter_path)

# Load the data
train_data = os.path.join(args.data_path, 'train.csv')
val_data = os.path.join(args.data_path, 'val.csv')

train_df = pd.read_csv(train_data)
val_df = pd.read_csv(val_data)

# Choose few-shot samples based on the percentile of the accuracy
def choose_samples(df, num_shots):
    df = df[df['acc'] > 0]
    df.sort_values('acc', inplace=True)

    quantiles = np.linspace(0, 1, num_shots)
    selected_indices = [
        int(q * (len(df) - 1))
        for q in quantiles
    ]
    selected_rows = df.iloc[selected_indices].copy()

    return selected_rows

train_df = choose_samples(train_df, args.num_shots)

# Build the prompts
role_prompt = open(os.path.join(args.prompt_path, 'role/' , args.role_prompt)).read()
instruction_prompt = open(os.path.join(args.prompt_path, 'instruction/' , args.instruction_prompt)).read() if args.instruction_prompt != 'None' else ''
definition_prompt = open(os.path.join(args.prompt_path, 'definition/' , args.definition_prompt)).read() if args.definition_prompt != 'None' else ''

base_prompt = instruction_prompt + '\n' + definition_prompt + '\n'

if args.num_shots != 0:
    example_str = ''
    for i, row in train_df.iterrows():
        example_str += f"Architecture: {row['arch_long']} \n{row['acc']} \n"

    base_prompt += example_str

info = {
    'params': vars(args),
    'prompt': base_prompt,
}
json.dump(info, open(output_file, 'w'), indent=4)

def extract_float(response_text):
    # Regex to match a float that starts with 0. followed by one or more digits
    pattern = re.compile(r'\b0\.\d+\b')

    last_match = None
    start_index = 0

    while True:
        match = pattern.search(response_text, start_index)
        if not match:
            # No more matches
            break
        # Update last_match and move start_index to just after this match
        last_match = match
        start_index = match.end()

    # Return the matched string or None if not found
    return float(last_match.group(0)) if last_match else None

res_list = []
for index in tqdm(range(val_df.shape[0])):
    candidate = val_df['arch_long'][index]

    prompt = f"{base_prompt}Architecture: \n{candidate} \n"
    messages = [
        {
            "role": "system",
            "content": role_prompt
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    correct_acc = val_df['acc'][index]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
    # Tokenize the prompt
    attention_mask = torch.ones_like(input_ids).to(device)

    prompt_len = len(input_ids[0])
    count = 0
    res = None
    for _ in range(10):
        count += 1

        if count == 0:
            # Generate the output
            output = model.generate(input_ids, attention_mask = attention_mask, max_new_tokens=50, num_return_sequences=1, temperature=0.7, do_sample=True)
        else:
            # Generate the output with more tokens
            output = model.generate(input_ids, attention_mask = attention_mask, max_new_tokens=500, num_return_sequences=1, temperature=0.7, do_sample=True)

        # Decode the output new tokens
        output_str = tokenizer.decode(output[:, input_ids.shape[1]:][0], skip_special_tokens=True)

        output_len = len(output_str.split())

        res = extract_float(output_str)
        if res is not None:
            break
    
    res_dict = {
        'index': index,
        'candidate': candidate,
        'correct_acc': correct_acc,
        'predicted_acc': res,
        'num_attempts': count,
        'prompt_len': prompt_len,
        'output_len': output_len,
        'output': output_str
    }

    res_list.append(res_dict)

    curr_res = json.load(open(output_file))
    curr_res['results'] = res_list
    json.dump(curr_res, open(output_file, 'w'), indent=4)

# Calculate MAE
mae = 0
for res in res_list:
    mae += abs(res['correct_acc'] - res['predicted_acc'])
mae /= len(res_list)

# Calculate spearman rank correlation
correct_acc = [res['correct_acc'] for res in res_list]
predicted_acc = [res['predicted_acc'] for res in res_list]

spearman_corr = stats.spearmanr(correct_acc, predicted_acc)[0]

# Calculate Kendall rank correlation
kendall_corr = stats.kendalltau(correct_acc, predicted_acc)[0]

performance = {
    'mae': mae,
    'spearman_corr': spearman_corr,
    'kendall_corr': kendall_corr
}

curr_res = json.load(open(output_file))
curr_res['performance'] = performance
json.dump(curr_res, open(output_file, 'w'), indent=4)