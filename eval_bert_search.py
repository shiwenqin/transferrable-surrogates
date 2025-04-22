import sys
sys.path.append('..')
from os.path import join, exists
from pickle import load
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from functools import partial
from pprint import pprint
from os.path import join
import yaml
import argparse
import json

import torch

from pcfg import PCFG
from grammars import grammars
from network import Network
from evaluation import evaluation_fn
from arguments import parse_arguments
from data import get_data_loaders
from utils import load_config, get_exp_path, Limiter
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/eval_bert_search.yaml')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--path', type=str, default='experiments/eval_bert_search')
parser.add_argument('--output_path', type=str, default='experiments/eval_bert_search')
args_ro = parser.parse_args()

args = parse_arguments(['--config',args_ro.config, '--device', args_ro.device])
args = load_config(args)
torch.manual_seed(args.seed)

train_loader, val_loader, _, _ = get_data_loaders(
    dataset=args.dataset,
    batch_size=args.batch_size,
    image_size=args.image_size,
    root="../einspace/data",
    load_in_gpu=args.load_in_gpu,
    device=args.device,
    log=args.verbose_eval,
)

def compile_fn(node, args):
    backbone = node.build(node, set_memory_checkpoint=True)
    return Network(
        backbone,
        node.output_params["shape"],
        args.num_classes,
        vars(args)
    ).to(args.device)

limiter = Limiter(
    limits={
        "time": args.time_limit,
        "max_id": args.max_id_limit,
        "depth": args.depth_limit,
        "memory": args.mem_limit,
        "individual_memory": args.individual_mem_limit,
        "batch_pass_seconds": args.batch_pass_limit,
    },
)
limiter.set_memory_checkpoint()
print(f"Memory checkpoint: {limiter.memory_checkpoint} MB")

# create the grammar
grammar = PCFG(
    grammar=grammars[args.search_space],
    limiter=limiter,
)
print(grammar)

eval_fn = partial(
    evaluation_fn,
    args=args,
    train_loader=train_loader,
    val_loader=val_loader,
)

print("Environment ready")

def load_results(path):
    #full_path = join(path, "search_results.pkl")
    if not exists(path):
        raise FileNotFoundError(f"No checkpoint found at {path}")
    results = load(open(path, "rb"))
    print(f"Successfully loaded {results['iteration'] + 1} result iterations")
    return results

res = load_results(args_ro.path)

rewards = np.array([r[1] for r in res['rewards']])
struct = [r[0] for r in res['rewards']]

fig = plt.figure(figsize=(10, 5))
sns.scatterplot(x=range(len(rewards)), y=rewards)
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.title('Reward over iterations')
plt.savefig(join(args_ro.output_path, 'reward_over_iterations.png'))

thres = 0.005

chosen = []
chosen_idx = []

curr_best = -1
for idx, reward in enumerate(rewards):
    if idx < 100:
        continue
    if reward - curr_best > thres:
        curr_best = reward
        chosen.append(reward)
        chosen_idx.append(idx)
    elif reward > curr_best:
        curr_best = reward
        chosen[-1] = reward
        chosen_idx[-1] = idx
    
fig = plt.figure(figsize=(10, 5))
sns.scatterplot(x=chosen_idx, y=chosen)
sns.lineplot(x=chosen_idx, y=chosen)
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.savefig(join(args_ro.output_path, 'reward_over_iterations_chosen.png'))
plt.savefig(join(args_ro.output_path, 'reward_over_iterations_chosen.pdf'))

print(f"Number of chosen models: {len(chosen)}")

eval_res = []
for idx in tqdm(chosen_idx):
    can = struct[idx]
    score = eval_fn(can[0])
    eval_res.append(score)

    res = {
        'chosen_idx': chosen_idx,
        'eval_res': eval_res,
        'chosen': chosen,
        'best_idx': chosen_idx[np.argmax(eval_res)],
    }

    with open(join(args_ro.output_path, 'eval_res.json'), 'w') as f:
        json.dump(res, f, indent=4)

fig = plt.figure(figsize=(10, 5))
sns.lineplot(x=chosen_idx, y=chosen, label='Bert Reward')
sns.scatterplot(x=chosen_idx, y=chosen)
sns.scatterplot(x=chosen_idx, y=eval_res)
sns.lineplot(x=chosen_idx, y=eval_res, label='Evaluation')
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.legend()
plt.savefig(join(args_ro.output_path, 'reward_over_iterations_chosen_eval.png'))
plt.savefig(join(args_ro.output_path, 'reward_over_iterations_chosen_eval.pdf'))