import json
from functools import partial
from pprint import pprint
from os.path import join
from pickle import load, dump
from csv import writer as csv

import torch

from search_strategies import create_search_strategy
from pcfg import PCFG
from grammars import grammars
from network import Network
from evaluation import evaluation_fn
from arguments import parse_arguments
from data import get_data_loaders
from utils import load_config, get_exp_path, Limiter
from functools import partial
from main import set_seed


# parse the arguments
args = parse_arguments()
args = load_config(args)
pprint(vars(args))

# set the seed
# torch.manual_seed(args.seed)
set_seed(args.seed)

# get data loaders
_, _, trainval_loader, test_loader = get_data_loaders(
    dataset=args.dataset,
    batch_size=args.batch_size,
    image_size=args.image_size,
    root="../einspace/data",
    load_in_gpu=args.load_in_gpu,
    device=args.device,
    log=args.verbose_eval,
    seed=args.seed,
)

# get batch for batch pass time limiting
for batch in trainval_loader:
    batch = batch[0].to(args.device)
    break

def compile_fn(node, args):
    backbone = node.build(node, set_memory_checkpoint=True)
    return Network(
        backbone,
        node.output_params["shape"],
        args.num_classes,
        vars(args)
    ).to(args.device)

# create the limiter
# this makes sure that the search does not exceed
# time, memory, depth, or node limits during the search
limiter = Limiter(
    limits={
        "time": args.time_limit,
        "max_id": args.max_id_limit,
        "depth": args.depth_limit,
        "memory": args.mem_limit,
        "individual_memory": args.individual_mem_limit,
        "batch_pass_seconds": args.batch_pass_limit,
    }
)
limiter.set_memory_checkpoint()
print(f"Memory checkpoint: {limiter.memory_checkpoint} MB")

# create the grammar
grammar = PCFG(
    grammar=grammars[args.search_space],
    limiter=limiter,
)
print(grammar)

args.epochs = args.test_epochs
eval_fn = partial(
    evaluation_fn,
    args=args,
    train_loader=trainval_loader,
    val_loader=test_loader,
)

# load the search results (as a list of tuples: (architecture, reward, duration, timestamp))
search_results = load(open(join(
    args.results_path,
    "search_results.pkl"
), "rb"))

eval_res = json.loads(open(join(
    args.results_path,
    "eval_res.json"
), "r").read())
best_index = eval_res["best_idx"]
best_val_score = max(eval_res["eval_res"])

# get the best architecture, sorted by "accuracy"
best_architecture, _, _, _, _ = search_results["rewards"][best_index]

# evaluate the best architecture
limiter.timer.start()
test_score = eval_fn(best_architecture[0])
eval_duration = limiter.timer()
print(f"Test score: {test_score:.4f} (val score of {best_val_score:.4f}) - Time: {eval_duration:.2f}s")

# save the best architecture and its evaluation in csv format
with open(join(args.results_path, "best_architecture.csv"), "w") as f:
    csv_writer = csv(f)
    csv_writer.writerow(["Architecture", "Accuracy", "Time"])
    csv_writer.writerow([best_architecture[0], test_score, eval_duration])
