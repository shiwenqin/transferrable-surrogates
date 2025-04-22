from functools import partial
from pprint import pprint
import random

import os
import torch
import random
import numpy as np
import psutil

from search_strategies import create_search_strategy
from pcfg import PCFG
from grammars import grammars
from network import Network
from evaluation import evaluation_fn
from arguments import parse_arguments
from data import get_data_loaders
from utils import load_config, Limiter
from functools import partial


def compile_fn(node, args):
    print(f"Before build: {psutil.Process().memory_info().rss / (1024 * 1024)}")
    backbone = node.build(node, set_memory_checkpoint=True)
    print(f"After build: {psutil.Process().memory_info().rss / (1024 * 1024)}")

    return Network(
        backbone,
        node.output_params["shape"],
        args.num_classes,
        vars(args)
    ).to(args.device)

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.use_deterministic_algorithms(True, warn_only=True)
    #torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # parse the arguments
    args = parse_arguments()
    args = load_config(args)
    pprint(vars(args))

    # set the seed
    set_seed(args.seed)

    if args.first_gen_path is not None:
        gen_path, ext = os.path.splitext(args.first_gen_path)
        args.first_gen_path = f"{gen_path}_pop_{args.population_size}.pkl"


    # get data loaders
    train_loader, val_loader, _, _ = get_data_loaders(
        dataset=args.dataset,
        batch_size=args.batch_size,
        image_size=args.image_size,
        root="../einspace/data",
        load_in_gpu=args.load_in_gpu,
        device=args.device,
        log=args.verbose_eval,
        seed=args.seed
    )

    # get batch for batch pass time limiting
    for batch in train_loader:
        batch = batch[0].to(args.device)
        break

    # create the limiter
    # this makes sure that the search does not exceed
    # time, memory (GPU and RAM), depth, or node limits during the search
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

    eval_fn = partial(
        evaluation_fn,
        args=args,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # create the input parameters
    input_params = {
        "shape": torch.Size([1, args.channels, *args.image_size]),
        "other_shape": None,
        "mode": "im",
        "other_mode": None,
        "branching_factor": 1,
        "last_im_shape": None,
    }

    # create the search strategy
    search = create_search_strategy(args, grammar, eval_fn, limiter, input_params, train_loader, compile_fn=partial(compile_fn, args=args), batch=batch)

    # run the search
    search.learn(steps=args.steps)
