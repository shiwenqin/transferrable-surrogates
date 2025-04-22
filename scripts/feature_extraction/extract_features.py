#pyright: basic
from pprint import pprint
import pickle

from arguments import parse_arguments
from utils import load_config
from data import get_data_loaders

import surrogates.encodings as se

import sys

args = parse_arguments()
args = load_config(args)
args.device = 'cuda:0'
pprint(args)

train_loader, val_loader, _, _ = get_data_loaders(
    dataset=args.dataset,
    batch_size=args.batch_size,
    image_size=args.image_size,
    root="../data",
    load_in_gpu=args.load_in_gpu,
    device=args.device,
    log=args.verbose_eval,
)

import pandas as pd
import os
import os.path

args.use_features = True
args.use_zcp = True
encoder = se.get_encoder(data_loader=train_loader, args=args, return_names=True)

class ind:
    def __init__(self, x):
        self.root = x

for seed in ['seed=0', 'seed=1', 'seed=2', 'seed=3', 'seed=4', 'seed=5', 'seed=6', 'seed=7']:
    if not os.path.isfile(f'transfer_runs/data_{args.dataset}_{seed}.pkl'):
        continue
    individuals = pickle.load(open(f'transfer_runs/data_{args.dataset}_{seed}.pkl', 'rb'))
    output_file = f'{args.dataset}_{seed}.csv'
    out = pd.read_csv(output_file, index_col='id') if os.path.isfile(output_file) else pd.DataFrame()

    feats = []
    for i in range(len(individuals['rewards'])):
        if i in out.index: #allow for re-running in case of problems
            continue
        print(f'Processing {i}')
        ii = individuals['rewards'][i]
        try: 
            feats.append({'id':i} | encoder.encode(ind(ii[0][0])) | {'accuracy': ii[1]})
            out = pd.concat([out, pd.DataFrame(feats).set_index('id')])
            out.to_csv(output_file, index_label='id')
            feats = []
        except BaseException as e:
            print(f"Skipping {i} because of {e}")
