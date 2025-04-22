import torch
import numpy as np
from foresight.pruners.predictive import find_measures
from foresight.pruners.measures import available_measures
from network import Network
from surrogates.encodings.base import BaseEncoder

DEF_ZCPS = ['grad_norm', 'snip', 'grasp', 'fisher', 'jacob_cov', 'plain', 'synflow']

class ZCPEncoder(BaseEncoder):
    def __init__(self, data_loader, num_classes, args, device, zcps=None, return_names=True):
        super().__init__()

        train_loader, val_loader, _, _ = get_data_loaders(
            dataset=args.dataset,
            batch_size=32,
            image_size=args.image_size,
            root="../einspace/data",
            load_in_gpu=args.load_in_gpu,
            device=args.device,
            log=args.verbose_eval,
            seed=args.seed
        )

        self.num_classes = num_classes
        self.config = vars(args)
        self.device = device
        self.zcps = zcps if zcps is not None else DEF_ZCPS
        self.return_names = return_names
    
    def encode_individual(self, individual):
        node = individual.root
         # build the network
        backbone = node.build(node, set_memory_checkpoint=True)
        model = Network(
            backbone,
            node.output_params["shape"],
            self.num_classes,
            self.config,
        )
        
        measures = find_measures(
            model,
            self.data_loader,
            ("random", 1, self.num_classes),
            device=self.device,
            measure_names=self.zcps,
        )
        if 'jacob_cov' in measures and isinstance(measures['jacob_cov'], np.complex128):
            measures['jacob_cov'] = measures['jacob_cov'].real

        def _get_val(measure):
            if isinstance(measure, torch.Tensor):
                return measure.cpu().item()
            return measure

        # get encoding
        return  {k: _get_val(measures[k]) for k in self.zcps} if self.return_names else [_get_val(measures[k]) for k in self.zcps]
