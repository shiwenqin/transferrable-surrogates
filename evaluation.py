import torch

from network import Network
from trainers import Trainer


def random_evaluation_fn(node, args, train_loader, val_loader):
    return torch.rand(1).item()

def evaluation_fn(node, args, train_loader, val_loader):
    # build the network
    backbone = node.build(node, set_memory_checkpoint=True)
    model = Network(
        backbone,
        node.output_params["shape"],
        args.num_classes,
        config=vars(args),
    )
    # train and evaluate sampled network
    try:
        trainer = Trainer(
            model,
            device=args.device,
            train_dataloader=train_loader,
            valid_dataloader=val_loader,
            test_dataloader=None,
            config=vars(args),
            log=args.verbose_eval,
        )
        best = trainer.train()
    except RuntimeError as e:
        print(f"Trainer error: {e}")
        return 0.0
    return best["val_score"]
