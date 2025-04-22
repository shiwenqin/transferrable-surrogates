from .resnet import resnet18_no_maxpool, resnet18_conv7x7_no_maxpool
from .wideresnet import wideresnet16_4
from .vit import vit_d2, vit_d4, vit_d8
from .mlpmixer import mlpmixer_d2, mlpmixer_d4, mlpmixer_d8

from search_strategies.random_search import Sampler
from grammars.einspace import *
from pcfg import PCFG
from utils import Limiter

import re


baseline_dict = {
    "resnet18": resnet18_no_maxpool,
    "wideresnet16_4": wideresnet16_4,
    "vit": vit_d4,
    "mlpmixer": mlpmixer_d8,
}


def build_baseline(baseline, input_params=None):
    """
    Take as input the baseline architecture string and return the architecture
    build the architecture with DerivationTreeNodes and return the root node

    e.g.
    baseline = "sequential(sequential(computation<linear64>, computation<relu>), computation<linear64>)"
    becomes
    architecture = [
        DerivationTreeNode(id=1, level="network", operation=Operation("sequential")),
        DerivationTreeNode(id=2, level="module", operation=Operation("sequential")),
        DerivationTreeNode(id=3, level="module", operation=Operation("computation")),
        DerivationTreeNode(id=4, level="computation", operation=Operation("linear64")),
        DerivationTreeNode(id=5, level="module", operation=Operation("computation")),
        DerivationTreeNode(id=6, level="computation", operation=Operation("relu")),
        DerivationTreeNode(id=7, level="module", operation=Operation("computation")),
        DerivationTreeNode(id=8, level="computation", operation=Operation("linear64")),
    ]
    """

    op_map = {
        "sequential": "sequential_module",
        "branching(2)": "branching_module_2",
        "branching(4)": "branching_module_4",
        "branching(8)": "branching_module_8",
        "routing": "routing_module",
        "computation": "computation_module",
        "im2col1k1s0p": "im2col(1, 1, 0)",
        "im2col1k2s0p": "im2col(1, 2, 0)",
        "im2col3k1s1p": "im2col(3, 1, 1)",
        "im2col3k2s1p": "im2col(3, 2, 1)",
        "im2col7k2s3p": "im2col(7, 2, 3)",
        "im2col4k4s0p": "im2col(4, 4, 0)",
        "im2col8k8s0p": "im2col(8, 8, 0)",
        "im2col16k16s0p": "im2col(16, 16, 0)",
        "permute21": "permute([0, 2, 1])",
        "permute132": "permute([0, 1, 3, 2])",
        "permute312": "permute([0, 3, 1, 2])",
        "permute321": "permute([0, 3, 2, 1])",
        "permute213": "permute([0, 2, 1, 3])",
        "permute231": "permute([0, 2, 3, 1])",
        "linear16": "linear(16)",
        "linear32": "linear(32)",
        "linear64": "linear(64)",
        "linear128": "linear(128)",
        "linear256": "linear(256)",
        "linear512": "linear(512)",
        "linear1024": "linear(1024)",
        "linear2048": "linear(2048)",
        "pos_enc": "positional_encoding"
    }

    # print(baseline)

    operations = []
    # first remove all whitespace
    baseline = re.sub(r"\s+", "", baseline)
    # print(baseline)
    # split into individual operations
    # by commas, and the following brackets []
    op_names = re.split(r",|\[|\]", baseline)
    op_names = [op.strip() for op in op_names if op.strip() != ""]
    # print(op_names)
    # fix operations with spaces in their hyperparams, e.g. 'cat(4, 2)'
    indices_to_remove = []
    for a, b in zip(range(len(op_names) - 1), range(1, len(op_names))):
        if "cat" in op_names[a]:
            op_names[a] = f"{op_names[a]}, {op_names[b]}"
            indices_to_remove.append(b)
    # print(indices_to_remove)
    op_names = [o for i, o in enumerate(op_names) if i not in indices_to_remove]
    # print(op_names)
    # print(len(op_names))
    
    for op_name in op_names:
        op = op_map[op_name] if op_name in op_map else op_name
        operation=eval(op)
        operations.append(operation)

    limiter = Limiter(
        limits={
            "time": 300,
            "max_id": 10000,
            "depth": 20,
            "memory": 8192,
            "individual_memory": 1024,
            "batch_pass_seconds": 0.1,
        }
    )
    pcfg = PCFG(
        grammar=grammar,
        limiter=limiter,
    )
    sampler = Sampler(
        pcfg,
        "iterative",
        verbose=False
    )
    limiter.timer.start()
    root = sampler.sample(input_params, operations)
    return root


# root = build_baseline(
#     resnet18_no_maxpool,
#     input_params={
#         "shape": torch.Size([1, 3, 32, 32]),
#         "other_shape": None,
#         "mode": "im",
#         "other_mode": None,
#         "branching_factor": 1,
#     }
# )

# print(root)
# print(root.build(root))
