import math
from copy import deepcopy

import torch
import torch.nn as nn

from search_state import Operation
import layers


def inherit_first_child(node):
    node.input_params = deepcopy(node.parent.input_params)

def inherit_other_child(node):
    node.input_params = deepcopy(node.parent.output_params)

def inherit_branching(node):
    node.input_params = deepcopy(node.parent.children[0].output_params)

def inherit_aggregation(node):
    if len(node.parent.children) >= 4:
        node.input_params = deepcopy(node.parent.children[1].output_params)
        node.input_params["other_shape"] = node.parent.children[2].output_params["shape"]
        node.input_params["other_mode"] = node.parent.children[2].output_params["mode"]
    elif len(node.parent.children) >= 3:
        node.input_params = deepcopy(node.parent.children[1].output_params)
        node.input_params["other_shape"] = node.parent.children[1].output_params["shape"]
        node.input_params["other_mode"] = node.parent.children[1].output_params["mode"]

def give_back_default(node):
    # print(f"Give back {node} with input params {node.input_params} and output params {node.output_params}")
    node.parent.output_params = node.output_params

# modules
def build_sequential_module(node):
    return layers.SequentialModule(
        first_fn=node.children[0].build(node.children[0]),
        second_fn=node.children[1].build(node.children[1])
    )

def infer_sequential_module(node):
    return node.input_params

def valid_sequential_module(node):
    return True

sequential_module = Operation(
    name="sequential",
    build=build_sequential_module,
    infer=infer_sequential_module,
    valid=valid_sequential_module,
    inherit = [
        inherit_first_child,
        inherit_other_child
    ],
    give_back = [
        give_back_default,
        give_back_default
    ],
    type="nonterminal",
    child_levels=[
        "module",
        "module"
    ],
)

def build_sequential_module_k(node, k):
    print(f"Building sequential({k})")
    # infer the output params of the whole subtree, recursively
    def infer(node):
        if not node.is_root():
            node.inherit_input_params()
        node.output_params = node.operation.infer(node)
        for child in node.children:
            infer(child)
        node.give_back_output_params()
        print(f"Inferred {node} with input params {node.input_params} and output params {node.output_params}")

    modules = []
    # store the original input params
    if not node.is_root():
        node.inherit_input_params()
    original_input_params = node.input_params
    for _ in range(k):
        infer(node.children[0])
        m = node.children[0].build(node.children[0])
        modules.append(m)
        # pass output params back in
        node.input_params = node.output_params
    print(modules)
    # reset input params
    node.input_params = original_input_params
    return layers.SequentialModule4(
        fns=modules,
    )

def build_sequential_module_4(node):
    return build_sequential_module_k(node, 4)

def infer_sequential_module_4(node):
    return node.input_params

def valid_sequential_module_4(node):
    return True

sequential_module_4 = Operation(
    name="sequential(4)",
    build=build_sequential_module_4,
    infer=infer_sequential_module_4,
    valid=valid_sequential_module_4,
    inherit = [
        inherit_first_child
    ],
    give_back = [
        give_back_default
    ],
    type="nonterminal",
    child_levels=[
        "module"
    ],
)

def build_sequential_module_8(node):
    return build_sequential_module_k(node, 8)

def infer_sequential_module_8(node):
    return node.input_params

def valid_sequential_module_8(node):
    return True

sequential_module_8 = Operation(
    name="sequential(8)",
    build=build_sequential_module_8,
    infer=infer_sequential_module_8,
    valid=valid_sequential_module_8,
    inherit = [
        inherit_first_child
    ],
    give_back = [
        give_back_default
    ],
    type="nonterminal",
    child_levels=[
        "module"
    ],
)

def build_branching_module_2(node):
    return layers.BranchingModule(
        branching_fn=node.children[0].build(node.children[0]),
        inner_fn=[
            node.children[1].build(node.children[1]),
            node.children[2].build(node.children[2]),
        ],
        aggregation_fn=node.children[3].build(node.children[3])
    )

def infer_branching_module_2(node):
    return node.input_params

def valid_branching_module_2(node):
    return True

branching_module_2 = Operation(
    name="branching(2)",
    build=build_branching_module_2,
    infer=infer_branching_module_2,
    valid=valid_branching_module_2,
    inherit=[
        inherit_first_child,
        inherit_branching,
        inherit_branching,
        inherit_aggregation
    ],
    give_back = [
        give_back_default,
        give_back_default,
        give_back_default,
        give_back_default
    ],
    type="nonterminal",
    child_levels=[
        "branching_fn_2",
        "module",
        "module",
        "aggregation_fn_2"
    ],
)

def build_branching_module_4(node):
    return layers.BranchingModule(
        branching_fn=node.children[0].build(node.children[0]),
        inner_fn=[
            node.children[1].build(node.children[1]) for _ in range(node.children[1].input_params["branching_factor"])
        ],
        aggregation_fn=node.children[2].build(node.children[2])
    )

def infer_branching_module_4(node):
    return node.input_params

def valid_branching_module_4(node):
    return True

branching_module_4 = Operation(
    name="branching(4)",
    build=build_branching_module_4,
    infer=infer_branching_module_4,
    valid=valid_branching_module_4,
    inherit=[
        inherit_first_child,
        inherit_branching,
        inherit_aggregation
    ],
    give_back = [
        give_back_default,
        give_back_default,
        give_back_default
    ],
    type="nonterminal",
    child_levels=[
        "branching_fn_4",
        "module",
        "aggregation_fn_4"
    ],
)

def build_branching_module_8(node):
    return layers.BranchingModule(
        branching_fn=node.children[0].build(node.children[0]),
        inner_fn=[
            node.children[1].build(node.children[1]) for _ in range(node.children[1].input_params["branching_factor"])
        ],
        aggregation_fn=node.children[2].build(node.children[2])
    )

def infer_branching_module_8(node):
    return node.input_params

def valid_branching_module_8(node):
    return True

branching_module_8 = Operation(
    name="branching(8)",
    build=build_branching_module_8,
    infer=infer_branching_module_8,
    valid=valid_branching_module_8,
    inherit=[
        inherit_first_child,
        inherit_branching,
        inherit_aggregation
    ],
    give_back = [
        give_back_default,
        give_back_default,
        give_back_default
    ],
    type="nonterminal",
    child_levels=[
        "branching_fn_8",
        "module",
        "aggregation_fn_8"
    ],
)

def build_routing_module(node):
    return layers.RoutingModule(
        prerouting_fn=node.children[0].build(node.children[0]),
        inner_fn=node.children[1].build(node.children[1]),
        postrouting_fn=node.children[2].build(node.children[2])
    )

def infer_routing_module(node):
    return node.input_params

def valid_routing_module(node):
    return True

routing_module = Operation(
    name="routing",
    build=build_routing_module,
    infer=infer_routing_module,
    valid=valid_routing_module,
    inherit=[
        inherit_first_child,
        inherit_other_child,
        inherit_other_child
    ],
    give_back = [
        give_back_default,
        give_back_default,
        give_back_default
    ],
    type="nonterminal",
    child_levels=[
        "prerouting_fn",
        "module",
        "postrouting_fn"
    ],
)

def build_computation_module(node):
    return layers.ComputationModule(
        computation_fn=node.children[0].build(node.children[0])
    )

def infer_computation_module(node):
    return node.input_params

def valid_computation_module(node):
    return True

computation_module = Operation(
    name="computation",
    build=build_computation_module,
    infer=infer_computation_module,
    valid=valid_computation_module,
    inherit=[inherit_first_child],
    give_back = [give_back_default],
    type="nonterminal",
    child_levels=["computation_fn"],
)

from functools import partial

def build_clone(branching_factor, node):
    return layers.CloneTensor(num_clones=branching_factor)

def infer_clone(branching_factor, node):
    return {
        "shape": node.input_params["shape"],
        "other_shape": node.input_params["other_shape"],
        "mode": node.input_params["mode"],
        "other_mode": node.input_params["other_mode"],
        "branching_factor": branching_factor,
        "last_im_shape": node.input_params["last_im_shape"],
    }

def valid_clone(node):
    return True

# branching functions
def clone(branching_factor):
    return Operation(
        name=f"clone({branching_factor})",
        build=partial(build_clone, branching_factor),
        infer=partial(infer_clone, branching_factor),
        valid=valid_clone,
        inherit=[inherit_first_child],
        give_back = [give_back_default],
        type="terminal",
        child_levels=[],
    )

def build_group(branching_factor, dim, node):
    return layers.GroupDim(splits=branching_factor, dim=dim, dim_total=node.input_params["shape"][dim])

def infer_group(branching_factor, dim, node):
    shape = torch.Size(
        list(node.input_params["shape"][:dim]) + 
        [node.input_params["shape"][dim] // branching_factor] + 
        list(node.input_params["shape"][dim + 1:])
    )
    return {
        "shape": shape,
        "other_shape": node.input_params["other_shape"],
        "mode": node.input_params["mode"],
        "other_mode": node.input_params["other_mode"],
        "branching_factor": branching_factor,
        "last_im_shape": node.input_params["last_im_shape"],
    }

def valid_group(branching_factor, dim, node):
    return (
        len(node.input_params["shape"]) > dim and
        node.input_params["shape"][dim] > 0 and
        node.input_params["shape"][dim] >= branching_factor and
        node.input_params["shape"][dim] % branching_factor == 0
    )

def group(branching_factor, dim):
    return Operation(
        name=f"group({branching_factor},{dim})",
        build=partial(build_group, branching_factor, dim),
        infer=partial(infer_group, branching_factor, dim),
        valid=partial(valid_group, branching_factor, dim),
        inherit=[inherit_first_child],
        give_back = [give_back_default],
        type="terminal",
        child_levels=[],
    )

# aggregation functions
def build_add(node):
    return layers.AddTensors()

def infer_add(node):
    return {
        "shape": node.input_params["shape"],
        "other_shape": node.parent.input_params["other_shape"],
        "mode": node.input_params["mode"],
        "other_mode": node.parent.input_params["other_mode"],
        "branching_factor": node.parent.input_params["branching_factor"],
        "last_im_shape": node.input_params["last_im_shape"],
    }

def valid_add(node):
    return (
        node.input_params["other_shape"] != None and
        torch.equal(
            torch.tensor(node.input_params["shape"]),
            torch.tensor(node.input_params["other_shape"])
        )
    )

def add(branching_factor):
    return Operation(
        name=f"add({branching_factor})",
        build=build_add,
        infer=infer_add,
        valid=valid_add,
        inherit=[inherit_first_child],
        give_back = [give_back_default],
        type="terminal",
        child_levels=[],
    )

def build_cat(dim, node):
    return layers.CatTensors(dim=dim)

def infer_cat(branching_factor, dim, node):
    if branching_factor == 2:
        shape = torch.Size(
            list(node.input_params["shape"][:dim]) + 
            [node.input_params["shape"][dim] + node.input_params["other_shape"][dim]] + 
            list(node.input_params["shape"][dim + 1:])
        )
    else:
        shape = torch.Size(
            list(node.input_params["shape"][:dim]) + 
            [node.input_params["shape"][dim] * branching_factor] + 
            list(node.input_params["shape"][dim + 1:])
        )
    return {
        "shape": shape,
        "other_shape": node.parent.input_params["other_shape"],
        "mode": node.input_params["mode"],
        "other_mode": node.parent.input_params["other_mode"],
        "branching_factor": node.parent.input_params["branching_factor"],
        "last_im_shape": node.input_params["last_im_shape"],
    }

def valid_cat(dim, node):
    return (
        len(node.input_params["shape"]) > dim and
        len(node.input_params["other_shape"]) > dim and
        node.input_params["other_shape"] != None and
        torch.equal(
            torch.tensor(node.input_params["shape"][:dim] + node.input_params["shape"][dim + 1:]),
            torch.tensor(node.input_params["other_shape"][:dim] + node.input_params["other_shape"][dim + 1:])
        )
    )

def cat(branching_factor, dim):
    return Operation(
        name=f"cat({branching_factor},{dim})",
        build=partial(build_cat, dim),
        infer=partial(infer_cat, branching_factor, dim),
        valid=partial(valid_cat, dim),
        inherit=[inherit_first_child],
        give_back = [give_back_default],
        type="terminal",
        child_levels=[],
    )

def build_dot_product(scaled, node):
    return layers.DotProduct(scaled=scaled)

def infer_dot_product(node):
    # computes the output shape of the matrix multiplication of two tensors of shapes (B, N, D1) and (B, D2, N)
    # the output shape is (B, D1, D2)
    shape = torch.Size(
        list(node.input_params["shape"][:-1]) +
        [node.input_params["other_shape"][-1]]
    )
    return {
        "shape": shape,
        "other_shape": node.parent.input_params["other_shape"],
        "mode": node.input_params["mode"],
        "other_mode": node.parent.input_params["other_mode"],
        "branching_factor": node.parent.input_params["branching_factor"],
        "last_im_shape": node.input_params["last_im_shape"],
    }

def valid_dot_product(node):
    return (
        node.input_params["other_shape"] != None and
        node.input_params["shape"][-1] == node.input_params["other_shape"][-2] and
        torch.equal(
            torch.tensor(node.input_params["shape"][:-2]),
            torch.tensor(node.input_params["other_shape"][:-2])
        )
    )

def dot_product(scaled):
    return Operation(
        name=f"dot_product{'(scaled)' if scaled else ''}",
        build=partial(build_dot_product, scaled),
        infer=infer_dot_product,
        valid=valid_dot_product,
        inherit=[inherit_first_child],
        give_back = [give_back_default],
        type="terminal",
        child_levels=[],
    )

def build_broadcast(node):
    return layers.BroadcastTensors(mode="add")

def infer_broadcast(node):
    a_shape, b_shape = node.input_params["shape"], node.input_params["other_shape"]
    m = build_broadcast(node)
    shape = m([torch.randn(*a_shape), torch.randn(*b_shape)]).shape
    mode = {3: "col", 4: "im"}[len(shape)]
    return {
        "shape": shape,
        "other_shape": node.parent.input_params["other_shape"],
        "mode": mode,
        "other_mode": node.parent.input_params["other_mode"],
        "branching_factor": node.parent.input_params["branching_factor"],
        "last_im_shape": node.input_params["last_im_shape"],
    }

def valid_broadcast(node):
    return True

def broadcast(branching_factor):
    return Operation(
        name=f"broadcast({branching_factor})",
        build=build_broadcast,
        infer=infer_broadcast,
        valid=valid_broadcast,
        inherit=[inherit_first_child],
        give_back = [give_back_default],
        type="terminal",
        child_levels=[],
    )

def build_identity(node):
    return nn.Identity()

def infer_identity(node):
    return node.input_params

def valid_identity(node):
    return True

identity = Operation(
    name="identity",
    build=build_identity,
    infer=infer_identity,
    valid=valid_identity,
    inherit=[inherit_first_child],
    give_back = [give_back_default],
    type="terminal",
    child_levels=[],
)

def build_permute(perm, node):
    return layers.Permute(perm)

def infer_permute(perm, node):
    return {
        "shape": torch.Size(torch.tensor(node.input_params["shape"])[perm]),
        "other_shape": node.input_params["other_shape"],
        "mode": node.input_params["mode"],
        "other_mode": node.input_params["other_mode"],
        "branching_factor": node.input_params["branching_factor"],
        "last_im_shape": node.input_params["last_im_shape"],
    }

def valid_permute(perm, node):
    return len(node.input_params["shape"]) == len(perm)

# prerouting functions
def permute(perm):
    return Operation(
        name=f"perm({','.join(map(str,perm))})",
        build=partial(build_permute, perm),
        infer=partial(infer_permute, perm),
        valid=partial(valid_permute, perm),
        inherit=[inherit_first_child],
        give_back = [give_back_default],
        type="terminal",
        child_levels=[],
    )

def build_im2col(kernel_size, stride, padding, node):
    return layers.Im2Col(
        input_shape=node.input_params["shape"],
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )

def infer_im2col(kernel_size, stride, padding, node):
    batch_size, channels, height, width = node.input_params["shape"]

    output_height = (height + 2 * padding - kernel_size) // stride + 1
    output_width = (width + 2 * padding - kernel_size) // stride + 1

    patch_size = output_height * output_width
    flattened_patch_size = kernel_size * kernel_size * channels

    shape = torch.Size([batch_size, patch_size, flattened_patch_size])
    last_im_shape = (output_height, output_width)

    return {
        "shape": shape,
        "other_shape": node.input_params["other_shape"],
        "mode": "col",
        "other_mode": node.input_params["other_mode"],
        "branching_factor": node.input_params["branching_factor"],
        "last_im_shape": last_im_shape,
    }

def valid_im2col(kernel_size, node):
    return (
        len(node.input_params["shape"]) == 4 and
        node.input_params["mode"] == "im" and
        node.input_params["shape"][-2] >= kernel_size and
        node.input_params["shape"][-1] >= kernel_size
    )

def im2col(kernel_size, stride, padding):
    return Operation(
        name=f"im2col({kernel_size},{stride},{padding})",
        build=partial(build_im2col, kernel_size, stride, padding),
        infer=partial(infer_im2col, kernel_size, stride, padding),
        valid=partial(valid_im2col, kernel_size),
        inherit=[inherit_first_child],
        give_back = [give_back_default],
        type="terminal",
        child_levels=[],
    )

def build_col2im(node):
    m = layers.Col2Im()
    m.output_shape = node.input_params["last_im_shape"]
    return m

def infer_col2im(node):
    batch_size, _, channels = node.input_params["shape"]
    output_height, output_width = node.input_params["last_im_shape"]
    shape = torch.Size([batch_size, channels, output_height, output_width])

    return {
        "shape": shape,
        "other_shape": node.input_params["other_shape"],
        "mode": "col",
        "other_mode": node.input_params["other_mode"],
        "branching_factor": node.input_params["branching_factor"],
        "last_im_shape": node.input_params["last_im_shape"],
    }

def valid_col2im(node):
    return (
        len(node.input_params["shape"]) == 3 and
        node.input_params["mode"] == "col" and
        node.input_params["last_im_shape"] != None and
        node.input_params["shape"][1] == torch.prod(torch.tensor(node.input_params["last_im_shape"]))
    )

col2im = Operation(
    name="col2im",
    build=build_col2im,
    infer=infer_col2im,
    valid=valid_col2im,
    inherit=[inherit_first_child],
    give_back = [give_back_default],
    type="terminal",
    child_levels=[],
)

# computation functions

def build_linear(dim, node):
    return layers.EinLinear(node.input_params["shape"][-1], dim)

def infer_linear(dim, node):
    return {
        "shape": torch.Size(
            list(node.input_params["shape"][:-1]) +
            [dim]
        ),
        "other_shape": node.input_params["other_shape"],
        "mode": node.input_params["mode"],
        "other_mode": node.input_params["other_mode"],
        "branching_factor": node.input_params["branching_factor"],
        "last_im_shape": node.input_params["last_im_shape"],
        "num_params": node.input_params["shape"][-1] * dim + dim,
    }

def valid_linear(node):
    return node.input_params["shape"][-1] > 0

def linear(dim):
    return Operation(
        name=f"linear({dim})",
        build=partial(build_linear, dim),
        infer=partial(infer_linear, dim),
        valid=valid_linear,
        inherit=[inherit_first_child],
        give_back = [give_back_default],
        type="terminal",
        child_levels=[],
    )

def build_linear_x(dim_factor, node):
    out_dim = int(node.input_params["shape"][-1] * dim_factor)
    return layers.EinLinear(node.input_params["shape"][-1], out_dim)

def infer_linear_x(dim_factor, node):
    out_dim = int(node.input_params["shape"][-1] * dim_factor)
    return {
        "shape": torch.Size(
            list(node.input_params["shape"][:-1]) + [out_dim]
        ),
        "other_shape": node.input_params["other_shape"],
        "mode": node.input_params["mode"],
        "other_mode": node.input_params["other_mode"],
        "branching_factor": node.input_params["branching_factor"],
        "last_im_shape": node.input_params["last_im_shape"],
        "num_params": (
            node.input_params["shape"][-1] *  out_dim + out_dim
        ),
    }

def valid_linear_x(dim_factor, node):
    out_dim = int(node.input_params["shape"][-1] * dim_factor)
    return out_dim > 0 and node.input_params["shape"][-1] > 0

def linear_x(dim_factor):
    return Operation(
        name=f"linear(x{dim_factor})",
        build=partial(build_linear_x, dim_factor),
        infer=partial(infer_linear_x, dim_factor),
        valid=partial(valid_linear_x, dim_factor),
        inherit=[inherit_first_child],
        give_back = [give_back_default],
        type="terminal",
        child_levels=[],
    )

def build_norm(node):
    return layers.EinNorm(node.input_params["shape"])

def infer_norm(node):
    output_params = node.input_params
    output_params.update({"num_params": 2 * node.input_params["shape"][1]})
    return output_params

def valid_norm(node):
    return True

norm = Operation(
    name="norm",
    build=build_norm,
    infer=infer_norm,
    valid=valid_norm,
    inherit=[inherit_first_child],
    give_back = [give_back_default],
    type="terminal",
    child_levels=[],
)

def build_relu(node):
    return nn.ReLU()

def infer_relu(node):
    return node.input_params

def valid_relu(node):
    return True

relu = Operation(
    name="relu",
    build=build_relu,
    infer=infer_relu,
    valid=valid_relu,
    inherit=[inherit_first_child],
    give_back = [give_back_default],
    type="terminal",
    child_levels=[],
)

def build_softmax(node):
    return nn.Softmax(dim=-1)

def infer_softmax(node):
    return node.input_params

def valid_softmax(node):
    return True

softmax = Operation(
    name="softmax",
    build=build_softmax,
    infer=infer_softmax,
    valid=valid_softmax,
    inherit=[inherit_first_child],
    give_back = [give_back_default],
    type="terminal",
    child_levels=[],
)

def build_positional_encoding(node):
    return layers.LearnablePositionalEncoding(node.input_params["shape"])

def infer_positional_encoding(node):
    output_params = node.input_params
    output_params.update({"num_params": math.prod(node.input_params["shape"][1:])})
    return output_params

def valid_positional_encoding(node):
    return True

positional_encoding = Operation(
    name="pos_enc",
    build=build_positional_encoding,
    infer=infer_positional_encoding,
    valid=valid_positional_encoding,
    inherit=[inherit_first_child],
    give_back = [give_back_default],
    type="terminal",
    child_levels=[],
)


# grammar definition
modules_without_computation_module = {
    "options": [
        sequential_module,
        branching_module_2,
        branching_module_4,
        branching_module_8,
        routing_module,
    ],
    "probs": [
        0.333,
        0.111,
        0.111,
        0.111,
        0.333,
    ],
}

deep_modules_without_computation_module = {
    "options": [
        sequential_module,
        sequential_module_4,
        sequential_module_8,
        branching_module_2,
        branching_module_4,
        branching_module_8,
        routing_module,
    ],
    "probs": [
        0.111,
        0.111,
        0.111,
        0.111,
        0.111,
        0.111,
        0.333,
    ],
}

modules = lambda prob: {
    "options": [
        sequential_module,
        branching_module_2,
        branching_module_4,
        branching_module_8,
        routing_module,
        computation_module,
    ],
    "probs": [
        (1 - prob) / 3,
        (1 - prob) / 9,
        (1 - prob) / 9,
        (1 - prob) / 9,
        (1 - prob) / 3,
        prob,
    ],
}

deep_modules = lambda prob: {
    "options": [
        sequential_module,
        sequential_module_4,
        sequential_module_8,
        branching_module_2,
        branching_module_4,
        branching_module_8,
        routing_module,
        computation_module,
    ],
    "probs": [
        (1 - prob) / 9,
        (1 - prob) / 9,
        (1 - prob) / 9,
        (1 - prob) / 9,
        (1 - prob) / 9,
        (1 - prob) / 9,
        (1 - prob) / 3,
        prob,
    ],
}

branching_fns_2 = {
    "options": [
        clone(2),
        group(2, 1),
        group(2, 2),
        group(2, 3),
    ],
    "probs": [
        0.25,
        0.25,
        0.25,
        0.25,
    ],
}

branching_fns_4 = {
    "options": [
        clone(4),
        group(4, 1),
        group(4, 2),
        group(4, 3),
    ],
    "probs": [
        0.25,
        0.25,
        0.25,
        0.25,
    ],
}

branching_fns_8 = {
    "options": [
        clone(8),
        group(8, 1),
        group(8, 2),
        group(8, 3),
    ],
    "probs": [
        0.25,
        0.25,
        0.25,
        0.25,
    ],
}

aggregation_fns_2 = {
    "options": [
        add(2),
        cat(2, 1),
        cat(2, 2),
        cat(2, 3),
        dot_product(scaled=False),
        dot_product(scaled=True),
    ],
    "probs": [
        0.166,
        0.166,
        0.166,
        0.166,
        0.166,
        0.166,
    ],
}

aggregation_fns_4 = {
    "options": [
        add(4),
        cat(4, 1),
        cat(4, 2),
        cat(4, 3),
    ],
    "probs": [
        0.25,
        0.25,
        0.25,
        0.25,
    ],
}

aggregation_fns_8 = {
    "options": [
        add(8),
        cat(8, 1),
        cat(8, 2),
        cat(8, 3),
    ],
    "probs": [
        0.25,
        0.25,
        0.25,
        0.25,
    ],
}

aggregation_fns_with_broadcast_2 = {
    "options": [
        add(2),
        cat(2, 1),
        cat(2, 2),
        cat(2, 3),
        dot_product(scaled=False),
        dot_product(scaled=True),
        broadcast(2),
    ],
    "probs": [
        0.143,
        0.143,
        0.143,
        0.143,
        0.143,
        0.143,
        0.143,
    ],
}

prerouting_fns = {
    "options": [
        identity,
        permute([0, 2, 1]),
        permute([0, 1, 3, 2]),
        permute([0, 3, 1, 2]),
        permute([0, 3, 2, 1]),
        permute([0, 2, 1, 3]),
        permute([0, 2, 3, 1]),
        im2col(1, 1, 0),
        im2col(1, 2, 0),
        im2col(3, 1, 1),
        im2col(3, 2, 1),
        # im2col(5, 1, 2),
        # im2col(7, 1, 3),
        # im2col(7, 2, 3),
        im2col(4, 4, 0),
        im2col(8, 8, 0),
        im2col(16, 16, 0),
    ],
    "probs": [
        0.071,
        0.071,
        0.071,
        0.071,
        0.071,
        0.071,
        0.071,
        0.071,
        0.071,
        0.071,
        0.071,
        0.071,
        0.071,
        0.071,
    ],
}

postrouting_fns = {
    "options": [
        identity,
        permute([0, 2, 1]),
        permute([0, 1, 3, 2]),
        permute([0, 3, 1, 2]),
        permute([0, 3, 2, 1]),
        permute([0, 2, 1, 3]),
        permute([0, 2, 3, 1]),
        col2im
    ],
    "probs": [
        0.125,
        0.125,
        0.125,
        0.125,
        0.125,
        0.125,
        0.125,
        0.125,
    ],
}

computation_fns = {
    "options": [
        identity,
        linear(16),
        linear(32),
        linear(64),
        linear(128),
        linear(256),
        linear(512),
        linear(1024),
        linear(2048),
        norm,
        relu,
        softmax,
        positional_encoding,
    ],
    "probs": [
        0.077,
        0.077,
        0.077,
        0.077,
        0.077,
        0.077,
        0.077,
        0.077,
        0.077,
        0.077,
        0.077,
        0.077,
        0.077,
    ],
}

grammar = {
    "network": modules_without_computation_module,
    "module": modules(0.32),
    "branching_fn_2": branching_fns_2,
    "branching_fn_4": branching_fns_4,
    "branching_fn_8": branching_fns_8,
    "aggregation_fn_2": aggregation_fns_2,
    "aggregation_fn_4": aggregation_fns_4,
    "aggregation_fn_8": aggregation_fns_8,
    "prerouting_fn": prerouting_fns,
    "postrouting_fn": postrouting_fns,
    "computation_fn": computation_fns,
}

quick_grammar = {
    "network": modules_without_computation_module,
    "module": modules(0.9),
    "branching_fn_2": branching_fns_2,
    "branching_fn_4": branching_fns_4,
    "branching_fn_8": branching_fns_8,
    "aggregation_fn_2": aggregation_fns_2,
    "aggregation_fn_4": aggregation_fns_4,
    "aggregation_fn_8": aggregation_fns_8,
    "prerouting_fn": prerouting_fns,
    "postrouting_fn": postrouting_fns,
    "computation_fn": computation_fns,
}

broadcast_grammar = {
    "network": modules_without_computation_module,
    "module": modules(0.32),
    "branching_fn_2": branching_fns_2,
    "branching_fn_4": branching_fns_4,
    "branching_fn_8": branching_fns_8,
    "aggregation_fn_2": aggregation_fns_with_broadcast_2,
    "aggregation_fn_4": aggregation_fns_4,
    "aggregation_fn_8": aggregation_fns_8,
    "prerouting_fn": prerouting_fns,
    "postrouting_fn": postrouting_fns,
    "computation_fn": computation_fns,
}

deep_broadcast_grammar = {
    "network": deep_modules_without_computation_module,
    "module": deep_modules(0.32),
    "branching_fn_2": branching_fns_2,
    "branching_fn_4": branching_fns_4,
    "branching_fn_8": branching_fns_8,
    "aggregation_fn_2": aggregation_fns_with_broadcast_2,
    "aggregation_fn_4": aggregation_fns_4,
    "aggregation_fn_8": aggregation_fns_8,
    "prerouting_fn": prerouting_fns,
    "postrouting_fn": postrouting_fns,
    "computation_fn": computation_fns,
}

quick_deep_broadcast_grammar = {
    "network": {
        "options": [
            sequential_module,
            sequential_module_4,
            sequential_module_8,
        ],
        "probs": [
            0.33,
            0.33,
            0.33
        ],
    },
    "module": {
        "options": [
            sequential_module,
            sequential_module_4,
            sequential_module_8,
            computation_module,
        ],
        "probs": [
            0.2,
            0.2,
            0.2,
            0.4,
        ],
    },
    "computation_fn": computation_fns,
}
