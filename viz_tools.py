import os
import html

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from pyvis.network import Network
from IPython.display import HTML

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="ticks", context="paper")

#from einspace.search_spaces import EinSpace


colours = {
    "first_fn": '#6fa8dcff',        # light blue 1
    "second_fn": '#6fa8dcff',       # light blue 1
    "inner_fn": '#6fa8dcff',        # light blue 1
    "computation_fn": '#6fa8dcff',  # light blue 1
    "branching_fn": '#ffd966ff',    # light yellow 1
    "aggregation_fn": '#a64d79ff',  # dark magenta 1
    "prerouting_fn": '#93c47dff',   # light green 1
    "postrouting_fn": '#93c47dff',  # light green 1
    "input": '#333333ff',           # black
    "output": '#cc4125ff',          # light red berry 1
    "mutation": '#ff007fff',        # bright pink
    "module": '#999999ff',          # grey
    "root": '#8d3b2fff',            # hex for dark brown #3d2b1fff
}

def recurse_diff(node1, node2):
    if "fn" in node1 and "fn" in node2:
        if node1["fn"] != node2["fn"]:
            return (node1, node2)
    if "fn" in node1 and node1["fn"] == "sequential_module":
        return (
            recurse_diff(node1["children"]["first_fn"], node2["children"]["first_fn"]) or
            recurse_diff(node1["children"]["second_fn"], node2["children"]["second_fn"])
        )
    elif "fn" in node1 and node1["fn"] == "branching_module":
        diff = recurse_diff(node1["children"]["branching_fn"], node2["children"]["branching_fn"])
        for child1, child2 in zip(node1["children"]["inner_fn"], node2["children"]["inner_fn"]):
            diff = diff or recurse_diff(child1, child2)
        diff = diff or recurse_diff(node1["children"]["aggregation_fn"], node2["children"]["aggregation_fn"])
        return diff
    elif "fn" in node1 and node1["fn"] == "routing_module":
        return (
            recurse_diff(node1["children"]["prerouting_fn"], node2["children"]["prerouting_fn"]) or
            recurse_diff(node1["children"]["inner_fn"], node2["children"]["inner_fn"]) or
            recurse_diff(node1["children"]["postrouting_fn"], node2["children"]["postrouting_fn"])
        )
    elif "fn" in node1 and node1["fn"] == "computation_module":
        return recurse_diff(node1["children"]["computation_fn"], node2["children"]["computation_fn"])
    return None

def shape_to_string(shape):
    return f"[{', '.join(map(str, shape))}]"

def recurse_add_node(node, net, parents=[], parent_fn="", mutation_id=None, override_colour=None):
    if "node_type" in node and node["node_type"] == "terminal":
        if "node_id" in node and node["node_id"] == mutation_id:
            c = colours["mutation"]
        elif override_colour:
            c = override_colour
        else:
            c = colours[parent_fn]
    elif "node_type" in node and node["node_type"] == "nonterminal":
        if "node_id" in node and node["node_id"] == mutation_id:
            override_colour = colours["mutation"]
            c = override_colour

    if "node_type" in node and node["node_type"] == "terminal":
        net.add_node(node["node_id"], label=node["fn"], title=f"{node['node_id']}: {shape_to_string(node['output_shape'])}", color=c, level=node["depth"])
        if parents:
            for parent in parents:
                net.add_edge(parent, node["node_id"])
        return node["node_id"]
    elif "fn" in node and node["fn"] == "sequential_module":
        parent = recurse_add_node(node["children"]["first_fn"], net, parents=parents, parent_fn="first_fn", mutation_id=mutation_id, override_colour=override_colour)
        parent = recurse_add_node(node["children"]["second_fn"], net, parents=[parent], parent_fn="second_fn", mutation_id=mutation_id, override_colour=override_colour)
    elif "fn" in node and node["fn"] == "branching_module":
        parent = recurse_add_node(node["children"]["branching_fn"], net, parents=parents, parent_fn="branching_fn", mutation_id=mutation_id, override_colour=override_colour)
        inner_parents = []
        for child in node["children"]["inner_fn"]:
            p = recurse_add_node(child, net, parents=[parent], parent_fn="inner_fn", mutation_id=mutation_id, override_colour=override_colour)
            inner_parents.append(p)
        parent = recurse_add_node(node["children"]["aggregation_fn"], net, parents=inner_parents, parent_fn="aggregation_fn", mutation_id=mutation_id, override_colour=override_colour)
    elif "fn" in node and node["fn"] == "routing_module":
        parent = recurse_add_node(node["children"]["prerouting_fn"], net, parents=parents, parent_fn="prerouting_fn", mutation_id=mutation_id, override_colour=override_colour)
        parent = recurse_add_node(node["children"]["inner_fn"], net, parents=[parent], parent_fn="inner_fn", mutation_id=mutation_id, override_colour=override_colour)
        parent = recurse_add_node(node["children"]["postrouting_fn"], net, parents=[parent], parent_fn="postrouting_fn", mutation_id=mutation_id, override_colour=override_colour)
    elif "fn" in node and node["fn"] == "computation_module":
        parent = recurse_add_node(node["children"]["computation_fn"], net, parents, parent_fn="computation_fn", mutation_id=mutation_id, override_colour=override_colour)
    return parent

def arch_to_nx(architecture, mutation_id=None):
    # Now we label all nodes within the architecture dictionary with a number
    # search_space = EinSpace(
    #     input_shape=architecture["input_shape"],
    #     input_mode=architecture["input_mode"],
    #     num_repeated_cells=1,
    # )
    #search_space.num_nodes = 2
    #search_space.recurse_num_nodes(architecture)
    # create networkx network object
    net = nx.Graph()
    net.add_node(1, label="input", color=colours["input"], title=f"1, {shape_to_string(architecture['input_shape'])}", x=0, y=0, physics=False)
    parent = recurse_add_node(architecture, net, parents=[1], mutation_id=mutation_id)
    net.add_node(parent + 1, label="output", color=colours["output"], title=net[parent], x=0, y=1000)
    net.add_edge(parent, parent + 1)
    return net
