# pyright: basic
import pickle
import functools
from collections import Counter, defaultdict
import pandas as pd
import networkx as nx
import networkx.exception as nxe

from surrogates.encodings.utils import Converter
#import einspace.search_spaces as espace
from surrogates.encodings.utils import (
    get_average_branching_factor,
    get_max_depth,
    get_size,
    recurse_count_nodes,
)
from surrogates.encodings.base import BaseEncoder
import viz_tools

from grammars.einspace import quick_grammar


class GRAFEncoder(BaseEncoder):
    def __init__(self, args, feats=None, return_names=True):
        super().__init__()

        self.converter = Converter()
        self.feats = (
            feats 
            if feats is not None
            else ['cnt', 'in', 'out', 'in_deg', 'out_deg', 'max_nodes_path', 'min_nodes_path']
        )
        self.return_names = return_names
        self.args = args
        
    def encode_individual(self, individual):
        node = individual.root
        node = self.converter.convert_to_old(node)
        node["input_mode"] = self.args.input_mode

        nodes = {}
        for k, v in quick_grammar.items():
            if k == 'network':
                continue

            for o in v['options']:
                name = self.converter.translate_name(o.name)
                nodes[name] = 0

        feats = extract_features(node, nodes)
        feats = features_to_flat_dict(feats, nodes)
        
        res = {}
        for k, v in feats.items():
            if any(k.startswith(f) for f in self.feats):
                res[k] = v

        return res if self.return_names else list(res.values())


# fn_groups = espace.EinSpace.available_options
#
# coarse_groups = {'routing': list(set(fn_groups['prerouting_fn'] + fn_groups['postrouting_fn'])),
#                  'branching': fn_groups['branching_fn'],
#                  'aggregation': fn_groups['aggregation_fn'],
#                  'computation': fn_groups['computation_fn']}
#
# coarse_groups = {g: [f.__name__ for f in fns] for g, fns in coarse_groups.items()}
# just a manual copy of the above, in case einspace changes

# coarse division of nodes into groups
coarse_groups = {'routing': {'im2col3k2s1p', 'permute312', 'im2col1k1s0p', 'col2im', 'permute132', 'im2col4k4s0p', 'permute321', 'im2col1k2s0p', 'permute213', 'im2col8k8s0p', 'permute21', 'im2col3k1s1p', 'permute231', 'im2col16k16s0p'},
                 'branching': {'clone_tensor2', 'clone_tensor4', 'clone_tensor8', 'group_dim2s1d', 'group_dim2s2d', 'group_dim2s3d', 'group_dim4s1d', 'group_dim4s2d', 'group_dim4s3d', 'group_dim8s1d', 'group_dim8s2d', 'group_dim8s3d'}, 
                 'aggregation': {'dot_product', 'scaled_dot_product', 'add_tensors', 'cat_tensors1d2t', 'cat_tensors2d2t', 'cat_tensors3d2t', 'cat_tensors1d4t', 'cat_tensors2d4t', 'cat_tensors3d4t', 'cat_tensors1d8t', 'cat_tensors2d8t', 'cat_tensors3d8t'}, 
                 'computation': {'linear16', 'linear32', 'linear64', 'linear128', 'linear256', 'linear512', 'linear1024', 'linear2048', 'norm', 'leakyrelu', 'softmax', 'learnable_positional_encoding'},
                 }

# more fine-grained division of node types into groups
fine_groups = {'im2col': {'im2col3k2s1p', 'im2col1k1s0p', 'col2im', 'im2col4k4s0p', 'im2col1k2s0p', 'im2col8k8s0p', 'im2col3k1s1p', 'im2col16k16s0p'},
               'permute': {'permute312', 'permute132', 'permute321', 'permute213',  'permute21', 'permute231'},
                'aggregation': {'dot_product', 'scaled_dot_product', 'add_tensors', 'cat_tensors1d2t', 'cat_tensors2d2t', 'cat_tensors3d2t', 'cat_tensors1d4t', 'cat_tensors2d4t', 'cat_tensors3d4t', 'cat_tensors1d8t', 'cat_tensors2d8t', 'cat_tensors3d8t'}, 
                'clone': {'clone_tensor2', 'clone_tensor4', 'clone_tensor8df'}, 
                'group_dim': {'group_dim2s1d', 'group_dim2s2d', 'group_dim2s3d', 'group_dim4s1d', 'group_dim4s2d', 'group_dim4s3d', 'group_dim8s1d', 'group_dim8s2d', 'group_dim8s3d'}, 
                'linear': {'linear16', 'linear32', 'linear64', 'linear128', 'linear256', 'linear512', 'linear1024', 'linear2048'}, 
                'other_layers': {'norm', 'leakyrelu', 'softmax', 'learnable_positional_encoding'},
                 }


def get_named_descriptor(arch, nodes):
    named_descriptor = {
            "feature_shape": arch["output_shape"][1],
            "num_terminals": get_size(arch, "terminal"),
            "num_nonterminals": get_size(arch, "nonterminal"),
            #"average_branching_factor": get_average_branching_factor(arch),
            "max_depth": get_max_depth(arch),
        }

    named_descriptor.update(recurse_count_nodes(arch, nodes))
    return named_descriptor


def extract_features(ind, nodes):
    # extract GRAF features from a neural network
    descriptor = get_named_descriptor(ind, nodes)
    graph = viz_tools.arch_to_nx(ind)
    digraph = nx.DiGraph()
    digraph.add_nodes_from(graph.nodes.data())
    for e in graph.edges:
        digraph.add_edge(*e)

    inputs = [n for n in graph.nodes.data() if n[1]['label'] == 'input']
    outputs = [n for n in graph.nodes.data() if n[1]['label'] == 'output']

    labels = {d['label'] for _, d in graph.nodes.data()} - set(['input', 'output'])
    
    out_degrees = {}
    in_degrees = {}
    for l in labels:
        in_l = []
        out_l = []
        for n in graph.nodes:
            in_l.append(sum(graph.nodes[nn]['label'] == l for nn in digraph.predecessors(n)))
            out_l.append(sum(graph.nodes[nn]['label'] == l for nn in digraph.successors(n)))
        out_degrees[l] = sum(out_l)/len(out_l)
        in_degrees[l] = sum(in_l)/len(in_l)

    assert len(inputs) == 1
    assert len(outputs) == 1

    input = inputs[0][0]
    output = outputs[0][0]

    input_neighbors = Counter(graph.nodes[n]['label'] for n in nx.neighbors(graph, input))
    output_neighbors = Counter(graph.nodes[n]['label'] for n in nx.neighbors(graph, output))

    path_stats = find_path_stats(digraph, input, output, labels, fine_groups)

    node_stats = {l: count_node_types(digraph,  output, {l}) for l in labels}

    net_data = {'input_neighbors': input_neighbors,
                'output_neighbors': output_neighbors,
                'all_labels': list(set(labels) - {'input', 'output'}),
                'path_stats': path_stats,
                'node_stats': node_stats,
                'descriptor': descriptor,
                'in_degrees': in_degrees,
                'out_degrees': out_degrees}

    return net_data

def dag_compute(graph: nx.DiGraph, to, agg_fn, init_val, ts=None):
    # general dynamic programming iteration over all vertices in a dag
    # source vertices get init_val, for other vertices the values from predecessors
    # are aggregated using the agg_fn
    if not ts:
        ts = nx.topological_sort(graph)
    vals = defaultdict(lambda: init_val)
    for n in ts:
        preds = list(graph.predecessors(n))
        if not preds:
            continue
        vals[n] = agg_fn([vals[p] for p in preds], n)
    return vals[to]

def longest_path(graph, to):
    # count the longest path in DAG
    return dag_compute(graph, to, lambda x, _: max(x) + 1, 0)

def count_paths(graph, to):
    # count the number of paths in DAG
    return dag_compute(graph, to, lambda x, _: sum(x), 1)

def count_node_types(graph: nx.DiGraph, to, node_types):
    # computes minimum and maximum number of types a given node type is used on
    # path to output (from any source - for neural nets, it will always be the input) 
    def node_fn(pred_vals, node, agg_fn):
        val = agg_fn(pred_vals)
        if graph.nodes.data()[node]['label'] in node_types:
            val += 1
        return val
    max_fn = lambda pr, n: node_fn(pr, n, agg_fn=max)
    min_fn = lambda pr, n: node_fn(pr, n, agg_fn=min)
    ts = list(nx.topological_sort(graph))
    return {'max': dag_compute(graph, to, init_val=0, agg_fn=max_fn, ts=ts), 
            'min': dag_compute(graph, to, init_val=0, agg_fn=min_fn, ts=ts)}

def path_length(graph, fr, to):
    # computes path lengths between input and output
    try: 
        sp = nx.shortest_path_length(graph, fr, to)
        lp = longest_path(graph, to)
    except nxe.NetworkXNoPath:
        return (0, 0)
    return (sp, lp)

def find_path_stats(graph: nx.Graph, fr, to, labels, groups=None):
    # tests all relevant subsets of groups/labels, 
    # whenever there is no path from input to output for a given set, its subsets are skipped
    labels = list(labels)
    if not groups:
        groups = {l: {l} for l in labels}
    nodes_by_type = defaultdict(list)
    for n,d in graph.nodes.data():
        nodes_by_type[d['label']].append(n)
    def backtrack(start, current_set):
        group_keys = list(groups.keys())
        for i in range(start, len(group_keys)):
            current_set.append(group_keys[i])
            g = graph.copy()
            dis_labels = functools.reduce(lambda x,y: x | y, (groups[x] for x in current_set), set())
            g.remove_nodes_from(sum((nodes_by_type[t] for t in dis_labels), start=[]))
            sp, lp = path_length(g, fr, to)
            pc = count_paths(g, to)
            if sp > 0:
                result.append({'allowed_labels': list(set(group_keys) - set(current_set)), 
                               'shortest': sp, 
                               'longest': lp, 
                               'count': pc})
                backtrack(i + 1, current_set)
            current_set.pop()
    
    sp, lp = path_length(graph, fr, to)
    pc = count_paths(graph, to)
    result = [{'allowed_labels': list(set(groups.keys())), 
               'shortest': sp, 
               'longest': lp, 
               'count': pc}]
    backtrack(0, [])
    return result

def features_to_flat_dict(feat_dict: dict, nodes) -> dict:
    # transform features as returned by the `extract_features` method to a flattened
    # dictionary - useful for creating pandas dataframes
    # also fills in missing vales with defaults
    #opts = espace.EinSpace.available_options
    #all_labels = opts['prerouting_fn'] + opts['postrouting_fn'] + opts['branching_fn'] + opts['aggregation_fn'] + opts['computation_fn']
    #all_labels = set(l.__name__ for l in all_labels)

    feats = {}
    counts = {f'cnt_{l}': feat_dict['descriptor'][l] for l in nodes.keys()}
    feats |= counts
    
    other_desc = {k: v for k,v in feat_dict['descriptor'].items() if k not in nodes.keys()}
    feats |= other_desc
    
    path = {f'max_nodes_path_{l}': feat_dict['node_stats'].get(l, {'max': 0})['max'] for l in nodes.keys()}
    path |= {f'min_nodes_path_{l}': feat_dict['node_stats'].get(l, {'min': 0})['min'] for l in nodes.keys()}
    feats |= path
    
    input_neighbors = {f'in_{l}': feat_dict['input_neighbors'].get(l, 0) for l in nodes.keys()}
    output_neighbors = {f'out_{l}': feat_dict['output_neighbors'].get(l, 0) for l in nodes.keys()}
    feats |= input_neighbors
    feats |= output_neighbors
    
    out_degs = {f'out_deg_{l}': feat_dict['out_degrees'].get(l, 0) for l in nodes.keys()}
    in_degs = {f'in_deg_{l}': feat_dict['in_degrees'].get(l, 0) for l in nodes.keys()}
    feats |= out_degs
    feats |= in_degs
    
    path_stats = max(feat_dict['path_stats'], key=lambda x: x['count'])
    del path_stats['allowed_labels']
    path_stats = {f'path_{k}': v for k, v in path_stats.items()}
    feats |= path_stats

    return feats

"""
if __name__ == '__main__':
    with open(f'benchmark.pkl', 'rb') as f:
        feats = []
        for i, ind in enumerate(pickle.load(f)['cifar10']):
            #features = extract_features(ind)
            #f = features_to_flat_dict(features) | {'id': i}
            feats.append(f)


    feats = pd.DataFrame(feats)
    feats.to_csv("cifar10_graf.csv")
    print(feats)
"""