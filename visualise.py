from os.path import join
from os import makedirs

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


colours = {
    "root": '#8d3b2fff',             # hex for dark brown #3d2b1fff
    "network": '#999999ff',          # grey
    "module": '#999999ff',           # grey
    "branching_fn_2": '#ffd966ff',   # light yellow 1
    "branching_fn_4": '#ffd966ff',   # light yellow 1
    "branching_fn_8": '#ffd966ff',   # light yellow 1
    "aggregation_fn_2": '#a64d79ff', # dark magenta 1
    "aggregation_fn_4": '#a64d79ff', # dark magenta 1
    "aggregation_fn_8": '#a64d79ff', # dark magenta 1
    "prerouting_fn": '#93c47dff',    # light green 1
    "postrouting_fn": '#93c47dff',   # light green 1
    "computation_fn": '#6fa8dcff',   # light blue 1
    "input": '#333333ff',            # black
    "output": '#cc4125ff',           # light red berry 1
    "mutation": '#ff007fff',         # bright pink
}


def visualise_derivation_tree(root, stack=None, current_node_id=None, scale=1, iteration=None, save_path=None, score=None, show=False):
    def add_edges(graph, root, stack=None, current_node_id=None):
        if root is not None:
            op_name = root.operation.name if root.operation else ""
            input_shape = list(root.input_params["shape"]) if "shape" in root.input_params else ""
            output_shape = list(root.output_params["shape"]) if "shape" in root.output_params else ""
            other_shape = list(root.input_params["other_shape"]) if "other_shape" in root.input_params and root.input_params["other_shape"] is not None else ""
            on_stack = True if stack is not None and root.id in stack else False
            current = True if current_node_id == root.id else False
            if current:
                edgecolor =  "#00ff00"
            elif on_stack:
                edgecolor = "magenta"
            else:
                edgecolor = "none"
            # print(f"Adding node {root.id} with operation {op_name}")
            graph.add_node(
                root.id,
                input_shape=input_shape,
                output_shape=output_shape,
                other_shape=other_shape,
                op_name=op_name,
                color=colours[root.level] if root.id > 1 else colours["root"],
                edgecolor=edgecolor,
            )
            for child in root.children:
                add_edges(graph, child, stack, current_node_id)
                # print(f"Adding edge from {root.id} to {child.id}")
                graph.add_edge(root.id, child.id)

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges to the graph
    stack_ids = [a.id for a, _ in stack] if stack is not None else None
    add_edges(G, root, stack_ids, current_node_id)

    # get the depth of the tree
    depth = max(nx.shortest_path_length(G, source=1).values()) + 1
    # get max degree of tree
    max_degree = max([G.out_degree(node) for node in G.nodes])
    # and the width of the tree (i.e. number of leaves)
    width = max_degree ** (depth - 1)

    # Draw the graph
    # tree layout
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    labels = {}
    for i, node in enumerate(G.nodes):
        labels[node] = f"{node}\n{nx.get_node_attributes(G, 'op_name')[node]}\n" \
            f"{nx.get_node_attributes(G, 'input_shape')[node]}\n" \
            f"{nx.get_node_attributes(G, 'output_shape')[node]}\n" \
            f"{nx.get_node_attributes(G, 'other_shape')[node]}"
        # colors = ["skyblue" if node.out_degree() > 0 else "salmon" for node in G.nodes]
    colors = nx.get_node_attributes(G, 'color').values()
    # draw border around nodes on stack
    border_colors = nx.get_node_attributes(G, 'edgecolor').values()

    fig = plt.figure(figsize=(8 * scale, 6 * scale))
    ax = fig.add_subplot(111)
    ax.set_title(f"Derivation Tree at iteration {iteration}")
    if score is not None:
        ax.set_title(f"Derivation Tree at iteration {iteration} with score {score:.2f}")
    nx.draw(
        G, pos, labels=labels, with_labels=True,
        node_size=(60 * scale) ** 2, node_color=colors, edgecolors=border_colors, linewidths=2,
        font_size=6 * scale, font_color="white", font_weight="bold",
        arrows=True, arrowsize=20 * scale, arrowstyle="->",
        edge_color="gray", width=2 * scale, ax=ax
    )
    # edit the text of the nodes
    # labels = nx.get_edge_attributes(G, "output_val")
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    # change the size of the figure
    # plt.gcf().set_size_inches(8, 6)
    # extend the margins
    plt.margins(0.05 + 0.05 * scale)
    if save_path is not None:
        makedirs(save_path, exist_ok=True)
        plt.savefig(join(save_path, f"derivation_tree_{iteration}.png"))
        plt.savefig(join(save_path, f"derivation_tree.pdf"))
    if show:
        plt.show()
    plt.close()


def visualise_search_tree(root, children, Q, N, path=None, scale=1, layout="twopi", iteration=None):
    def add_edges(graph, root, children, Q, N):
        if root is not None:
            graph.add_node(
                root.id,
                op_name=root.operation.name if root.operation else "",
                score=Q[root],
                visits=N[root],
                # color=colours[root.node.level] if root.id > 1 else colours["root"],
            )
            if root in children:
                for child in children[root]:
                    add_edges(graph, child, children, Q, N)
                    edge_color = "magenta" if (root.id, child.id) in path else "grey"
                    thickness = (Q[child] / (N[child] + 0.01)) * 5 + 0.2
                    graph.add_edge(root.id, child.id, color=edge_color, thickness=thickness)

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges to the graph
    add_edges(G, root, children, Q, N)

    # get the depth of the tree
    depth = max(nx.shortest_path_length(G, source=1).values()) + 1
    # get max degree of tree
    max_degree = max([G.out_degree(node) for node in G.nodes])
    # and the width of the tree (i.e. number of leaves)
    width = max_degree ** (depth - 1)

    # Draw the graph
    # tree layout
    pos = nx.nx_agraph.graphviz_layout(G, prog=layout)
    labels = {}
    for i, node in enumerate(G.nodes):
        labels[node] = f"{node}\n{nx.get_node_attributes(G, 'op_name')[node]}\n" \
            f"{nx.get_node_attributes(G, 'score')[node]:.2f}/" \
            f"{nx.get_node_attributes(G, 'visits')[node]}"
    colors = nx.get_node_attributes(G, 'color').values()
    edge_color = [G[u][v]["color"] for u, v in G.edges]
    edge_thickness = [G[u][v]["thickness"] * scale for u, v in G.edges]

    fig = plt.figure(figsize=(12 * scale, 12 * scale))
    ax = fig.add_subplot(111)
    ax.set_title(f"Search Tree at iteration {iteration}")
    nx.draw(
        G, pos, labels=labels, with_labels=True,
        node_size=(42 * scale) ** 2, node_color=colors, linewidths=2,
        font_size=6 * scale, font_color="white", font_weight="bold",
        arrows=True, arrowsize=20 * scale, arrowstyle="->",
        edge_color=edge_color, width=edge_thickness, ax=ax
    )
    # edit the text of the nodes
    # labels = nx.get_edge_attributes(G, "output_val")
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    # change the size of the figure
    # plt.gcf().set_size_inches(8, 6)
    # extend the margins
    plt.margins(0.05 + 0.05 * scale)
    plt.show()


def visualise_search_tree_2(root, children, Q, N, path=None, score_fn=None, scale=1, layout="twopi", iteration=None, save_path=None, show=False):
    def add_edges(graph, node, parent, children, Q, N):
        if node is not None:
            graph.add_node(
                node.id,
                op_name=node.operation.name if node.operation else "",
                score=score_fn(node, parent) if parent is not None and score_fn is not None and (node in children) and N[node] != 0 else 0,
                visits=N[node],
                # color=colours[node.node.level] if node.id > 1 else colours["root"],
            )
            if node in children:
                for child in children[node]:
                    add_edges(graph, child, node, children, Q, N)
                    edge_color = "#ab3396" if (node.id, child.id) in path else "grey"
                    thickness = 2 if (node.id, child.id) in path else 1
                    graph.add_edge(node.id, child.id, color=edge_color, thickness=thickness)

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges to the graph
    add_edges(G, root, None, children, Q, N)

    # get the depth of the tree
    depth = max(nx.shortest_path_length(G, source=1).values()) + 1
    # get max degree of tree
    max_degree = max([G.out_degree(node) for node in G.nodes])
    # and the width of the tree (i.e. number of leaves)
    width = max_degree ** (depth - 1)

    # Draw the graph
    # tree layout
    pos = nx.nx_agraph.graphviz_layout(G, prog=layout)
    labels = {}
    for i, node in enumerate(G.nodes):
        labels[node] = f"{node}\n{nx.get_node_attributes(G, 'op_name')[node]}\n" \
            f"{nx.get_node_attributes(G, 'score')[node]:.2f}/" \
            f"{nx.get_node_attributes(G, 'visits')[node]}"
    n_colors = 100
    palette = sns.color_palette("ch:start=.2,rot=-.3", n_colors=n_colors)
    # assign the colours according to their score/visits on a scale of 0 to 9
    scores = nx.get_node_attributes(G, 'score').values()
    # if any score is above 1
    if max(scores) > 1:
        scores = [score / max(scores) for score in scores]
    visits = nx.get_node_attributes(G, 'visits').values()
    colors = [palette[int((n_colors - 1) * score)] if node != 1 else '#000000' for node, score in zip(G.nodes, scores)]
    sizes = [(10 * scale) ** 2 if visit > 0 else (10 * scale) ** 2 for visit in visits]
    edge_color = [G[u][v]["color"] for u, v in G.edges]
    edge_thickness = [G[u][v]["thickness"] * scale for u, v in G.edges]

    fig = plt.figure(figsize=(12 * scale, 12 * scale))
    ax = fig.add_subplot(111)
    ax.set_title(f"Search Tree at iteration {iteration}")
    nx.draw(
        G, pos, labels=None, with_labels=False,
        node_size=sizes, node_color=colors, linewidths=2,
        font_size=6 * scale, font_color="white", font_weight="bold",
        arrows=True, arrowsize=20 * scale, arrowstyle="-",
        edge_color=edge_color, width=edge_thickness, ax=ax
    )
    # edit the text of the nodes
    # labels = nx.get_edge_attributes(G, "output_val")
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    # change the size of the figure
    # plt.gcf().set_size_inches(8, 6)
    # extend the margins
    # plt.margins(0.05 + 0.05 * scale)
    if save_path is not None:
        makedirs(save_path, exist_ok=True)
        plt.savefig(join(save_path, f"search_tree_{iteration}.png"))
        plt.savefig(join(save_path, f"search_tree.pdf"))
    if show:
        plt.show()
    plt.close()
