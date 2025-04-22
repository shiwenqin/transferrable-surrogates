from collections import OrderedDict


class Converter:
    def translate_name(self, name):
        if name in ["sequential", "routing", "branching(2)", "branching(4)", "branching(8)", "computation"]:
            name = name.replace("(", "").replace("2", "").replace("4", "").replace("8", "").replace(")", "")
            name += "_module"
        if "im2col" in name:
            # convert im2col(a, b, c) to im2colakbscp for any a, b and c
            # extract a, b and c
            a, b, c = map(lambda x: x.strip(), name.split("(")[1].split(")")[0].split(","))
            # insert a, b and c in the new name
            name = f"im2col{a}k{b}s{c}p"
        if "clone" in name:
            # convert clone(a) to clonea for any a
            a = name.split("(")[1].split(")")[0]
            name = f"clone_tensor{a}"
        if "group" in name:
            # convert group(a, b) to group_dimasbd for any a, b
            a, b = map(lambda x: x.strip(), name.split("(")[1].split(")")[0].split(","))
            name = f"group_dim{a}s{b}d"
        if "cat" in name:
            # convert cat(a, b) to cat_tensorsbdat for any a, b
            a, b = map(lambda x: x.strip(), name.split("(")[1].split(")")[0].split(","))
            name = f"cat_tensors{b}d{a}t"
        if "dot_product" in name:
            if "scaled" in name:
                # convert dot_product(scaled=True) to scaled_dot_product
                name = "scaled_dot_product"
            else:
                # convert dot_product to dot_product
                name = "dot_product"
        if "add" in name:
            # convert add to add
            name = "add_tensors"
        if "perm" in name:
            # if two commas
            if name.count(",") == 2:
                # convert permute(0, 2, 1) to permute21
                a, b, c = map(lambda x: x.strip(), name.split("(")[1].split(")")[0].split(","))
                name = f"permute{b}{c}"
            # if three commas
            elif name.count(",") == 3:
                # convert permute(0, 2, 3, 1) to permute231
                a, b, c, d = map(lambda x: x.strip(), name.split("(")[1].split(")")[0].split(","))
                name = f"permute{b}{c}{d}"
        if "linear" in name:
            # convert linear(a) to lineara for any a
            a = name.split("(")[1].split(")")[0]
            name = f"linear{a}"
        if "relu" in name:
            # convert relu to leakyrelu
            name = "leakyrelu"
        if "pos_enc" in name:
            # convert positional_encoding to learnable_positional_encoding
            name = "learnable_positional_encoding"
        return name
    
    def convert_to_old(self, root):
        """
        Convert any architecture from the new representation to the old representation
        In a recursive manner that can handle any depth of the architecture
        Input:
            root: DerivationTreeNode (with children)
        Output:
            OrderedDict
        """
        self.next_id = self.get_max_id(root) + 1
        return self.recursive_convert_to_old(root)

    def recursive_convert_to_old(self, root):
        """
        Convert any architecture from the new representation to the old representation
        In a recursive manner that can handle any depth of the architecture
        Input:
            root: DerivationTreeNode (with children)
        Output:
            OrderedDict
        """
        if root.operation.name == "sequential":
            return OrderedDict({
                "fn": self.translate_name(root.operation.name),
                "children": OrderedDict({
                    "first_fn": self.recursive_convert_to_old(root.children[0]),
                    "second_fn": self.recursive_convert_to_old(root.children[1]),
                }),
                "input_shape": root.input_params["shape"],
                "output_shape": root.output_params["shape"],
                "depth": root.depth,
                "node_type": "nonterminal",
                "node_id": root.id,
            })
        elif root.operation.name == "routing":
            return OrderedDict({
                "fn": self.translate_name(root.operation.name),
                "children": OrderedDict({
                    "prerouting_fn": self.recursive_convert_to_old(root.children[0]),
                    "inner_fn": self.recursive_convert_to_old(root.children[1]),
                    "postrouting_fn": self.recursive_convert_to_old(root.children[2]),
                }),
                "input_shape": root.input_params["shape"],
                "output_shape": root.output_params["shape"],
                "depth": root.depth,
                "node_type": "nonterminal",
                "node_id": root.id,
            })
        elif root.operation.name == "branching(2)":
            return OrderedDict({
                "fn": self.translate_name(root.operation.name),
                "children": OrderedDict({
                    "branching_fn": self.recursive_convert_to_old(root.children[0]),
                    "inner_fn": [
                        self.recursive_convert_to_old(root.children[1]),
                        self.recursive_convert_to_old(root.children[2]),
                    ],
                    "aggregation_fn": self.recursive_convert_to_old(root.children[3]),
                }),
                "input_shape": root.input_params["shape"],
                "output_shape": root.output_params["shape"],
                "depth": root.depth,
                "node_type": "nonterminal",
                "node_id": root.id,
            })
        elif root.operation.name == "branching(4)":
            branch, next_id = self.copy_branch(root.children[1], self.next_id, 4)
            self.next_id = next_id
            return OrderedDict({
                "fn": self.translate_name(root.operation.name),
                "children": OrderedDict({
                    "branching_fn": self.recursive_convert_to_old(root.children[0]),
                    "inner_fn": [self.recursive_convert_to_old(c) for c in branch],
                    "aggregation_fn": self.recursive_convert_to_old(root.children[2]),
                }),
                "input_shape": root.input_params["shape"],
                "output_shape": root.output_params["shape"],
                "depth": root.depth,
                "node_type": "nonterminal",
                "node_id": root.id,
            })
        elif root.operation.name == "branching(8)":
            branch, next_id = self.copy_branch(root.children[1], self.next_id, 8)
            self.next_id = next_id
            return OrderedDict({
                "fn": self.translate_name(root.operation.name),
                "children": OrderedDict({
                    "branching_fn": self.recursive_convert_to_old(root.children[0]),
                    "inner_fn": [self.recursive_convert_to_old(c) for c in branch],
                    "aggregation_fn": self.recursive_convert_to_old(root.children[2]),
                }),
                "input_shape": root.input_params["shape"],
                "output_shape": root.output_params["shape"],
                "depth": root.depth,
                "node_type": "nonterminal",
                "node_id": root.id,
            })
        elif root.operation.name == "computation":
            return OrderedDict({
                "fn": self.translate_name(root.operation.name),
                "children": OrderedDict({
                    "computation_fn": self.recursive_convert_to_old(root.children[0]),
                }),
                "input_shape": root.input_params["shape"],
                "output_shape": root.output_params["shape"],
                "depth": root.depth,
                "node_type": "nonterminal",
                "node_id": root.id,
            })
        elif root.is_leaf():
            return OrderedDict({
                "fn": self.translate_name(root.operation.name),
                "input_shape": root.input_params["shape"],
                "output_shape": root.output_params["shape"],
                "depth": root.depth,
                "node_type": "terminal",
                "node_id": root.id,
            })
        else:
            raise ValueError(f"Unknown level {root.level}")

    def copy_branch(self, child, next_id, n_times):
        """
        Copy a branch n_times
        Input:
            child: DerivationTreeNode
            max_id: int
            n_times: int
        Output:
            DerivationTreeNode
        """
        branch = []
        for _ in range(n_times):
            new_child, next_id = self.copy_subtree(child, next_id)
            branch.append(new_child)
        return branch, next_id


    def convert_to_new(self, root):
        """
        Convert any architecture from the old representation to the new representation
        In a recursive manner that can handle any depth of the architecture
        Input:
            root: OrderedDict (with children)
        Output:
            DerivationTreeNode
        """
        raise NotImplementedError
    
    def copy_subtree(self, root, next_id):
        """
        Copy the subtree rooted at root
        Input:
            root: DerivationTreeNode (with children)
        Output:
            DerivationTreeNode
        """
        new_root = root.copy()
        root.id = next_id
        next_id += 1
        for child in root.children:
            new_child, next_id = self.copy_subtree(child, next_id)
            new_root.children.append(new_child)

        return new_root, next_id

        
    def get_max_id(self, root):
        """
        Get the maximum id in the subtree rooted at root
        Input:
            root: DerivationTreeNode (with children)
        Output:
            int
        """
        id = root.id
        for child in root.children:
            id = max(id, self.get_max_id(child))
        return id


##### Adapted from https://github.com/bosswissam/pysize

def recurse_num_nodes(d, num_nodes):
    """Label all nodes within the architecture dictionary with a number."""

    if "sequential_module" in d["fn"].__name__:
        num_nodes += 1
        num_nodes += recurse_num_nodes(d["children"]["first_fn"])
        num_nodes += 1
        num_nodes += recurse_num_nodes(d["children"]["second_fn"])
    elif "branching_module" in d["fn"].__name__:
        num_nodes += 1
        num_nodes += recurse_num_nodes(d["children"]["branching_fn"])
        for i in range(len(d["children"]["inner_fn"])):
            num_nodes += 1
            num_nodes += recurse_num_nodes(d["children"]["inner_fn"][i])
        num_nodes += 1
        num_nodes += recurse_num_nodes(d["children"]["aggregation_fn"])
    elif "routing_module" in d["fn"].__name__:
        num_nodes += 1
        num_nodes += recurse_num_nodes(d["children"]["prerouting_fn"])
        num_nodes += 1
        num_nodes += recurse_num_nodes(d["children"]["inner_fn"])
        num_nodes += 1
        num_nodes += recurse_num_nodes(d["children"]["postrouting_fn"])
    elif "computation_module" in d["fn"].__name__:
        num_nodes += 1
        num_nodes += recurse_num_nodes(d["children"]["computation_fn"])
    else:
        pass
    return num_nodes


def get_size(obj, count_type="terminal"):
    """Recursively finds size of objects"""
    size = 0
    if count_type == "function":
        if callable(obj):
            size = 1
    elif count_type in ["terminal", "nonterminal"]:
        try:
            if isinstance(obj, dict):
                if "node_type" in obj:
                    if obj["node_type"] == count_type:
                        size = 1
        except:
            pass

    if isinstance(obj, (dict, OrderedDict)):
        size += sum([get_size(v, count_type) for v in obj.values()])
    if isinstance(obj, (list, tuple)):
        size += sum([get_size(v, count_type) for v in obj])
    return size


def get_average_branching_factor(arch):
    # computes the average branching factor of individuals in a population
    b = recurse_sum(arch, "input_branching_factor")
    n = recurse_sum(arch, "node")
    return b / n


def recurse_sum(node, count_type="node"):
    if "node_type" in node and node["node_type"] == "terminal":
        if "branching_factor" in count_type:
            return node[count_type]
        elif count_type == "node":
            return 1
    else:
        total = 0
        if "fn" in node and node["fn"] == "sequential_module":
            total += recurse_sum(node["children"]["first_fn"], count_type)
            total += recurse_sum(node["children"]["second_fn"], count_type)
        elif "fn" in node and node["fn"] == "branching_module":
            total += recurse_sum(node["children"]["branching_fn"], count_type)
            for child in node["children"]["inner_fn"]:
                total += recurse_sum(child, count_type)
            total += recurse_sum(
                node["children"]["aggregation_fn"], count_type
            )
        elif "fn" in node and node["fn"] == "routing_module":
            total += recurse_sum(node["children"]["prerouting_fn"], count_type)
            total += recurse_sum(node["children"]["inner_fn"], count_type)
            total += recurse_sum(
                node["children"]["postrouting_fn"], count_type
            )
        elif "fn" in node and node["fn"] == "computation_module":
            total += recurse_sum(
                node["children"]["computation_fn"], count_type
            )
    return total


def recurse_count_nodes(node, num_nodes):
    if "fn" in node and node["fn"] == "sequential_module":
        num_nodes["sequential_module"] += 1
        num_nodes = recurse_count_nodes(
            node["children"]["first_fn"], num_nodes
        )
        num_nodes = recurse_count_nodes(
            node["children"]["second_fn"], num_nodes
        )
    elif "fn" in node and node["fn"] == "branching_module":
        num_nodes["branching_module"] += 1
        num_nodes = recurse_count_nodes(
            node["children"]["branching_fn"], num_nodes
        )
        for child in node["children"]["inner_fn"]:
            num_nodes = recurse_count_nodes(child, num_nodes)
        num_nodes = recurse_count_nodes(
            node["children"]["aggregation_fn"], num_nodes
        )
    elif "fn" in node and node["fn"] == "routing_module":
        num_nodes["routing_module"] += 1
        num_nodes = recurse_count_nodes(
            node["children"]["prerouting_fn"], num_nodes
        )
        num_nodes = recurse_count_nodes(
            node["children"]["inner_fn"], num_nodes
        )
        num_nodes = recurse_count_nodes(
            node["children"]["postrouting_fn"], num_nodes
        )
    elif "fn" in node and node["fn"] == "computation_module":
        num_nodes["computation_module"] += 1
        num_nodes = recurse_count_nodes(
            node["children"]["computation_fn"], num_nodes
        )
    else:
        num_nodes[node["fn"]] += 1
    return num_nodes


def recurse_list_nodes(node, node_type, nodes):
    if "node_type" in node and node["node_type"] == node_type:
        nodes.append(node)
    if "fn" in node and node["fn"] == "sequential_module":
        nodes = recurse_list_nodes(
            node["children"]["first_fn"], node_type, nodes
        )
        nodes = recurse_list_nodes(
            node["children"]["second_fn"], node_type, nodes
        )
    elif "fn" in node and node["fn"] == "branching_module":
        nodes = recurse_list_nodes(
            node["children"]["branching_fn"], node_type, nodes
        )
        for child in node["children"]["inner_fn"]:
            nodes = recurse_list_nodes(child, node_type, nodes)
        nodes = recurse_list_nodes(
            node["children"]["aggregation_fn"], node_type, nodes
        )
    elif "fn" in node and node["fn"] == "routing_module":
        nodes = recurse_list_nodes(
            node["children"]["prerouting_fn"], node_type, nodes
        )
        nodes = recurse_list_nodes(
            node["children"]["inner_fn"], node_type, nodes
        )
        nodes = recurse_list_nodes(
            node["children"]["postrouting_fn"], node_type, nodes
        )
    elif "fn" in node and node["fn"] == "computation_module":
        nodes = recurse_list_nodes(
            node["children"]["computation_fn"], node_type, nodes
        )
    return nodes


def predict_num_parameters(arch):
    leaves = recurse_list_nodes(arch, "terminal", [])
    num_params = 0
    for leaf in leaves:
        # print(leaf["fn"], leaf["input_shape"], leaf["output_shape"])
        if "linear" in leaf["fn"]:
            num_params += (
                leaf["input_shape"][-1] * leaf["output_shape"][-1]
                + leaf["output_shape"][-1]
            )
        if "positional_encoding" in leaf["fn"]:
            if len(leaf["output_shape"]) == 3:
                num_params += leaf["output_shape"][1] * leaf["output_shape"][2]
            elif len(leaf["output_shape"]) == 4:
                num_params += (
                    leaf["output_shape"][1]
                    * leaf["output_shape"][2]
                    * leaf["output_shape"][3]
                )
        if "norm" in leaf["fn"]:
            if len(leaf["output_shape"]) == 3:
                num_params += leaf["output_shape"][2] * 2
            elif len(leaf["output_shape"]) == 4:
                num_params += leaf["output_shape"][1] * 2
    return num_params


def recurse_max(node, node_property, current_max):
    if "node_type" in node and node["node_type"] == "terminal":
        return max(current_max, node[node_property])
    else:
        if "fn" in node and node["fn"] == "sequential_module":
            current_max = max(
                current_max,
                recurse_max(
                    node["children"]["first_fn"], node_property, current_max
                ),
                recurse_max(
                    node["children"]["second_fn"], node_property, current_max
                ),
            )
        elif "fn" in node and node["fn"] == "branching_module":
            current_max = max(
                current_max,
                recurse_max(
                    node["children"]["branching_fn"],
                    node_property,
                    current_max,
                ),
                max(
                    [
                        recurse_max(child, node_property, current_max)
                        for child in node["children"]["inner_fn"]
                    ]
                ),
                recurse_max(
                    node["children"]["aggregation_fn"],
                    node_property,
                    current_max,
                ),
            )
        elif "fn" in node and node["fn"] == "routing_module":
            current_max = max(
                current_max,
                recurse_max(
                    node["children"]["prerouting_fn"],
                    node_property,
                    current_max,
                ),
                recurse_max(
                    node["children"]["inner_fn"], node_property, current_max
                ),
                recurse_max(
                    node["children"]["postrouting_fn"],
                    node_property,
                    current_max,
                ),
            )
        elif "fn" in node and node["fn"] == "computation_module":
            current_max = max(
                current_max,
                recurse_max(
                    node["children"]["computation_fn"],
                    node_property,
                    current_max,
                ),
            )
    return current_max


def get_max_depth(arch):
    return recurse_max(arch, node_property="depth", current_max=0)
