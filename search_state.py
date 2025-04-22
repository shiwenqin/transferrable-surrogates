from copy import deepcopy
import sys
import time
import pickle

import psutil

from pcfg import OutOfOptionsError
from rich import print
from tqdm import tqdm


class Operation:
    def __init__(self, name, build, infer, valid, inherit, give_back, type, child_levels=[]):
        self.name = name
        self.build = build
        self.infer = infer
        self.valid = valid
        self.inherit = inherit
        self.give_back = give_back
        self.type = type
        self.child_levels = child_levels

    def is_valid(self, node):
        return self.valid(node)

    def is_terminal(self):
        return self.type == "terminal"

    def copy(self):
        # own implementation of deepcopy
        return Operation(
            name=self.name,
            build=self.build,
            infer=self.infer,
            valid=self.valid,
            inherit=self.inherit,
            give_back=self.give_back,
            type=self.type,
            child_levels=self.child_levels
        )

    def __repr__(self):
        return f"Operation({self.name}, {self.type}, {self.child_levels})"

    def __sizeof__(self):
        # computes the total size of this object
        return sum(map(sys.getsizeof, self.__dict__.values()))

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class DerivationTreeNode:
    def __init__(
            self,
            id,
            level="network",
            parent=None,
            input_params={},
            depth=0,
            limiter=None,
            operation=None,
        ):
        self.id = id
        self.level = level
        self.parent = parent
        self.children = []
        self.input_params = input_params
        self.output_params = {}
        self.depth = depth
        self.limiter = limiter
        self.operation = operation
        self.available_rules = None

    def initialise(self, operation, stack=None, max_id=None, id_stack=True):
        if stack:
            # might need to find a more effective way to do this
            self.memory = (deepcopy(stack), max_id)
            # self.memory = (stack.copy(), max_id)

        self.operation = operation
        # print(f"Initializing node {self.id} with operation {self.operation}")

        if not self.limiter.check_build_safe(self):
            raise MemoryError(f"Trying to initialize a large operation: {self.limiter.last_op_mem_estimate} MB. Operation: {self.operation.name}")

        # Compute the output params for the current node
        if self.operation.is_terminal():
            self.output_params = self.operation.infer(self)

            if not self.limiter.check_build_safe(self):
                raise MemoryError(f"Individual memory limit reached when initialising: {self}")
        else:
            self.children = []
            for i, child_level in enumerate(operation.child_levels):
                child = DerivationTreeNode(
                    id=max_id + i + 1,
                    level=child_level,
                    parent=self,
                    depth=self.depth + 1,
                    limiter=self.limiter,
                )
                self.add_child(child)
        # print(f"initialised node {self.id} with operation {self.operation}")
        for child in reversed(self.children):
            # print(f"Adding child {child.id} to stack")
            if stack:
                if id_stack:
                    stack.append((child.id, False))
                else:
                    stack.append((child, False))
            max_id = max(child.id, max_id)
        return stack, max_id

    def add_child(self, child):
        self.children.append(child)
        child.set_parent(self)
        # if self.verbose: print(f"Added child {child.id}({hex(id(child))}) to node {self.id}({hex(id(self))})")

    def set_parent(self, parent):
        self.parent = parent

    def get_precursor(self):
        if self.is_root():
            self.precursor = None
        self_idx = self.parent.children.index(self)
        if self_idx == 0: # first child
            precursor = self.parent
        else: # not first child
            precursor = self.parent.children[self_idx - 1]
            while precursor.children: # find most recent 'cousin'
                precursor = precursor.children[-1]
        return precursor

    def inherit_input_params(self):
        child_idx = self.parent.children.index(self)
        self.parent.operation.inherit[child_idx](self)

    def give_back_output_params(self):
        if not self.is_root():
            child_idx = self.parent.children.index(self)
            self.parent.operation.give_back[child_idx](self)

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return self.children == []

    def is_first_child(self):
        return self.parent.children[0] == self

    def get_root(self):
        if self.is_root():
            return self
        return self.parent.get_root()

    def serialise(self):
        # return a list of the nodes in the derivation tree, in pre-order traversal
        nodes = [self]
        for child in self.children:
            nodes.extend(child.serialise())
        return nodes

    def num_params(self):
        root = self.get_root()
        def count_params(node):
            if node.is_leaf():
                output_params = node.operation.infer(node)
                if "num_params" in output_params:
                    return output_params["num_params"]
                else:
                    return 0
            else:
                return sum(count_params(child) for child in node.children)
        return count_params(root)

    def limit_options(self, operation):
        if self.available_rules:
            # get index of the operation in the available rules
            # print("")
            # print(f"Node {self.id}: {self}")
            # print(f"Options {self.available_rules['options']}")
            # print(f"Options {[op.name for op in self.available_rules['options']]}")
            # print(f"Removed {operation.name}")
            # print("")
            op_names = [op.name for op in self.available_rules["options"]]
            idx = op_names.index(operation.name)
            self.available_rules["options"].pop(idx)
            self.available_rules["probs"].pop(idx)
        else:
            pass
            # print(f"Operation {operation.name} not in available rules")
        # print(f"Options left {[op.name for op in self.available_rules['options']]}")

    def build(self, node, set_memory_checkpoint=False):
        if set_memory_checkpoint:
            self.limiter.set_memory_checkpoint()
        #print(f"Check memory: {self.operation}")

        # check memory first
        if not self.limiter.check_memory():
            raise MemoryError(f"Memory limit reached: {self.limiter.memory}")

        if not self.limiter.check_build_safe(node):
            raise MemoryError(f"Individual memory limit reached when building: {self.operation.name}")

        # build the network
        if not self.limiter.check_build_safe(node):
            raise MemoryError(f"Memory limit reached: {self.limiter.memory}")

        network = self.operation.build(node)
        #if set_memory_checkpoint:
        #    self.limiter.reset_memory_checkpoint()

        return network

    def copy(self):
        # own implementation of deepcopy
        node = DerivationTreeNode(
            id=self.id,
            level=self.level,
            input_params=deepcopy(self.input_params),
            depth=self.depth,
            limiter=self.limiter,
            operation=self.operation.copy() if self.operation else None,
        )
        node.output_params = deepcopy(self.output_params)
        node.available_rules = deepcopy(self.available_rules) if self.available_rules else None
        return node

    def replace(self, node):
        # replace the subtree rooted at this node with another node
        # print(f"Replacing node {self.id} with node {node.id}")
        if self.is_root():
            return node
        # set the parent/children of the new node to those of the old node
        node.parent = self.parent
        node.children = self.children
        # set the parent of the children to the new node
        for child in node.children:
            child.parent = node
        # print(f"Parent of node {node.id} is {node.parent.id}")
        # print(f"Children of node {node.id} are {[child.id for child in node.children]}")

        # print(f"Parent of self {self.id} is {self.parent.id}")
        # print(f"Children of self.parent {self.parent.id} are {[child.id for child in self.parent.children]}")
        # set the child of the old parent to the new node
        child_idx = self.parent.children.index(self)
        # print(f"Replacing child {self.id} with child {node.id} at index {child_idx}")
        self.parent.children[child_idx] = node
        # delete the old node
        del self


    def __sizeof__(self):
        # computes the total size of this object
        return sum(map(sys.getsizeof, self.__dict__.values()))

    def __repr__(self):
        return (
            f"DerivationTreeNode(" \
            f"id={self.id}, level={self.level}, operation={self.operation}, " \
            f"input_params={self.input_params}, output_params={self.output_params}, " \
            f"depth={self.depth}, " \
            f"address={hex(id(self))}" \
            # f"memory={self.memory if hasattr(self, 'memory') else None}, " \
            # f"size={round(self.__sizeof__() / 1e6, 2)} MB" \
            f")"
        )

    def __str__(self):
        # convert into a string representation
        # that uses bracket notation to represent the tree

        # determine types of brackets to use
        if self.operation:
            if "branching" in self.operation.name:
                brackets = "{}"
            elif "sequential" in self.operation.name:
                brackets = "()"
            elif "routing" in self.operation.name:
                brackets = "[]"
            elif "computation" in self.operation.name:
                brackets = "<>"
            else:
                brackets = None

            # Initialize the string representation
            repr = f"{self.operation.name}"
            if brackets:
                repr += brackets[0]

            # Append the string representation of each child
            children_repr = ", ".join(str(child) for child in self.children)
            repr += children_repr

            # Append the closing bracket
            if brackets:
                repr += brackets[1]
        else:
            repr = f"None"

        return repr
    
    def param_string(self):
        if self.operation:
            d = {
                "out_feature_shape": list(self.output_params["shape"][1:]),
                # "branching_factor": self.output_params["branching_factor"],
                # "num_params": self.output_params["num_params"] if "num_params" in self.output_params else 0,
            }
            # if self.output_params["other_shape"]:
            #     d["other_shape"] = list(self.output_params["other_shape"][1:])
            return str(d)
        else:
            return ""

    def to_long_string(self):
        # convert into a string representation
        # that uses bracket notation to represent the tree

        # determine types of brackets to use
        if self.operation:
            if "branching" in self.operation.name:
                brackets = "[]"
            elif "sequential" in self.operation.name:
                brackets = "[]"
            elif "routing" in self.operation.name:
                brackets = "[]"
            elif "computation" in self.operation.name:
                brackets = "[]"
            else:
                brackets = None

            # Initialize the string representation
            repr = self.operation.name
            if brackets:
                repr += brackets[0]
            else:
                repr += self.param_string()

            # Append the string representation of each child
            children_repr = ", ".join(child.to_long_string() for child in self.children)
            repr += children_repr

            # Append the closing bracket
            if brackets:
                repr += brackets[1]
        else:
            repr = f"None"

        return repr

    def param_string(self):
        if self.operation:
            d = {
                "out_feature_shape": list(self.output_params["shape"][1:]),
                # "branching_factor": self.output_params["branching_factor"],
                # "num_params": self.output_params["num_params"] if "num_params" in self.output_params else 0,
            }
            # if self.output_params["other_shape"]:
            #     d["other_shape"] = list(self.output_params["other_shape"][1:])
            return str(d)
        else:
            return ""

    def to_long_string(self):
        # convert into a string representation
        # that uses bracket notation to represent the tree

        # determine types of brackets to use
        if self.operation:
            if "branching" in self.operation.name:
                brackets = "[]"
            elif "sequential" in self.operation.name:
                brackets = "[]"
            elif "routing" in self.operation.name:
                brackets = "[]"
            elif "computation" in self.operation.name:
                brackets = "[]"
            else:
                brackets = None

            # Initialize the string representation
            repr = self.operation.name
            if brackets:
                repr += brackets[0]
            else:
                repr += self.param_string()

            # Append the string representation of each child
            children_repr = ", ".join(child.to_long_string() for child in self.children)
            repr += children_repr

            # Append the closing bracket
            if brackets:
                repr += brackets[1]
        else:
            repr = f"None"

        return repr

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


class Stack:
    def __init__(self, stack=[]):
        self.stack = stack

    def append(self, node):
        self.stack.append(node)

    def pop(self):
        return self.stack.pop()

    def restore(self, stack, node):
        # print(f"Restoring stack")
        # print(f"Previous stack: {self}")
        # print(f"Precursor node to restore to: {node}")
        self.stack = stack.stack
        self.stack[-1] = (self.stack[-1][0], False)
        # self.stack.append((node, False))
        # print(f"New stack: {stack}")
        node.limit_options(node.operation)

    def is_empty(self):
        return self.stack == []

    def is_completed(self):
        """ Check if the stack only contain nodes that have been visited """
        for node, visited in self.stack:
            if not visited:
                return False
        return True

    def copy(self):
        # serialise the stack and then reconstruct it
        old_node_list = self.stack[0][0].serialise()
        new_node_list = []
        for node in tqdm(old_node_list, desc="Copying nodes"):
            # print(f"Copying node {node}")
            # create a new copy of the node
            new_node_list.append(node.copy())
        # now we construct the stack from the list of nodes
        old_stack = self.stack
        new_stack = []
        for node, visited in tqdm(old_stack, desc="Copying stack"):
            # find the right node
            idx = [node.id for node in new_node_list].index(node.id)
            new_stack.append((new_node_list[idx], visited))
        # now we must set the parent and children in the new stack
        # following the order of the old stack
        for i in tqdm(range(len(old_stack)), desc="Connecting parents and children"):
            # set the parent of the first node to None
            if i == 0:
                new_stack[i][0].parent = None
            else:
                # find which id the parent of this node has in the old stack
                parent_id = old_stack[i][0].parent.id
                # find that node in the new stack
                parent_idx = [node.id for node in new_node_list].index(parent_id)
                # set the parent of this node to that node
                new_stack[i][0].parent = new_node_list[parent_idx]
            # repeat the same process for the children
            for child in old_stack[i][0].children:
                # find which id the child of this node has in the old stack
                child_idx = [node.id for node in new_node_list].index(child.id)
                # set the child of this node to that node
                new_stack[i][0].add_child(new_node_list[child_idx])
        return Stack(new_stack)

    def __sizeof__(self):
        # computes the total size of this object
        return sum(map(sys.getsizeof, self.__dict__.values()))

    def __repr__(self):
        repr = "Stack(\n"
        if self.stack:
            for node in self.stack[:-1]:
                repr += f"\t{node},\n"
            repr += f"\t{self.stack[-1]}\n"
        repr += ")"
        return repr

    def __str__(self):
        repr = "Stack(\n"
        if self.stack:
            for node in self.stack[:-1]:
                repr += f"\t{node},\n"
            repr += f"\t{self.stack[-1]}\n"
        repr += ")"
        return repr
