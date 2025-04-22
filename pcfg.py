from copy import deepcopy
from random import choices
import sys
from rich import print
import psutil


class OutOfOptionsError(Exception):
    pass


class PCFG:
    def __init__(self, grammar, limiter):
        self.grammar = grammar
        self.limiter = limiter

    def sample(self, node, verbose=False):
        available_options, available_probs = self.get_available_options(node, verbose)
        node.available_rules = {"options": available_options, "probs": available_probs}
        # if verbose: print(f"Sampled options for node {node.id} at level {node.level}: {available_options}, {available_probs}")
        options, probs = self.filter_options(node, available_options, available_probs, verbose)
        if verbose: print(f"Filtered options for node {node.id} at level {node.level}: {[op.name for op in options]}")
        if verbose: print(f"Full list of options at node {node.id}: {[op.name for op in available_options]}")
        # node.available_rules = {"options": options, "probs": probs}
        if len(options) > 0:
            # if verbose: print(f"Sampled options for node {node.id} at level {node.level}: {options}, {probs}")
            operation = choices(
                options,
                weights=probs,
                k=1
            )[0]
            if verbose: print(f"Sampled operation {operation.name} for node {node.id} at level {node.level}")
        else:
            raise OutOfOptionsError(f"Out of options for node {node.id} at level {node.level}")
        return operation

    def get_available_options(self, node, verbose=False):
        if node.level not in self.grammar:
            return None
        if node.available_rules == None:
            available_options = deepcopy(self.grammar[node.level]["options"])
            available_probs = deepcopy(self.grammar[node.level]["probs"])
        else:
            available_options = node.available_rules["options"]
            available_probs = node.available_rules["probs"]
        # print(f"Available options for node {node.id} at level {node.level}: {[op.name for op in available_options]}")
        # print(f"Available probabilities for node {node.id} at level {node.level}: {available_probs}")
        return available_options, available_probs

    def filter_options(self, node, options, probs, verbose=False):
        indices = [i for i, op in enumerate(options) if op.valid(node)]

        # extra check to see if repeated sequential modules can be closed
        # print(f"Possible options: {[options[i].name for i in indices]}")
        indices_to_remove = []
        for i in indices:
            if not self.check_sequential_closure(node, options, indices, i):
                # print(f"Removed option {options[i].name} for node {node.id} at level {node.level}")
                indices_to_remove.append(i)
        indices = [i for i in indices if i not in indices_to_remove]
        # print(f"Possible options after filtering: {[options[i].name for i in indices]}")

        # check if max depth has been reached
        # print(f"Node depth: {node.depth}, duration: {duration}, node id: {node.id}")

        replace_rules = False  # in case of limits reached, keel only the computational module available
        success = self.limiter.check(node, verbose)
        if not success:
            # if we are choosing between modules,
            # remove all but the computation module option
            if any([op.name == "computation" for op in options]):
                indices = [i for i in indices if options[i].name == "computation"]
                replace_rules = True

        if verbose: print(f"Filtered options for node {node.id} at level {node.level}: {[options[i].name for i in indices]}")
        options, probs = [options[i] for i in indices], [probs[i] for i in indices]
        # renormalize the probabilities
        probs = [p / sum(probs) for p in probs]

        if replace_rules:
            node.available_rules = {"options": options, "probs": probs}

        return options, probs

    def check_sequential_closure(self, node, options, indices, i):
        # extra check to see if a sequential(k) module is being closed
        # of so, the inner modules must be repeatable k times
        # first we must find if there is a parent node that is a sequential(k) module
        def is_seq_k_parent(node):
            if node.is_root():
                return None
            elif node.parent.operation.name.startswith("sequential("):
                return node.parent
            else:
                return is_seq_k_parent(node.parent)
        # find out if the current node is the final in the subtree of a sequential(k) module
        def is_final_child(parent, node):
            if node in parent.children:
                if node == parent.children[-1] and node.level != "module":
                    return True
                else:
                    return False
            elif parent.children:
                if is_final_child(parent.children[-1], node):
                    return True
            return False

        option = options[i]

        # check if the current node is the final in the subtree of a sequential(k) module
        seq_k_parent = is_seq_k_parent(node)
        if seq_k_parent and is_final_child(seq_k_parent, node):
            k = int(seq_k_parent.operation.name[-2])
            # check if the inner modules are repeatable k times
            return self.check_sequential_closure_inner(seq_k_parent, node, k, option)
        else:
            return True

    def check_sequential_closure_inner(self, root, current_node, k, operation):
        # check if the inner modules are repeatable k times
        current_node.operation = operation
        print(f"Checking repeatable {k} times: {root}, {operation.name}")
        def infer(node):
            if not node.is_root():
                node.inherit_input_params()
            node.output_params = node.operation.infer(node)
            for child in node.children:
                infer(child)
            node.give_back_output_params()
        # store the original input params
        if not root.is_root():
            root.inherit_input_params()
        original_root_input_params = deepcopy(root.input_params)
        original_current_node_input_params = deepcopy(current_node.input_params)
        # flag for whether the inner modules are repeatable k times
        repeatable = True
        for _ in range(k):
            try:
                infer(root.children[0])
                # pass output params back in
                root.input_params = deepcopy(root.output_params)
                print(f"Input params: {root.input_params}")
                if 0 in root.input_params["shape"]:
                    raise ValueError(f"Invalid input params: {root.input_params}")
            except Exception as e:
                print(f"Error: {e}")
                print(f"Failed to repeat {k} times: {root}, {operation.name}")
                repeatable = False
                break
        if repeatable:
            print(f"Successfully repeated {k} times: {root}, {operation.name}")
        # reset input params and current node operation
        root.input_params = original_root_input_params
        current_node.input_params = original_current_node_input_params
        current_node.operation = None
        return repeatable

    def __repr__(self):
        # return the object in a readable format
        result = ["Grammar:"]
        for level, rules in self.grammar.items():
            result.append(f"\t{level}:")
            for i, (rule, prob) in enumerate(zip(rules["options"], rules["probs"])):
                result.append(f"\t\t{rule.name:<20}(p={prob})")
        return "\n".join(result)

    def __str__(self):
        # return the object in a readable format
        result = ["Grammar:"]
        for level, rules in self.grammar.items():
            result.append(f"\t{level}:")
            for i, (rule, prob) in enumerate(zip(rules["options"], rules["probs"])):
                result.append(f"\t\t{rule.name:<20}(p={prob})")
        return "\n".join(result)

    def __sizeof__(self):
        # computes the total size of this object
        return sum(map(sys.getsizeof, self.__dict__.values()))
