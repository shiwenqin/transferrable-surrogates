"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
from collections import defaultdict
from copy import deepcopy
from functools import partial
from os.path import join, exists
from os import makedirs, rename, remove
import pickle
import gc
import sys

from search_strategies.random_search import Sampler
from pcfg import OutOfOptionsError
from search_state import Stack, DerivationTreeNode
from visualise import visualise_derivation_tree
from visualise import visualise_search_tree_2 as visualise_search_tree
from plot import Plotter
from utils import CPU_Unpickler

from rich import print
import math, random
from tqdm import tqdm
from scipy.stats import norm

from guppy import hpy

ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])


class TimeLimitExceededError(Exception):
    pass


class SearchTreeNode:
    """
    A representation of a single search state.
    MCTS works by constructing a search tree of these Nodes.
    """
    def __init__(
            self,
            id,
            pcfg,
            operation,
            stack,
            max_id,
            verbose=False,
            backtrack=True
        ):
        self.id = id
        self.pcfg = pcfg
        self.operation = operation
        self.stack = stack
        self.max_id = max_id
        self.verbose = verbose
        self.backtrack = backtrack

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __str__(self):
        return f"SearchTreeNode(id={self.id}, operation={self.operation.name if self.operation else None})"

    def __repr__(self):
        return str(self)

class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."
    def __init__(
            self,
            evaluation_fn,
            pcfg,
            limiter,
            input_params,
            seed=0,
            mode="iterative",
            backtrack=True,
            verbose=False,
            visualise=False,
            visualise_scale=0.5,
            vis_interval=10,
            figures_path=None,
            results_path=None,
            continue_search=False,
            # mcts specific parameters
            acquisition_fn="uct",
            exploration_weight=1.0,
            incubent_type="parent",
            reward_mode="sum",
            add_full_paths=False,
        ):
        self.evaluation_fn = evaluation_fn
        self.pcfg = pcfg
        self.limiter = limiter
        self.input_params = input_params
        self.seed = seed
        self.mode = mode
        self.backtrack = backtrack
        self.verbose = verbose
        self.visualise = visualise
        self.visualise_scale = visualise_scale
        self.vis_interval = vis_interval
        self.figures_path = figures_path
        self.results_path = results_path
        self.continue_search = continue_search
        # mcts specific parameters
        self.exploration_weight = exploration_weight
        self.acquisition_fn = acquisition_fn
        self.incubent_type = incubent_type
        self.reward_mode = reward_mode
        self.add_full_paths = add_full_paths

        initial_derivation_tree_max_id = 1
        initial_search_tree_max_id = 1

        # initialise the sampler
        self.sampler = Sampler(
            pcfg=self.pcfg,
            mode=self.mode,
            verbose=self.verbose
        )

        if self.acquisition_fn == "uct":
            self.acquisition_select = self._uct
        elif self.acquisition_fn == "ei":
            self.acquisition_select = self._ei
        else:
            raise ValueError("Invalid acquisition function")

        # default initialisation of the search
        self.Q = defaultdict(int)
        self.N = defaultdict(int)
        self.children = dict()
        self.rewards = []
        self.iteration = 0
        self.search_tree_max_id = initial_search_tree_max_id
        self.root = DerivationTreeNode(
            initial_derivation_tree_max_id,
            "network",
            input_params=self.input_params,
            limiter=self.pcfg.limiter,
        )
        self.search_tree_root = SearchTreeNode(
            id=self.search_tree_max_id,
            pcfg=self.pcfg,
            operation=None,
            stack=Stack([(self.root, False)]),
            max_id=1,
            verbose=self.verbose,
            backtrack=self.backtrack
        ) # The root of the search tree
        # set the random seed
        self.set_rng_state(seed=self.seed)

        # continue search from previous results
        if self.continue_search:
            self.load_results()

        # fix for the clock
        for _ in range(10):
            self.limiter.timer.start()
            self.limiter.timer()
        print(f"Initialised MCTS at {self.limiter.timer.start_time}")

    def set_rng_state(self, seed=None, state=None):
        if state:
            random.setstate(state)
        elif seed:
            random.seed(seed)

    def learn(self, steps=10):
        """
        Run the Monte Carlo Tree Search algorithm for a number of steps
        """
        print("-----------------------")
        print("Monte Carlo Tree Search")
        print(f"Steps: {steps}")
        print("-----------------------")

        if self.iteration == 0:
            # expand the initial search tree from the root
            print("Initialising search tree: expanding children of root node")
            # start a timer for the first expansion
            self.limiter.timer.start()
            self._expand_path([self.search_tree_root])

        # iterate through the search tree, expanding and simulating
        for iteration in tqdm(range(self.iteration, steps), desc="MCTS", initial=self.iteration, total=steps):
            # set the memory checkpoint
            self.limiter.set_memory_checkpoint()
            print(f"Memory checkpoint: {self.limiter.memory_checkpoint} MB")

            # do a single iteration of MCTS
            end_node, path, reward = self.do_rollout(self.search_tree_root, iteration)

            # save to results
            self.save_results(iteration)

            self.plot(end_node, path, reward, iteration)

            # Force garbage collection
            print(f"Performing Garbage Collection")
            gc.collect()
            for obj in gc.garbage:
                if isinstance(obj, SearchTreeNode):
                    print(f"\tFound cyclic reference: {obj}")

    def do_rollout(self, node, iteration):
        "Make the tree one layer bigger. (Train for one iteration.)"
        # select the node to expand
        path = self._select(node)
        if self.verbose: print("Path", path)

        # start timer
        self.limiter.timer.start()
        # expand the node
        leaf = path[-1]
        self._expand_path(path)

        success = False
        while not success:
            try:
                # print("Rollout")
                # self.limiter.summarise_memory()
                # h = hpy()
                # print(h.heap())
                # print some more memory stats
                # print(f"Memory of self.nodes: {self.total_memory(self.nodes) / 1e6:.2f} MB")
                # print(f"Memory of self.children: {self.total_memory(self.children) / 1e6:.2f} MB")
                # print(f"Memory of self.Q: {self.total_memory(self.Q) / 1e6:.2f} MB")
                # print(f"Memory of self.N: {self.total_memory(self.N) / 1e6:.2f} MB")
                # print(f"Memory of self.rewards: {self.total_memory(self.rewards) / 1e6:.2f} MB")
                # print(f"Memory of self.search_tree_root: {self.total_memory(self.search_tree_root) / 1e6:.2f} MB")
                # print(f"Memory of self: {self.total_memory(self) / 1e6:.2f} MB")

                if self.verbose: print("Simulating architecture")
                simulation_path = deepcopy(path)
                # start timer
                self.limiter.timer.start()
                # simulate the architecture
                root = self._simulate(simulation_path)
                sample_duration = self.limiter.timer()

                # check if batch pass does not exceed the time limit
                if not self.limiter.check_batch_pass_time(root, check_memory=True):
                    print("Batch pass time or memory exceeded, trying again")
                    continue

                # start timer
                self.limiter.timer.start()
                # evaluate the architecture
                reward = self._reward(root)
                eval_duration = self.limiter.timer()
                if self.verbose: print(f"Leaf node after simulate {leaf}")
                if self.verbose: print("Simulated architecture, with reward:", reward)
                success = True
            except (RuntimeError, MemoryError) as e:
                print(f"Error in Rollout: {e}")

        # serialize the architecture
        serialised_architecture = root.serialise()

        if self.add_full_paths:
            if self.verbose: print(f"Adding full paths")
            if self.verbose: print(f"Path: {path}")
            if self.verbose: print(f"Serialised architecture: {root}")
            path = [self.search_tree_root]
            # expand the search tree with the whole new architecture
            for i in range(len(serialised_architecture)):
                if self.verbose: print(f"Checking node {path[-1]}")
                if path[-1] not in self.children:
                    if self.verbose: print(f"Node {path[-1]} not in children, expanding")
                    self._expand_path(path)
                else:
                    if self.verbose: print(f"Node {path[-1]} already in children")
                # add the right child to the path
                if self.verbose: print(f"Finding child for {serialised_architecture[i]}")
                if self.verbose: print(f"Children: {self.children[path[-1]]}")
                for child in self.children[path[-1]]:
                    if child.operation == serialised_architecture[i].operation:
                        path.append(child)
                        if self.verbose: print(f"Extended path: {path}")
                        break
            # self.children[path[-1]] = []
            if self.verbose: print(f"Final path: {path}")

        # backpropagate the reward
        self._backpropagate(path, reward)
        if self.verbose: print("Backpropagated reward")

        # save the rewards
        self.rewards.append((serialised_architecture, reward, sample_duration, eval_duration))
        print(f"Iteration {iteration}, reward: {reward:.2f}, sample duration: {sample_duration:.2f}, eval duration: {eval_duration:.2f}")
        print(f"Architecture: {root}")

        return root, path, reward

    def _select(self, node):
        "Find an unexplored descendant of `node`"
        if self.verbose: print(f"---- Selecting a node ----")
        path = []
        while True:
            path.append(node)
            if self.verbose: print(f"Appended {node} to path")
            if node not in self.children or not self.children[node]:
                # node is either unexplored or termina
                if self.verbose: print(f"This node is either unexplored or a terminal, return path: {path}")
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                if self.verbose: print(f"Found unexplored child among children: {n}")
                if self.verbose: print(f"Returning path: {path}")
                return path
            if self.verbose: print(f"Node {node} has children:")
            for child in self.children[node]:
                if self.verbose: print(f"\t{child}")
                if self.verbose: print(f"\tQ: {self.Q[child]}, N: {self.N[child]}")
            # All children of node should already be expanded:
            assert all(n in self.children for n in self.children[node])
            # descend a layer deeper
            node = max(self.children[node], key=partial(self.acquisition_select, parent=node))
            if self.verbose: print(f"Max score child: {node}")

    def _expand_path(self, path, leaf=None):
        """
        path is a list of SearchTreeNodes, each containing a DerivationTreeNode
        This method visits each node in the path and keeps track of the stack of what nodes to visit next
        It ends by visiting the last node in the path and expanding it, thereby adding children to it (SearchTreeNodes)
        """
        success = False
        while not success:
            operations = [node.operation for node in path if node.operation]

            if self.verbose: print("Expanding path")
            for i, node in enumerate(path):
                if self.verbose: print(f"\t{node}")
            if self.verbose: print("Operations")
            for i, operation in enumerate(operations):
                if self.verbose: print(f"\t{operation.name}")
            root = DerivationTreeNode(
                id=1,
                level="network",
                input_params=self.input_params,
                limiter=self.pcfg.limiter,
            )
            self.nodes = {root.id: root}

            max_id = root.id
            stack = Stack([(root.id, False)])

            while not stack.is_empty() and len(operations) > 0:
                if self.verbose: print(f"Stack: {stack}")
                if self.verbose: print(f"Operations")
                for i, operation in enumerate(operations):
                    if self.verbose: print(f"\t{operation.name}")

                node_id, visited = stack.pop()
                node = self.nodes[node_id]
                if self.verbose: print(f"Node: {node.id}, visited: {visited}")
                if self.verbose: print(f"Node: {node}")

                if visited:
                    # Propagate the output params to the parent
                    node.give_back_output_params()
                    if not node.is_root():
                        if self.verbose: print(f"Propagated output params from node {node.id} to parent {node.parent.id}")
                        if self.verbose: print(f"Output params for node: {node.parent.id}, {node.parent.output_params}")
                else:
                    stack.append((node.id, True))
                    if not node.is_root():
                        # inherit the input params from the parent
                        node.inherit_input_params()
                        if self.verbose: print(f"Inherited input params from parent {node.parent.id} to node {node.id}")
                        if self.verbose: print(f"Input params for node: {node.id}, {node.input_params}")
                    if self.verbose: print(f"Sampling node {node.id}")
                    # select operation and initialise the node, children etc.
                    operation = operations.pop(0)
                    stack, max_id = node.initialise(
                        operation,
                        stack,
                        max_id,
                        id_stack=True,
                    )
                    if self.verbose: print(f"Initialised node {node.id} with operation {operation}")
                    if self.verbose: print(f"New stack: {stack}")
                    for child in node.children:
                        if child.id not in self.nodes:
                            self.nodes[child.id] = child
            # if the stack is empty, we have reached the end of the path
            if stack.is_empty() or stack.is_completed():
                print("Reached a terminal node when expanding. Trying again...")
                print(f"Stack: {stack}")
                continue # try again

            if self.verbose: print("Expanding the last node in the path")
            # expand the last node in the path
            if self.verbose: print(f"Stack: {stack}")
            node_id, visited = stack.pop()
            node = self.nodes[node_id]
            if self.verbose: print(f"Node: {node_id}, visited: {visited}")
            while visited:
                # Propagate the output params to the parent
                node.give_back_output_params()
                if not node.is_root():
                    if self.verbose: print(f"Propagated output params from node {node.id} to parent {node.parent.id}")
                    if self.verbose: print(f"Output params for node: {node.parent.id}, {node.parent.output_params}")
                if self.verbose: print(f"Stack: {stack}")
                node_id, visited = stack.pop()
                node = self.nodes[node_id]
                if self.verbose: print(f"Node: {node_id}, visited: {visited}")
            if self.verbose: print(f"Node: {node}")

            if not node.is_root():
                # inherit the input params from the parent
                node.inherit_input_params()
                if self.verbose: print(f"Inherited input params from parent {node.parent.id} to node {node.id}")
                if self.verbose: print(f"Input params for node: {node.id}, {node.input_params}")
            if self.verbose: print(f"Sampling node {node.id}")

            # find the available and filtered options for the current node
            options, probs = self.pcfg.get_available_options(node, verbose=self.verbose)
            options, _ = self.pcfg.filter_options(
                node,
                options,
                probs,
                verbose=self.verbose
            )
            if self.verbose: print(f"Available options of node {node.id}: {[op.name for op in options]}")
            
            children = []
            for i, operation in enumerate(options):
                child_node, child_stack, child_max_id = self.step(
                    node=deepcopy(node),
                    visited=visited,
                    stack=deepcopy(stack),
                    max_id=max_id,
                    operation=operation,
                )

                child = SearchTreeNode(
                    id=self.search_tree_max_id + i + 1,
                    pcfg=self.pcfg,
                    operation=child_node.operation,
                    stack=child_stack,
                    max_id=child_max_id,
                    verbose=self.verbose,
                    backtrack=self.backtrack
                )
                children.append(child)
                if self.verbose: print(f"Child node {child}\nwith parent {child_node.parent}")
                if self.verbose: print(f"Stack of child node {child.id}: {child.stack}")
            if leaf is None:
                leaf = path[-1]
            if leaf not in self.children:
                self.search_tree_max_id += len(children)
                self.children[leaf] = children
            if self.verbose: print(f"Expanded node {leaf}\nwith children\n{self.children[leaf]}")
            self.nodes = {}
            success = True

    def step(self, node, visited, stack, max_id, operation):
        """
        Step through the search tree
        """
        if self.verbose: print(f"Stepping node dtid={node.id}, visited={visited} with operation {node.operation}")
        stack.append((node.id, True))
        if not node.is_root():
            # inherit the input params from the parent
            node.inherit_input_params()
            if self.verbose: print(f"Inherited input params from parent {node.parent.id}({hex(id(node.parent))}) to node {node.id}({hex(id(node))})")
            if self.verbose: print(f"Input params for node: {node.id}, {node.input_params}")
        stack, max_id = node.initialise(operation, stack, max_id, id_stack=True)
        self.nodes[node.id] = node
        if self.verbose: print(f"Initialised node dtid={node.id} with operation {operation}")
        if self.verbose: print(f"Output params for node: {node.id}, {node.output_params}")
        while not stack.is_empty() and stack.stack[-1][1]:
            # while the last element in the stack is visited
            # we want to pop it and go back to the parent
            _node_id, visited = stack.pop()
            _node = self.nodes[_node_id]
            # Propagate the output params to the parent
            _node.give_back_output_params()
            if self.verbose and not _node.is_root(): print(f"Propagated output params from node {_node.id}({hex(id(_node))}) to parent {_node.parent.id}({hex(id(_node.parent))})")
            if self.verbose and not _node.is_root(): print(f"Output params for node: {_node.parent.id}, {_node.parent.output_params}")
        if self.verbose: print(f"Returning node {node}")
        if self.verbose: print(f"Returning stack {stack}")
        return node, stack, max_id

    def _simulate(self, path):
        "Returns the reward for a random simulation (to completion) of `node`"
        operations = [node.operation for node in path if node.operation]
        root = self.sampler.sample(self.input_params, operations)
        return root

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            if self.reward_mode == "sum":
                self.Q[node] += reward
            elif self.reward_mode == "max":
                self.Q[node] = max(self.Q[node], reward)
            # print(f"Node {node}")
            # print(f"N: {self.N[node]}, Q: {self.Q[node]}")

    def _get_score(self, idx):
        if self.reward_mode == "sum":
            return self.Q[idx] / self.N[idx]
        elif self.reward_mode == "max":
            return self.Q[idx]

    def _uct(self, node, parent):
        "Upper confidence bound for trees"
        log_N_vertex = math.log(self.N[parent])
        mu = self._get_score(node)
        return mu + self.exploration_weight * math.sqrt(
            log_N_vertex / self.N[node]
        )

    def _ei(self, node, parent):
        "Expected Improvement for trees"
        log_N_vertex = math.log(self.N[parent])

        # what to compare it to
        if self.incubent_type == "global": # highest avg reward of all nodes
            y_star = max([self._get_score(n) for n in self.Q])
        elif self.incubent_type == "parent": # same as avg of children
            y_star = self._get_score(parent)
        elif self.incubent_type == "children":
            y_star = max([self._get_score(n) for n in self.children[parent]])

        h = lambda z, mu, sigma: norm.pdf(z, loc=0, scale=1) + z * norm.cdf(z, loc=0, scale=1)
        mu = self._get_score(node)
        sigma = math.sqrt(log_N_vertex / self.N[node]) + 0.0001

        return sigma * h((mu - y_star) / sigma, mu, sigma)

    def _reward(self, node):
        "Return the reward for the node"
        reward = self.evaluation_fn(node.get_root())
        return reward

    def is_terminal(self, node):
        return node.stack.is_empty()

    def plot(self, root, path, reward, iteration):
        # visualise the derivation and search trees
        visualise_derivation_tree(
            root.get_root(),
            scale=self.visualise_scale,
            iteration=iteration,
            save_path=self.figures_path,
            score=reward,
            show=self.visualise,
        )
        if self.verbose: print("Path", path)
        visualise_search_tree(
            self.search_tree_root,
            self.children,
            self.Q,
            self.N,
            path=[(a.id, b.id) for a, b in zip(path[0:], path[1:])],
            score_fn=lambda node, parent: self.Q[node] / self.N[node],
            scale=self.visualise_scale,
            iteration=iteration,
            save_path=self.figures_path,
            show=self.visualise,
        )
        visualise_search_tree(
            self.search_tree_root,
            self.children,
            self.Q,
            self.N,
            path=[(a.id, b.id) for a, b in zip(path[0:], path[1:])],
            score_fn=self.acquisition_select,
            scale=self.visualise_scale,
            iteration=f"acquisition_{iteration}",
            save_path=self.figures_path,
            show=self.visualise,
        )
        if iteration % self.vis_interval == 0:
            plotter = Plotter({"rewards": self.rewards})
            # find best architecture
            idx, best_arch, best_reward = plotter.find_best_architecture()
            # visualise it
            visualise_derivation_tree(
                best_arch[0], iteration=f"best_{idx}", score=best_reward, show=False,
                save_path=self.figures_path
            )
            # plot results
            plotter.plot_results("rewards", self.figures_path)
            # plot number of parameters
            plotter.plot_num_params(self.figures_path)
            # plot number of nodes
            plotter.plot_num_nodes(self.figures_path)

    def save_results(self, iteration):
        if self.results_path:
            makedirs(self.results_path, exist_ok=True)
            temp_path = join(self.results_path, f"search_results_temp.pkl")
            final_path = join(self.results_path, f"search_results.pkl")
            try:
                with open(temp_path, "wb") as f:
                    pickle.dump({
                        "rewards": self.rewards,
                        "Q": self.Q,
                        "N": self.N,
                        "children": self.children,
                        "iteration": iteration,
                        "search_tree_max_id": self.search_tree_max_id,
                        "search_tree_root": self.search_tree_root,
                        "rng_state": random.getstate(),
                    }, f)
                rename(temp_path, final_path)
            except KeyboardInterrupt:
                print("Saving interrupted. Partial results saved.")
                if exists(temp_path):
                    remove(temp_path)

    def load_results(self):
        # load the search results
        path = join(self.results_path, "search_results.pkl")
        if exists(path):
            with open(path, "rb") as f:
                data = CPU_Unpickler(f).load()
                self.Q = data["Q"]
                self.N = data["N"]
                self.children = data["children"]
                self.rewards = data["rewards"]
                self.iteration = data["iteration"] + 1
                self.search_tree_max_id = data["search_tree_max_id"]
                self.search_tree_root = data["search_tree_root"]
                # set the random seed
                self.set_rng_state(state=data["rng_state"])
                print(f"Continuing search from iteration {self.iteration}")
        else:
            print("No previous search results found, starting from scratch")

    def total_memory(self, obj, seen=None):
        """Recursively calculates the total memory of a Python object, including nested objects."""
        if seen is None:
            seen = set()  # To track objects we've already accounted for

        obj_id = id(obj)
        if obj_id in seen:  # Avoid counting the same object multiple times
            return 0

        seen.add(obj_id)

        size = sys.getsizeof(obj)  # Size of the object itself

        if isinstance(obj, dict):
            size += sum(self.total_memory(k, seen) + self.total_memory(v, seen) for k, v in obj.items())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            size += sum(self.total_memory(i, seen) for i in obj)
        elif hasattr(obj, '__dict__'):  # If the object has __dict__, include its attributes
            size += self.total_memory(vars(obj), seen)
        elif hasattr(obj, '__slots__'):  # If the object uses __slots__, include those
            size += sum(self.total_memory(getattr(obj, slot), seen) for slot in obj.__slots__ if hasattr(obj, slot))

        return size
