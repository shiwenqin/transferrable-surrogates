from os.path import join, exists
from os import makedirs, rename, remove
import pickle
import random

from tqdm import tqdm

from visualise import visualise_derivation_tree
from search_state import DerivationTreeNode, Stack
from pcfg import OutOfOptionsError
from plot import Plotter


class Sampler:
    def __init__(
            self,
            pcfg,
            mode,
            time_limit=300,
            max_id_limit=1000,
            depth_limit=20,
            mem_limit=4096,
            verbose=False
        ):
        self.pcfg = pcfg
        self.mode = mode
        self.time_limit = time_limit
        self.max_id_limit = max_id_limit
        self.depth_limit = depth_limit
        self.mem_limit = mem_limit
        self.verbose = verbose

        if self.mode == "iterative":
            self.__call__ = self.sample_iterative
        elif self.mode == "recursive":
            raise NotImplementedError("Recursive mode not implemented")

    def sample(self, input_params, operations=None, root=None):
        # catch RuntimeErrors and MemoryErrors and try again
        return self.__call__(input_params, operations, root)

    def sample_iterative(self, input_params, operations=None, root=None):
        if root is None:
            root = DerivationTreeNode(
                id=1,
                level="network",
                input_params=input_params,
                limiter=self.pcfg.limiter,
            )
        self.nodes = {root.id: root}

        max_id = root.id
        stack = Stack([(root.id, False)])

        while not stack.is_empty():
            # print(f"Architecture so far: {root}")
            if self.verbose: print(f"Architecture so far: {root}")
            if operations is not None:
                if self.verbose: print(f"Operations: {[op.name for op in operations]}")

            if self.verbose: print(f"Stack: {stack}")
            node_id, visited = stack.pop()
            node = self.nodes[node_id]
            if self.verbose: print(f"Node: {node.id}, visited: {visited}")
            if self.verbose: print(f"Node: {node}")

            if visited:
                # Propagate the output params to the parent
                node.give_back_output_params()
                if not node.is_root():
                    if self.verbose: print(f"Propagated output params from node {node.id} to parent {node.parent.id}")
                    if self.verbose: print(f"Input params for node: {node.parent.id}, {node.parent.input_params}")
                    if self.verbose: print(f"Output params for node: {node.parent.id}, {node.parent.output_params}")
            else:
                stack.append((node.id, True))
                if not node.is_root():
                    # inherit the input params from the parent
                    node.inherit_input_params()
                    if self.verbose: print(f"Inherited input params from parent {node.parent.id} to node {node.id}")
                    if self.verbose: print(f"Input params for node: {node.id}, {node.input_params}")
                try:
                    if self.verbose: print(f"Sampling node {node.id}")
                    # select operation and initialise the node, children etc.
                    operation = operations.pop(0) if operations else self.pcfg.sample(
                        node,
                        verbose=self.verbose,
                    )
                    if operation not in self.pcfg.get_available_options(node)[0]:
                        raise RuntimeError(f"Operation {operation} not in available options")
                    if self.verbose: print(f"Selected operation: {operation}")
                    stack, max_id = node.initialise(
                        operation,
                        stack,
                        max_id,
                    )
                    if self.verbose: print(f"Output params for node: {node.id}, {node.output_params}")
                    for child in node.children:
                        if child.id not in self.nodes:
                            self.nodes[child.id] = child
                    # print(f"Architecture so far: {root}")
                    # print(len(stack.stack), len([a for a, v in stack.stack if not v]), len(operations), node.level, operation.name)
                except OutOfOptionsError:
                    # get the precursor node, and remove the previously chosen operation from its options
                    node = node.get_precursor()
                    # backtrack to the previous state of the stack
                    stack, _ = node.memory
                    stack.restore(stack, node)
                    if self.verbose: print(f"Backtracked to node {node.id}")
                    # print(f"Backtracked to node {node.id}")
        return root


class RandomSearch:
    def __init__(
            self,
            evaluation_fn,
            pcfg,
            limiter,
            input_params,
            seed=0,
            mode="iterative",
            backtrack=True,
            time_limit=300,
            max_id_limit=1000,
            depth_limit=20,
            mem_limit=4096,
            verbose=False,
            visualise=False,
            visualise_scale=0.5,
            vis_interval=10,
            figures_path=None,
            results_path=None,
            continue_search=False,
        ):
        self.evaluation_fn = evaluation_fn
        self.pcfg = pcfg
        self.limiter = limiter
        self.input_params = input_params
        self.seed = seed
        self.mode = mode
        self.backtrack = backtrack
        self.time_limit = time_limit
        self.max_id_limit = max_id_limit
        self.depth_limit = depth_limit
        self.mem_limit = mem_limit
        self.verbose = verbose
        self.visualise = visualise
        self.visualise_scale = visualise_scale
        self.vis_interval = vis_interval
        self.figures_path = figures_path
        self.results_path = results_path
        self.continue_search = continue_search

        self.sampler = Sampler(
            pcfg=self.pcfg,
            mode=self.mode,
            time_limit=self.time_limit,
            max_id_limit=self.max_id_limit,
            depth_limit=self.depth_limit,
            mem_limit=self.mem_limit,
            verbose=self.verbose
        )

        self.rewards = []
        self.iteration = 0

        self.set_rng_state(seed=self.seed)

        if self.continue_search:
            self.load_results()

    def set_rng_state(self, seed=None, state=None):
        if state:
            random.setstate(state)
        elif seed:
            random.seed(seed)

    def learn(self, steps):
        print("-------------")
        print("Random Search")
        print(f"Steps: {steps}")
        print("--------------")

        for iteration in tqdm(range(self.iteration, steps), desc="RS", initial=self.iteration, total=steps):
            success = False
            while not success:
                try:
                    # start timer
                    self.limiter.timer.start()
                    # sample the network
                    root = self.sampler.sample(self.input_params)
                    sample_duration = self.limiter.timer()

                    # start timer
                    self.limiter.timer.start()
                    # evaluate the network
                    reward = self.evaluation_fn(root)
                    eval_duration = self.limiter.timer()

                    success = True
                except (RuntimeError, MemoryError) as e:
                    print(f"Error in sampling new architecture: {e}")

            self.rewards.append((root.serialise(), reward, sample_duration, eval_duration))
            print(f"Iteration {iteration}, reward: {reward:.2f}, sample duration: {sample_duration:.2f}, eval duration: {eval_duration:.2f}")
            # print(f"Architecture:")
            # for line in root.serialise():
            #     print(line)

            self.plot(root, reward, iteration)

            # save the results
            self.save_results(iteration)

    def plot(self, root, reward, iteration):
        # visualise the derivation tree
        visualise_derivation_tree(
            root,
            scale=self.visualise_scale,
            iteration=iteration,
            save_path=self.figures_path,
            score=reward,
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
                        "iteration": iteration,
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
                data = pickle.load(f)
                self.rewards = data["rewards"]
                self.iteration = data["iteration"] + 1
                # set the random seed
                self.set_rng_state(state=data["rng_state"])
                print(f"Continuing search from iteration {self.iteration}")
        else:
            print("No previous search results found, starting from scratch")
