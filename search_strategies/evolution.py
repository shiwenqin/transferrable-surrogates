from collections import deque
from copy import deepcopy
from os.path import join, exists, splitext
from os import makedirs
import math
import pickle
import random

from tqdm import tqdm

from search_strategies.random_search import Sampler
from baselines import build_baseline, baseline_dict
from visualise import visualise_derivation_tree
from plot import Plotter
from utils import try_save


class Individual(object):
    """A class representing a model containing an architecture, its modules and its accuracy."""

    def __init__(
        self,
        id,
        parent_id,
        root=None,
        accuracy=None,
        age=0,
        hpo_dict=None,
    ):
        self.id = id
        self.parent_id = parent_id
        self.root = root
        self.accuracy = accuracy
        self.age = age
        self.hpo_dict = hpo_dict

        self.alive = True

    def __repr__(self):
        """Prints a readable version of this bitstring."""
        return f"Individual(id={self.id}, accuracy={self.accuracy}, age={self.age}"

    def __eq__(self, other):
        return self.id == other.id


class Population(deque):
    """A class representing a population of models."""

    def __init__(self, individuals):
        self.individuals = individuals

    def __repr__(self):
        """Prints a readable version of this bitstring."""
        return f"Population(individuals={self.individuals})"

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, idx):
        return self.individuals[idx]

    def __setitem__(self, idx, value):
        self.individuals[idx] = value

    def append(self, individual):
        self.individuals.append(individual)

    def popleft(self):
        individual = self.individuals.pop(0)
        individual.alive = False

    def without(self, individual):
        return Population([i for i in self.individuals if i != individual])

    def max(self, key):
        return max(self.individuals, key=key)

    def sample(self, k):
        return random.choice(self.individuals, k=k)

    def extend(self, individuals):
        self.individuals.extend(individuals)

    def sort(self, key):
        self.individuals.sort(key=key)

    def __iter__(self):
        return iter(self.individuals)

    def __next__(self):
        return next(self.individuals)

    def __contains__(self, item):
        return item in self.individuals

    def index(self, item):
        return self.individuals.index(item)

    def remove(self, item):
        self.individuals.remove(item)

    def tournament_selection(self, k, key):
        sample = []
        while len(sample) < k:
            candidate = random.choice(list(self.individuals))
            sample.append(candidate)
        return max(sample, key=key)

    def age(self):
        for individual in self.individuals:
            individual.age += 1

    def tolist(self):
        return self.individuals


class Evolver(Sampler):
    def __init__(
        self,
        pcfg=None,
        mode="iterative",
        time_limit=300,
        max_id_limit=1000,
        depth_limit=20,
        mem_limit=4096,
        limiter=None,
        mutation_strategy="random",
        mutation_rate=1.0,
        crossover_strategy="one_point",
        crossover_rate=0.5,
        selection_strategy="tournament",
        tournament_size=10,
        elitism=True,
        verbose=False,
    ):
        super().__init__(
            pcfg=pcfg,
            mode=mode,
            time_limit=time_limit,
            max_id_limit=max_id_limit,
            depth_limit=depth_limit,
            mem_limit=mem_limit,
            verbose=verbose
        )
        self.limiter = limiter
        self.mutation_strategy = mutation_strategy
        self.mutation_rate = mutation_rate
        self.crossover_strategy = crossover_strategy
        self.crossover_rate = crossover_rate
        self.selection_strategy = selection_strategy
        self.tournament_size = tournament_size
        self.elitism = elitism

    def evolve(self, population):
        # select the parents (and avoid incest)
        parent1 = self.select(population)
        parent2 = self.select(population.without(parent1))
        print(f"Parent 1: {parent1.accuracy}, {parent1.root}")
        if self.crossover_rate > 0:
            print(f"Parent 2: {parent2.accuracy}, {parent2.root}")
        # crossover the parents
        child = self.crossover(parent1, parent2)
        print(f"Child: {child.root}")
        # mutate the child
        child = self.mutate(child)
        print(f"Mutated child: {child.root}")
        return child.root

    def select(self, population):
        if self.selection_strategy == "tournament":
            return population.tournament_selection(self.tournament_size, key=lambda x: x.accuracy)
        elif self.selection_strategy == "first":
            return population[0]
        elif self.selection_strategy == "last":
            return population[-1]

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            if self.crossover_strategy == "one_point":
                return self.one_point_crossover(parent1, parent2)
            elif self.crossover_strategy == "two_point":
                return self.two_point_crossover(parent1, parent2)
        return parent1

    def one_point_crossover(self, parent1, parent2):
        successes = [False, False]
        tries = 0
        while not any(successes):
            if tries > 10:
                raise RuntimeError("Crossover failed to generate valid children.")
            # filter valid nodes from parents without copying
            valid_nodes1 = [
                node for node in parent1.root.serialise()
                if node.operation.type == 'nonterminal'
                and node.parent is not None  # Exclude root nodes
            ]
            valid_nodes2 = [
                node for node in parent2.root.serialise()
                if node.operation.type == 'nonterminal'
                and node.parent is not None  # Exclude root nodes
            ]

            # ensure valid nodes for swapping
            if not valid_nodes1 or not valid_nodes2:
                raise ValueError("No valid nodes available for crossover.")

            # randomly select nodes
            node1 = random.choice(valid_nodes1)
            node2 = random.choice(valid_nodes2)

            # create deep copies of the parents for the children
            child1_copy = deepcopy(parent1)
            child2_copy = deepcopy(parent2)

            # locate the corresponding nodes in the deep copies
            node1_copy = next(
                node for node in child1_copy.root.serialise() if node.id == node1.id
            )
            node2_copy = next(
                node for node in child2_copy.root.serialise() if node.id == node2.id
            )

            # locate parent and index of the copied nodes
            parent1_ref = node1_copy.parent
            idx1 = node1_copy.parent.children.index(node1_copy)

            parent2_ref = node2_copy.parent
            idx2 = node2_copy.parent.children.index(node2_copy)

            # swap children in the deep copies
            parent1_ref.children[idx1] = node2_copy
            parent2_ref.children[idx2] = node1_copy

            # update parent references for swapped nodes
            node1_copy.parent = parent2_ref
            node2_copy.parent = parent1_ref

            # re-infer all params
            try:
                self.limiter.timer.start()
                child1 = self.sample(
                    input_params=child1_copy.root.input_params,
                    root=child1_copy.root,
                    operations=[
                        node.operation
                        for node in child1_copy.root.serialise()
                    ],
                )
                successes[0] = True
            except:
                pass
            try:
                self.limiter.timer.start()
                child2 = self.sample(
                    input_params=child2_copy.root.input_params,
                    root=child2_copy.root,
                    operations=[
                        node.operation
                        for node in child2_copy.root.serialise()
                    ],
                )
                successes[1] = True
            except:
                pass
            tries += 1

        # create new individuals
        if successes[0]:
            child1_individual = Individual(
                id=max(parent1.id, parent2.id) + 1,
                parent_id=parent1.id,
                root=child1
            )
        if successes[1]:
            child2_individual = Individual(
                id=max(parent1.id, parent2.id) + 1 if not successes[0] else child1_individual.id + 1,
                parent_id=parent2.id,
                root=child2
            )

        # TODO FIXME return both children?
        # if successes[0] and successes[1]:
        #     return [child1_individual, child2_individual]
        if successes[0]:
            return child1_individual
        elif successes[1]:
            return child2_individual

    def two_point_crossover(self, parent1, parent2):
        # TODO Equivalence between one-point and two-point strategies
        # TODO Implement
        return this_is_a_stub

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            if self.mutation_strategy == "random":
                return self.random_mutation(individual, allowed_types=["terminal", "nonterminal"])
            elif self.mutation_strategy == "random_terminal":
                return self.random_mutation(individual, allowed_types=["terminal"])
        return individual

    def random_mutation(self, individual, allowed_types="all"):
        success = False
        while not success:
            try:
                # print(f"Mutating architecture:")
                root = deepcopy(individual.root)
                # print(f"{root}")
                nodes = root.serialise()
                allowed_nodes = [node for node in nodes if node.operation.type in allowed_types]
                # print(allowed_nodes)
                # choose a random node to mutate
                node = random.choice(allowed_nodes)
                # print(f"Mutating node:")
                # print(f"{node}")
                # mutate the node
                root = self.mutate_node(root, node)
                individual = Individual(
                    id=individual.id,
                    parent_id=individual.parent_id,
                    root=root,
                    accuracy=None,
                    age=0,
                    hpo_dict=None,
                )
                success = True
            except Exception as e:
                print("MutationError:", e)
        return individual

    def mutate_node(self, root, node):
        if node.is_leaf():
            # remove the current option from the available options of this node
            if node.available_rules is None:
                options, probs = self.pcfg.get_available_options(node)
                node.available_rules = {
                    "options": options,
                    "probs": probs,
                }
            node.limit_options(node.operation)
            # print(f"Available options: {[op.name for op in node.available_rules['options']]}")
        # sample a new subtree rooted at this node
        self.limiter.timer.start()
        new_node = self.sample(input_params=node.input_params, root=node)
        # print(f"New subtree:")
        # print(f"{new_node}")
        # replace the old node with the new subtree
        node.replace(new_node)
        # print(f"Mutated architecture:")
        # print(f"{root}")
        # test to see if the new architecture is valid
        # this will run through the entire network with the existing operations
        # and raise an error if the network is invalid
        # print(f"Testing mutated architecture:")
        # print(f"Inputs to sample: {root.input_params}")
        # print(f"Root: {root}")
        # print(f"Operations: {[node.operation.name for node in root.serialise()]}")
        self.limiter.timer.start()
        self.sample(
            input_params=root.input_params,
            root=root,
            operations=[
                node.operation
                for node in root.serialise()
            ],
        )
        # print(f"Mutation successful")
        # print(f"New architecture:")
        # print(f"{root}")
        return root


class Evolution:
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
            # evolution specific parameters
            regularised=True, # use regularised evolution
            population_size=100, # number of individuals in the population
            architecture_seed=None,
            mutation_strategy="random", # "random"
            mutation_rate=1.0, # probability of mutation
            crossover_strategy="one_point", # "one_point" or "two_point"
            crossover_rate=0.5, # probability of crossover
            selection_strategy="tournament", # "tournament" or "roulette"
            tournament_size=10, # only used if selection_strategy is "tournament"
            elitism=True, # keep the best individual in the population
            n_tries=None, # number of tries to use in evolution before randomly generating an individual
            first_gen_path=None,
            gen_next_freq=None,
            gen_next_n=20,
            compile_fn=None, # function to compile the architecture to check batch pass time
            batch=None # batch to use for batch pass time
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
        # evolution specific parameters
        self.regularised = regularised
        self.population_size = population_size

        self.architecture_seed = architecture_seed
        if self.architecture_seed:
            self.architecture_seed = (
                architecture_seed.split('+') * 
                math.ceil(self.population_size / len(architecture_seed.split('+')))
            )[:self.population_size]
        self.seed_population = {}
        print(f"Architecture seed: {self.architecture_seed}")
        self.n_tries = n_tries

        self.evolver = Evolver(
            pcfg=pcfg,
            mode=mode,
            time_limit=time_limit,
            max_id_limit=max_id_limit,
            depth_limit=depth_limit,
            mem_limit=mem_limit,
            limiter=limiter,
            mutation_strategy=mutation_strategy,
            mutation_rate=mutation_rate,
            crossover_strategy=crossover_strategy,
            crossover_rate=crossover_rate,
            selection_strategy=selection_strategy,
            tournament_size=tournament_size,
            elitism=elitism,
            verbose=verbose,
        )

        self.rewards = []
        self.iteration = 0
        self.population = Population([])

        self.set_rng_state(seed=self.seed)

        self.first_gen_path = first_gen_path
        self.gen_next_freq = gen_next_freq
        self.gen_next_n = gen_next_n

        self.compile_fn = compile_fn
        self.batch = batch

        if self.continue_search:
            self.load_results()

        # fix for the clock
        self.limiter.timer.start()
        print(f"Initialised Evolution at {self.limiter.timer.start_time}")

    def set_rng_state(self, seed=None, state=None):
        if state:
            random.setstate(state)
        elif seed:
            random.seed(seed)

    def learn(self, steps):
        print("-------------")
        print("Evolution")
        print(f"Steps: {steps}")
        print("--------------")

        # populate the first generation
        for iteration in tqdm(range(self.iteration, self.population_size), desc="Initialising population", initial=self.iteration, total=self.population_size):
            if self.architecture_seed:
                self.step(iteration, "seed")
            else:
                self.step(iteration, "sample")

        if self.iteration < self.population_size:
            self.iteration = self.population_size

        # save the first generation if a path is provided (and it doesn't already exist)
        if self.first_gen_path is not None and not exists(self.first_gen_path):
            self.save_results(self.iteration - 1, is_first_gen=True)

        for iteration in tqdm(range(self.iteration, steps), desc="Evolving population", initial=self.iteration, total=steps):
            if self.gen_next_freq is not None and iteration % self.gen_next_freq == 0:
                print(f"Generating and saving next {self.gen_next_n} individuals")
                self.generate_and_save(iteration)

            self.step(iteration, "evolve")

    def generate_and_save(self, iteration, **kwargs):        
        if not self.results_path:
            print("No results path provided, skipping generating N next individuals...")
            return

        makedirs(self.results_path, exist_ok=True)
        path = join(self.results_path, f"next_{self.gen_next_n}_gen_{iteration}.pkl")
        temp_path = splitext(path)[0] + "_temp.pkl"

        # generate and evaluate N individuals
        n_inds = []

        for _ in range(self.gen_next_n):
            ind = self.sample_individual(iteration, "evolve", **kwargs)
            n_inds.append((ind['individual'], ind['reward'][1]))

        # save the results
        try_save(path, temp_path, n_inds)

    def sample_individual(self, iteration, mode, **kwargs):
        success = False
        n_tries = 0
        while not success:
            n_tries += 1
            try:
                # start timer
                self.limiter.timer.start()
                # sample a new individual
                should_be_random = self.n_tries is not None and n_tries > self.n_tries
                if mode == "seed":
                    seed_arch_name = self.architecture_seed.pop(0)
                    seed_arch = baseline_dict[seed_arch_name]
                    root = build_baseline(seed_arch, self.input_params)
                elif mode == "sample" or should_be_random:
                    root = self.evolver.sample(self.input_params)
                elif mode == "evolve":
                    root = self.evolver.evolve(self.population)
                    print(f"Evolved architecture: {root}")
                sample_duration = self.limiter.timer()

                if not self.check_batch_time_and_memory(root):
                    continue

                # evaluate the individual unless it's from the evaluated seed population
                if mode == 'seed' and seed_arch_name in self.seed_population:
                    print(f"Seed architecture already evaluated: {seed_arch_name}")
                    root, reward, sample_duration, eval_duration = self.seed_population[seed_arch_name]
                else:
                    reward, eval_duration = self.eval_individual(root, **kwargs)
                    # save evaluated seed architecture
                    if mode == "seed":
                        self.seed_population[seed_arch_name] = (root, reward, sample_duration, eval_duration)
                
                if reward is None:
                    continue

                success = True
            except (RuntimeError, MemoryError) as e:
                print(f"Error in generating new individual: {e}")
                #print("GPU or RAM Memory error, trying again")
        
        rew = (root.serialise(), reward, sample_duration, eval_duration)
        ind = Individual(id=iteration, parent_id=None, root=root, accuracy=reward)
        return {'reward': rew, 'individual': ind, 'sample_duration': sample_duration, 'eval_duration': eval_duration}

    def check_batch_time_and_memory(self, root):
        # check if batch pass does not exceed the time limit
        if self.batch is not None and self.compile_fn is not None:
            if not self.limiter.check_batch_pass_time(root, self.compile_fn, self.batch, check_memory=True):
                print("Batch pass time or memory exceeded, trying again")
                return False
        return True

    def eval_individual(self, root, **kwargs):
        # start the timer
        self.limiter.timer.start()

        # evaluate the network
        print(f"Evaluating architecture: {root}")
        reward = self.evaluation_fn(root)
        eval_duration = self.limiter.timer()

        if reward is None or reward == 0:
            return None, eval_duration

        return reward, eval_duration

    def step(self, iteration, mode, sample=None, **kwargs):
        if sample is not None:
            ind = sample
        else:
            ind = self.sample_individual(iteration, mode, **kwargs)

        # add the new individual to the population
        self.rewards.append(ind['reward'])
        self.population.append(ind['individual'])
        print(f"Iteration {iteration}, reward: {ind['reward'][1]:.2f}, sample duration: {ind['sample_duration']:.2f}, eval duration: {ind['eval_duration']:.2f}")
        # print(f"Architecture:")
        # for line in root.serialise():
        #     print(line)

        if self.regularised:
            # remove the oldest individual from the population
            if len(self.population) > self.population_size:
                self.population.popleft()

        # save the results
        self.save_results(iteration)

        self.plot(ind['individual'].root, ind['reward'][1], iteration)

    def plot(self, root, reward, iteration):
        if self.visualise:
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
            #visualise_derivation_tree(
            #    best_arch[0], iteration=f"best_{idx}", score=best_reward, show=False,
            #    save_path=self.figures_path
            #)
            # plot results
            plotter.plot_results("rewards", self.figures_path)
            # plot number of parameters
            plotter.plot_num_params(self.figures_path)
            # plot number of nodes
            plotter.plot_num_nodes(self.figures_path)

    def save_results(self, iteration, is_first_gen=False):
        if is_first_gen:
            # save first gen to a file outside of the results directory
            if not self.first_gen_path:
                return
            path = self.first_gen_path
            temp_path, ext = splitext(self.first_gen_path)
            temp_path = f"{temp_path}_temp{ext}"
        else:
            if not self.results_path:
                return
            makedirs(self.results_path, exist_ok=True)
            path = join(self.results_path, "search_results.pkl")
            temp_path = join(self.results_path, "search_results_temp.pkl")

        # write the search results to a temp file first, then rename it
        data = {
            "rewards": self.rewards,
            "iteration": iteration,
            "population": self.population.tolist(),
            "rng_state": random.getstate(),
        }

        try_save(path, temp_path, data)

    def load_results(self):
        # load the search results
        path = join(self.results_path, "search_results.pkl")
        load_path = None
        if exists(path):
            load_path = path
            print(f"Continuing search from iteration {self.iteration}")
        elif self.first_gen_path and exists(self.first_gen_path):
            # option to load the first generation from a file
            load_path = self.first_gen_path
            print("Loading first generation from file")
        else:
            print("No previous search results found, starting from scratch")
            return

        with open(load_path, "rb") as f:
            data = pickle.load(f)
            self.rewards = data["rewards"]
            self.iteration = data["iteration"] + 1
            self.population = Population(data["population"])
            print(len(self.population))
            # set the random seed
            self.set_rng_state(state=data["rng_state"])
