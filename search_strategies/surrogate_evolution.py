import torch
from tqdm import tqdm

from os.path import exists, join, splitext
from search_strategies.evolution import Evolution
from surrogates.evaluators import SelectiveEvaluator


class SurrogateEvolution(Evolution):
    def __init__(
            self,
            evaluation_fn,
            evaluator,
            pcfg,
            limiter,
            input_params,
            refit_steps=1,
            ground_truth_steps=None,
            surrogate_start_iter=0,
            **kwargs
        ):
        self.evaluator = evaluator
        
        super().__init__(evaluation_fn, pcfg, limiter, input_params, **kwargs)
        self.refit_steps = refit_steps

        self.surrogate_start_iter = surrogate_start_iter
        self.ground_truth_steps = ground_truth_steps

    def learn(self, steps):
        print("-------------")
        print("Evolution")
        print(f"Steps: {steps}")
        print("--------------")

        # populate the first generation
        for iteration in tqdm(range(self.iteration, self.population_size), desc="Initialising population", initial=self.iteration, total=self.population_size):
            if self.architecture_seed:
                self.step(iteration, "seed", eval_mode="fitness")
            else:
                self.step(iteration, "sample", eval_mode="fitness")

        if self.iteration < self.population_size:
            self.iteration = self.population_size

        # save the first generation if a path is provided (and it doesn't already exist)
        if self.first_gen_path is not None and not exists(self.first_gen_path):
            self.save_results(self.iteration - 1, is_first_gen=True)

        # fit the surrogate model
        self.evaluator.fit_surrogate(self.population)

        if self.iteration < self.population_size:
            self.iteration = self.population_size

        self.evolve(steps)

    def evolve(self, steps):
        for iteration in tqdm(range(self.iteration, steps), desc="Evolving population", initial=self.iteration, total=steps):
            if self.gen_next_freq is not None and iteration % self.gen_next_freq == 0:
                print(f"Generating and saving next {self.gen_next_n} individuals")
                self.generate_and_save(iteration, eval_mode="fitness")

            # apply surrogate only after `self.surrogate_start_iter` iterations
            if iteration < self.surrogate_start_iter:
                self.step(iteration, "evolve", eval_mode="fitness")
                continue

            sample = None
            # eval ground truth once in `self.ground_truth_steps` iterations
            if self.ground_truth_steps is not None and iteration % self.ground_truth_steps == 0:
                sample = self.sample_individual(iteration, "evolve", eval_mode="fitness")

            self.step(iteration, "evolve", sample=sample)

            if iteration % self.refit_steps == 0:
                self.evaluator.fit_surrogate(self.population)

    def sample_individual(self, iteration, mode, eval_mode="default"):
        results = super().sample_individual(iteration, mode, eval_mode=eval_mode)

        # different results format to indicate if fitness is evaluated or estimated via the surrogate
        reward = results['reward']
        results['reward'] = (reward[0], reward[1]['score'], reward[2], reward[3], reward[1]['type'])

        ind = results['individual']
        ind.accuracy = reward[1]['score']
        ind.reward_type = reward[1]['type']
        return results

    def eval_individual(self, root, eval_mode="default"):
        # start timer
        self.limiter.timer.start()

        # place the check here to avoid large individuals being evaluated or going into surrogate encoding
        if self.batch is not None and self.compile_fn is not None:
            if not self.limiter.check_batch_pass_time(root, self.compile_fn, self.batch, check_memory=True):
                print("Batch pass time or memory exceeded, trying again")
                return None, None
                
        # evaluate the network
        if eval_mode == "fitness":
            reward = self.evaluator.evaluate_fitness(root)
        elif eval_mode == "surrogate":
            reward = self.evaluator.evaluate_surrogate(root)
        elif eval_mode == "default":
            reward = self.evaluator.evaluate(root)
        else:
            raise ValueError(f"Invalid evaluator mode: {eval_mode}")

        eval_duration = self.limiter.timer()
        if reward is None or reward['score'] == 0:
            return None, eval_duration

        return reward, eval_duration

    def save_results(self, iteration, is_first_gen=False):
        super().save_results(iteration, is_first_gen=is_first_gen)
        path = self.first_gen_path if is_first_gen else self.results_path
        if path is None:
            return
        
        if is_first_gen:
            cache_path = f"{splitext(path)[0]}_encoding_cache{splitext(path)[1]}"
        else:
            cache_path = join(self.results_path, "encoding_cache.pkl")
        self.evaluator.surrogate.save_cache(cache_path)

    def load_results(self):
        super().load_results()
        for ind in self.population:
            if not hasattr(ind, 'reward_type'):
                ind.reward_type = 'fitness'

        # load encoder cache from file
        cache_path = join(self.results_path, "encoding_cache.pkl")
        if exists(cache_path):
            load_path = cache_path
        elif self.first_gen_path:
            path = self.first_gen_path
            load_path = f"{splitext(path)[0]}_encoding_cache{splitext(path)[1]}"
            if not exists(load_path):
                return
        else:
            return

        self.evaluator.surrogate.load_cache(load_path)


class ChooseFromSampledEvolution(SurrogateEvolution):
    def __init__(
            self,
            evaluation_fn,
            evaluator,
            pcfg,
            limiter,
            input_params,
            refit_steps=1,
            ground_truth_steps=None,
            surrogate_n_sampled=20,
            surrogate_n_chosen=1,
            **kwargs
    ):
        super().__init__(evaluation_fn, evaluator, pcfg, limiter, input_params, refit_steps=refit_steps,
                         ground_truth_steps=ground_truth_steps, **kwargs)
        
        self.surrogate_n_sampled = surrogate_n_sampled
        self.surrogate_n_chosen = surrogate_n_chosen

    def evolve(self, steps):
        sampled = []
        for iteration in tqdm(range(self.iteration, steps), desc="Evolving population", initial=self.iteration, total=steps):
            if self.gen_next_freq is not None and iteration % self.gen_next_freq == 0:
                print(f"Generating and saving next {self.gen_next_n} individuals")
                self.generate_and_save(iteration, eval_mode="fitness")

            # apply surrogate only after `self.surrogate_start_iter` iterations
            if iteration < self.surrogate_start_iter:
                self.step(iteration, "evolve", eval_mode="fitness")
                continue

            # use surrogate to sample `self.surrogate_n_sampled` individuals,
            # get `self.surrogate_n_chosen` best individuals and evaluate ground truth for them
            if not len(sampled):
                sampled = self.get_surrogate_sample(iteration)
                if not len(sampled):
                    sampled = [self.sample_individual(iteration, "evolve", eval_mode="fitness")]

            # get next individual and update its iteration id (it was set to its sample iteration)
            sample = sampled.pop(0)
            sample['individual'].id = iteration
            self.step(iteration, "evolve", sample=sample)

            if iteration % self.refit_steps == 0:
                self.evaluator.fit_surrogate(self.population)
    
    def get_surrogate_sample(self, iteration):
        sampled = []
        # get n offspring evaluated by the surrogate
        for i in range(self.surrogate_n_sampled):
            print(f"Sample {i}")
            try:
                sampled.append(self.sample_individual(iteration, "evolve"))
            except ValueError as e:
                print(f"Skipping sampling individual {i} due to error: {e}")
                raise
                continue

        return self.get_chosen_sample(sampled)
        
    
    def get_chosen_sample(self, sampled):
        # get best n individuals
        best_n = sorted(sampled, key=lambda x: x['reward'][1], reverse=True)
        best_n = list(best_n)

        chosen = []
        # evaluate ground truth for them
        for i, ind in enumerate(best_n):
            try:
                reward, eval_duration = self.eval_individual(ind['individual'].root, eval_mode="fitness")
            except (ValueError, MemoryError, torch.OutOfMemoryError) as e:
                print(f"Skipping evaluating individual sample {i} due to error: {e}")
                continue

            if reward is None:
                continue

            # fix format
            ind['reward'] = (ind['individual'].root.serialise(), reward['score'], ind['sample_duration'], eval_duration, reward['type'])
            ind['individual'].accuracy = reward['score']
            ind['individual'].reward_type = reward['type']

            # append to the sample, finish if we have enough
            chosen.append(ind)
            if len(chosen) == self.surrogate_n_chosen:
                break

        return chosen


class RejectionFilterEvolution(ChooseFromSampledEvolution):
    def __init__(
            self,
            evaluation_fn,
            evaluator,
            pcfg,
            limiter,
            input_params,
            refit_steps=1,
            ground_truth_steps=None,
            surrogate_n_sampled=20,
            surrogate_n_chosen=1,
            **kwargs
    ):
        assert isinstance(evaluator, SelectiveEvaluator), "Evaluator must be a SelectiveEvaluator"
        super().__init__(evaluation_fn, evaluator, pcfg, limiter, input_params, refit_steps=refit_steps,
                         ground_truth_steps=ground_truth_steps, surrogate_n_sampled=surrogate_n_sampled,
                         surrogate_n_chosen=surrogate_n_chosen, **kwargs)
        
    def get_chosen_sample(self, sampled):
        chosen = []
        # evaluate ground truth for them
        for i, ind in enumerate(sampled):
            # skip those under quantile
            if not self.evaluator.should_evaluate(ind['individual'].root):
                print(f"Skipping individual sample {i} due to rejection.")
                continue
            try:
                reward, eval_duration = self.eval_individual(ind['individual'].root, eval_mode="fitness")
            except (ValueError, MemoryError, torch.OutOfMemoryError) as e:
                print(f"Skipping evaluating individual sample {i} due to error: {e}")
                continue

            if reward is None:
                continue

            # fix format
            ind['reward'] = (ind['individual'].root.serialise(), reward['score'], ind['sample_duration'], eval_duration, reward['type'])
            ind['individual'].accuracy = reward['score']
            ind['individual'].reward_type = reward['type']

            # append to the sample, finish if we have enough
            chosen.append(ind)
            if len(chosen) == self.surrogate_n_chosen:
                break

        return chosen
