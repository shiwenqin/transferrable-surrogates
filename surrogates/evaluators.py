import numpy as np
import random
from search_state import DerivationTreeNode


class TempIndividual:
    def __init__(self, root):
        self.root = root
        self.id = None


# TODO batched
class SurrogateEvaluator:
    def __init__(self, evaluation_fn, surrogate):
        self.evaluation_fn = evaluation_fn
        self.surrogate = surrogate

    def fit_surrogate(self, individuals):
        self.surrogate.fit(individuals)

    def evaluate_fitness(self, root):
        return {'score': self.evaluation_fn(root), 'type': 'fitness'}

    def evaluate_surrogate(self, root):
        if isinstance(root, DerivationTreeNode):
            root = TempIndividual(root)

        return {'score': self.surrogate.predict(root), 'type': 'surrogate'}

    def evaluate(self, root):
        return self.evaluate_surrogate(root)


class ExploringEvaluator(SurrogateEvaluator):
    def __init__(self, evaluation_fn, surrogate, exploration_factor=0.1):
        super().__init__(evaluation_fn, surrogate)
        self.exploration_factor = exploration_factor

    def evaluate(self, root):
        if random.random() < self.exploration_factor:
            return self.evaluate_fitness(root)
        return self.evaluate_surrogate(root)


class SelectiveEvaluator(SurrogateEvaluator):
    def __init__(self, evaluation_fn, surrogate, only_ground_truth=True, quantile=0.3, is_maximising=True):
        super().__init__(evaluation_fn, surrogate)
        self.only_ground_truth = only_ground_truth
        self.quantile = quantile
        self.threshold = None
        self.is_maximising = is_maximising

    def fit_surrogate(self, individuals):
        if self.only_ground_truth:
            accuracies = [individual.accuracy for individual in individuals if not hasattr(individual, 'reward_type') or individual.reward_type == 'fitness']
        else:
            accuracies = [individual.accuracy for individual in individuals]

        # get quantile
        self.threshold = sorted(accuracies, reverse=self.is_maximising)[int(len(accuracies) * self.quantile)]
        
        return super().fit_surrogate(individuals)

    def should_evaluate(self, root):
        if self.threshold is None:
            return True

        estimate = self.evaluate_surrogate(root)
        should_eval = (estimate['score'] > self.threshold) if self.is_maximising else (estimate['score'] < self.threshold)
        return should_eval