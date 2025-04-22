import pandas as pd
import pickle
import torch
from os.path import splitext
from utils import try_save


class SimpleSurrogate:
    def __init__(self, surrogate, encoder, soft_labels=False, cache_scores=True, saved_cache_path=None, fit_on_cached=False):
        self.surrogate = surrogate
        self.encoder = encoder

        self.cache_scores = cache_scores

        if saved_cache_path is not None:
            self.load_cache(saved_cache_path)
        else:
            self.encoding_cache = {}
            self.accuracy_cache = {}
            self.arch_cache = {}

        self.soft_labels = soft_labels
        self.fit_on_cached = fit_on_cached
    
    def fit(self, individuals):
        # get only inds with true evals
        if not self.soft_labels:
            individuals = [individual for individual in individuals if individual.reward_type == 'fitness']

        # encode the individuals, skip those that failed
        enc = self.try_encode(individuals)
        ids = set()
        X, y = [], []
        for x, ind in zip(enc, individuals):
            if x is not None:
                X.append(x)
                y.append(ind.accuracy)
                ids.add(ind.id)
                if self.cache_scores:
                    self.cache_individual(ind, x, ind.accuracy)

        # fit on the whole archive if enabled
        if self.fit_on_cached:
            for i, enc in self.encoding_cache.items():
                if i not in ids:
                    X.append(enc)
                    y.append(self.accuracy_cache[i])
        X = pd.DataFrame(X)

        self.surrogate.fit(X, y)

    def predict(self, individual, batched=False):
        individual = individual if batched else [individual]

        # skip invalid individuals to still support batching
        y = [0.0 for _ in individual]
        idx = []
        X = []
        for i, ind in enumerate(self.try_encode(individual)):
            if ind is not None:
                X.append(ind)
                idx.append(i)
        X = pd.DataFrame(X)

        # fill in successful predictions to correct indices
        if len(X):
            pred = self.surrogate.predict(X)
            for i, p in zip(idx, pred):
                y[i] = p

        return y if batched else y[0]

    def try_encode(self, individuals):
        X = []

        for individual in individuals:
            try:
                x = self.encode_cached(individual)
            except (ValueError, MemoryError, torch.OutOfMemoryError) as e:
                print(f"Skipping encoding individual {individual.id} due to error: {e}")
                x = None
            X.append(x)

        return X
    
    def encode_cached(self, individual):
        if self.cache_scores and is_in_cache(self.encoding_cache, individual):
            return self.encoding_cache[individual.id]
        
        return self.encoder.encode(individual)

    def cache_individual(self, individual, x, y):
        if not has_id(individual):
            return
        
        self.encoding_cache[individual.id] = x
        self.accuracy_cache[individual.id] = y
        self.arch_cache[individual.id] = individual.root

    def save_cache(self, path):
        temp_path, ext = splitext(path)
        temp_path = f"{temp_path}_temp{ext}"
        data = {
            'encoding': self.encoding_cache,
            'accuracy': self.accuracy_cache,
            'arch': self.arch_cache
        }
        try_save(path, temp_path, data)

    def load_cache(self, path):
        with open(path, 'rb') as f:
            cache = pickle.load(f)
        self.encoding_cache = cache['encoding']
        self.accuracy_cache = cache['accuracy']
        self.arch_cache = cache['arch']
    

def has_id(individual):
    return hasattr(individual, 'id') and individual.id is not None


def is_in_cache(cache, individual):
    return has_id(individual) and individual.id in cache