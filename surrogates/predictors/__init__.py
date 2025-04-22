import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from surrogates.predictors.ensemble import BaggingEnsemble 
#from surrogates.predictors.autogluon_pred import AutoGluon
from surrogates.predictors.bert_pred import BertPredictor

xgb_args = {
    "tree_method": "hist",
    "subsample": 0.9,
    "n_estimators": 10000,
    "learning_rate": 0.01
}

predictor_cls = {
    'rf': lambda seed, **kwargs: RandomForestRegressor(random_state=seed, **kwargs),
    'xgb': lambda seed, **kwargs: XGBRegressor(random_state=seed, **kwargs),
    'xgb_tuned': lambda seed: XGBRegressor(random_state=seed, **xgb_args),
    'mlp': lambda seed: MLPRegressor([90, 180, 180], learning_rate_init=0.01, max_iter=1000),
    'ensemble_lightgbm': lambda seed: BaggingEnsemble('ensemble_lightgbm', random_state=seed),
    'ensemble_xgb': lambda seed: BaggingEnsemble('ensemble_xgb', random_state=seed),
    'ensemble_mix': lambda seed: BaggingEnsemble('ensemble_mix',random_state=seed),
    #'autogluon': lambda seed: AutoGluon(),
    #'tabpfn': lambda seed: AutoGluon(TabPFN=True),
    'random': lambda seed: RandomSurrogate(random_state=seed),
    #'bert': lambda seed: BertPredictor(random_state=seed),
    #'tabpfn': lambda seed: AutoGluon(TabPFN=True),
}


class RandomSurrogate:
    def __init__(self, min_val=0.0, max_val=1.0, random_state=42):
        self.random_state = np.random.RandomState(random_state)
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, X, y):
        pass

    def predict(self, X):
        return self.random_state.uniform(self.min_val, self.max_val, len(X))
