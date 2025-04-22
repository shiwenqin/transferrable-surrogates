import numpy as np

import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

def get_ensemble_members(model_type="ensemble_xgb", random_state=0):
    ensemble_list = []
    if model_type == "ensemble_xgb":
        for depth in [5, 9, 3]:
            for n_estimators in [200, 500, 800]:
                for learning_rate in [0.01, 0.1, 0.1, 1.0]:
                    ensemble_list.append(
                        xgb.XGBRegressor(
                            max_depth=depth,
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            eval_metric="rmse",
                            random_state = random_state
                        )
                    )
    elif model_type == "ensemble_lightgbm":
        for depth in [5, 9, 3]:
            for n_estimators in [200, 500, 800]:
                for learning_rate in [0.01, 0.1, 0.001, 1.0]:
                    ensemble_list.append(
                        LGBMRegressor(
                            max_depth=depth,
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            random_state = random_state
                        )
                    )
    elif model_type == "ensemble_mix":
        # define all the models
        ensemble_list = [
            xgb.XGBRegressor(random_state = random_state),
            xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.01, random_state = random_state),
            xgb.XGBRegressor(n_estimators=800, max_depth=3, learning_rate=0.1, random_state = random_state),
            xgb.XGBRegressor(n_estimators=200, max_depth=9, learning_rate=1, random_state = random_state),
            LGBMRegressor(random_state = random_state),
            LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.01, random_state = random_state),
            LGBMRegressor(n_estimators=800, max_depth=3, learning_rate=0.1, random_state = random_state),
            LGBMRegressor(n_estimators=200, max_depth=9, learning_rate=1, random_state = random_state),
            LinearRegression(),
            Ridge(random_state = random_state),
            RandomForestRegressor(n_estimators=400, max_depth=5, random_state = random_state),
        ]

    print(ensemble_list)
    return ensemble_list


ensemble_lengths = {"ensemble_xgb": 27, "ensemble_lightgbm": 27, "ensemble_mix": 11}


class BaggingEnsemble:
    def __init__(self, member_model_type, random_state):
        self.ensemble = get_ensemble_members(member_model_type, random_state)

    def fit(self, X: np.array, y: np.array):
        for i, regressor in enumerate(self.ensemble):
            regressor.fit(X, y)

    def predict(self, X: np.array):
        X = np.array(X)
        predictions = []
        for i, regressor in enumerate(self.ensemble):
            predictions.append(regressor.predict(X))
        all_predictions = np.array(predictions).T
        pred = np.mean(all_predictions, axis=-1)
        return pred

