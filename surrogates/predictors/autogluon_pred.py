import pandas as pd
import numpy as np



class AutoGluon:
    def __init__(self, TabPFN=False):
        from autogluon.tabular import TabularPredictor
        
        self.label = 'target'
        self.TabPFN = TabPFN
        self.predictor = TabularPredictor(
            label=self.label,
            problem_type='regression',
        )

        tabpfnmix_default = {
        "model_path_regressor": "autogluon/tabpfn-mix-1.0-regressor",
        "n_ensembles": 1,
        "max_epochs": 30,
        }

        self.hyperparameters = {
            "TABPFNMIX": [
                tabpfnmix_default,
            ],
        }

    def fit(self, X: np.array, y: np.array):
        train_data_df = pd.DataFrame(X)
        train_data_df['target'] = y 

        if self.TabPFN:
            self.predictor = self.predictor.fit(
                train_data=train_data_df,
                hyperparameters=self.hyperparameters,
                verbosity=3,
            )
        else:
            self.predictor = self.predictor.fit(
                train_data=train_data_df,
                verbosity=3,
            )


    def predict(self, X: np.array):
        test_data_df = pd.DataFrame(X)
        
        self.predictor.leaderboard(test_data_df, display=True)

        pred = self.predictor.predict(test_data_df)
 
        return pred

