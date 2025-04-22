import pandas as pd
import numpy as np
import scipy.stats as ss
import glob

DATASETS = ['addnist', 'chesseract', 'cifartile', 'geoclassing', 'gutenberg', 'isabella', 'language', 'multnist', 'cifar10']


def load_data(file):
    data = pd.read_csv(file)
    data = data.set_index('id')
    
    y = data['accuracy']
    X = data[[c for c in data.columns if c != 'accuracy']]

    return X, y

def process_targets(y, agg_type=None):
    if agg_type is None:
        return y
    if agg_type == 'minmax':
        min_y, max_y = y.min(), y.max()
        y = (y - min_y)/(max_y - min_y)
    if agg_type == 'percentile':
        ecdf = ss.ecdf(y)
        y[:] = ecdf.cdf.evaluate(y) 
    return y

def prepare_train_data(transfer_files, X, y, agg_type=None):
    dfs_X = []
    dfs_y = []
    for tf in transfer_files:
        df_X, df_y = load_data(tf)
        ixs = df_y > 0.0
        dfs_X.append(df_X[ixs])
        df_y = df_y[ixs]
        dfs_y.append(process_targets(df_y, agg_type))

    df_X, df_y = X, pd.Series(y)
    ixs = df_y > 0.0
    dfs_X.append(df_X[ixs])
    dfs_y.append(process_targets(df_y[ixs], agg_type))

    return pd.concat(dfs_X), pd.concat(dfs_y)

class TransferPredictor:

    def __init__(self, predictor, data_dir, dataset, agg_type):
        self.predictor = predictor
        self.data_dir = data_dir
        self.dataset = dataset
        self.agg_type = agg_type
        self.train_files = sum([glob.glob(f'{self.data_dir}/{ds}_*') for ds in DATASETS if ds != dataset], start=[]) 
        assert self.train_files, "No transfer data found"

    def fit(self, X, y):
        X, y = prepare_train_data(self.train_files, X, y, self.agg_type)
        self.predictor.fit(X, y)

    def predict(self, X):
        return self.predictor.predict(X)
