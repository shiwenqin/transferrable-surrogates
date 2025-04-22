from sklearn import ensemble
import functools
import pandas as pd
import scipy.stats as ss
import glob
import numpy as np
import xgboost as xgb
import multiprocessing
import itertools

@functools.lru_cache(maxsize=None)
def load_data(file, remove_zero_acc=True):
    data = pd.read_csv(file)
    data = data.set_index('id')
    data['synflow'] = data['synflow'].clip(np.finfo(np.float32).min, np.finfo(np.float32).max)

    y = data['accuracy']
    X = data[[c for c in data.columns if c != 'accuracy']]
    
    if remove_zero_acc:
        ixs = y > 0
        y = y[ixs]
        X = X[ixs]
       
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

def eval_fit(real, predicted):
    return {'tau': ss.kendalltau(real, predicted).statistic, 
            'rho': ss.spearmanr(real, predicted).statistic}

DATASETS = ['addnist', 'chesseract', 'cifartile', 'geoclassing', 'gutenberg', 'isabella', 'language', 'multnist']


def test_transfer(clf, ds, agg_type):
    train_data = [load_data(f) for f in glob.glob(f'transfer_data/*') if f'{ds}_' not in f]
    train_X = pd.concat([d[0] for d in train_data])
    train_y = pd.concat([process_targets(d[1], agg_type) for d in train_data])

    test_data = [load_data(f, remove_zero_acc=False) for f in glob.glob(f'transfer_data/{ds}_*')]
    test_X = pd.concat([d[0] for d in test_data])
    test_y = pd.concat([d[1] for d in test_data])

    clf.fit(train_X, train_y)
    py = clf.predict(test_X)

    m = 'xgb'
    if isinstance(clf, ensemble.RandomForestRegressor):
        m = 'rf'

    r = {'target': ds, 'agg_type': agg_type, 'model': m} | eval_fit(test_y, py)
    print(r)
    return r

xgb_args = {
    "tree_method": "hist",
    "subsample": 0.9,
    "n_estimators": 10000,
    "learning_rate": 0.01,
    "n_jobs": 1
}   

if __name__ == '__main__':
    pool = multiprocessing.Pool(8)

    rf = ensemble.RandomForestRegressor(random_state=42)
    xgb_reg = xgb.XGBRegressor(random_state=42, **xgb_args)

    res = pool.starmap(test_transfer, itertools.product([rf, xgb_reg], DATASETS, ['minmax', 'percentile']))

    pd.DataFrame(res).to_csv('leave_one_out_transfer.csv')
