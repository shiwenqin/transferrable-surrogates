#pyright: basic
from sklearn import ensemble, preprocessing
import pandas as pd
import numpy as np
import scipy.stats as ss
import glob
import xgboost
import os
import functools

DATA_DIR = os.getenv('SCRATCHDIR', '.')
DATA_DIR = f'{DATA_DIR}/transfer_data/'
N_CPUS = int(os.getenv('PBS_NCPUS', '16'))

@functools.lru_cache(maxsize=None)
def load_data(file):
    data = pd.read_csv(file)
    data = data.set_index('id')
    data['synflow'] = data['synflow'].clip(np.finfo(np.float32).min, np.finfo(np.float32).max)

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

def prepare_train_data(transfer_files, current_file, transfer_max_ix, current_max_ix, remove_zero_acc=False, agg_type=None):
    dfs_X = []
    dfs_y = []
    for tf in transfer_files:
        df_X, df_y = load_data(tf)
        ixs = (df_X.index < transfer_max_ix)
        if remove_zero_acc:
            ixs = ixs & (df_y > 0.0)
        dfs_X.append(df_X[ixs])
        df_y = df_y[ixs]
        dfs_y.append(process_targets(df_y, agg_type))
    
    if current_max_ix > 0: 
        df_X, df_y = load_data(current_file)
        ixs = (df_X.index < current_max_ix)
        if remove_zero_acc:
            ixs = ixs & (df_y > 0.0)
        dfs_X.append(df_X[ixs])
        dfs_y.append(process_targets(df_y[ixs], agg_type))

    return pd.concat(dfs_X), pd.concat(dfs_y)


def prepare_test_data(current_file, current_min_ix, current_max_ix):
    df_X, df_y = load_data(current_file)
    ixs = (df_X.index < current_max_ix) & (df_X.index >= current_min_ix)

    return df_X[ixs], df_y[ixs]

DATASETS = ['addnist', 'chesseract', 'cifartile', 'geoclassing', 'gutenberg', 'isabella', 'language', 'multnist']
LIMITS = [100, 200, 300, 400, 500, 600, 700, 800, 900]
SEEDS = ['seed=0', 'seed=1']

def eval_fit(real, predicted):
    return {'tau': ss.kendalltau(real, predicted).statistic, 
            'rho': ss.spearmanr(real, predicted).statistic}

def eval_transfer_train(clf, dataset, seed, use_unseen=False, use_cifar=False, remove_zero_acc=False, agg_type=None, tr_limit=100000):
    ret = []
    current_file = f'{DATA_DIR}/{dataset}_{seed}.csv'
    train_files = []
    if use_cifar:
        train_files.extend(glob.glob(f'{DATA_DIR}/cifar10_*'))
    if use_unseen:
        train_files.extend(sum([glob.glob(f'{DATA_DIR}/{ds}_*') for ds in DATASETS if ds != dataset], start=[]))
    for lim in LIMITS:
        res = {'dataset': dataset,
               'seed': int(seed.split('=')[-1]), 
               'limit': lim, 
               'tr_limit': tr_limit,
               'use_unseen': use_unseen,
               'use_cifar': use_cifar,
               'remove_zero_acc': remove_zero_acc,
               'agg_type': agg_type,
               'train_current': True}
        tr_X, tr_y = prepare_train_data(train_files, current_file, tr_limit, lim, remove_zero_acc, agg_type)
        ts_X, ts_y = prepare_test_data(current_file, lim, lim+100)
        clf.fit(tr_X, tr_y)
        py = clf.predict(ts_X)
        ev = eval_fit(py, ts_y)
        res.update(ev)
        print(res)
        ret.append(res)
    return ret
            
def eval_transfer_notrain(clf, dataset, seed, use_unseen=False, use_cifar=False, remove_zero_acc=False, agg_type=None, tr_limit=100000):
    if not use_unseen and not use_cifar:
        return []
    ret = []
    current_file = f'{DATA_DIR}/{dataset}_{seed}.csv'
    train_files = []
    if use_unseen:
        train_files.extend(sum([glob.glob(f'{DATA_DIR}/{ds}_*') for ds in DATASETS if ds != dataset], start=[]))
    if use_cifar:
        train_files.extend(glob.glob(f'{DATA_DIR}/cifar10_*'))
    tr_X, tr_y = prepare_train_data(train_files, current_file, tr_limit, 0, remove_zero_acc, agg_type)
    clf.fit(tr_X, tr_y)
    for lim in LIMITS:
        res = {'dataset': dataset,
               'seed': int(seed.split('=')[-1]), 
               'limit': lim, 
               'tr_limit': tr_limit,
               'use_unseen': use_unseen,
               'use_cifar': use_cifar,
               'remove_zero_acc': remove_zero_acc,
               'agg_type': agg_type,
               'train_current': False}
        ts_X, ts_y = prepare_test_data(current_file, lim, lim+100)
        py = clf.predict(ts_X)
        ev = eval_fit(py, ts_y)
        res.update(ev)
        print(res)
        ret.append(res)
    return ret

if __name__ == '__main__':
    import multiprocessing
    import itertools

    pool = multiprocessing.Pool(N_CPUS)
    
    xgb_args = {
        "tree_method": "hist",
        "subsample": 0.9,
        "n_estimators": 10000,
        "learning_rate": 0.01,
        "n_jobs": 1
    }   

    clf = ensemble.RandomForestRegressor(random_state=42)
    res = pool.starmap(eval_transfer_notrain, itertools.product([clf], DATASETS, SEEDS, [True, False], [True, False], [True, False], [None, 'minmax', 'percentile']))
    res += pool.starmap(eval_transfer_train, itertools.product([clf], DATASETS, SEEDS, [True, False], [True, False], [True, False], [None, 'minmax', 'percentile']))
    df = pd.DataFrame(sum(res, start=[]))
    df.to_csv('results_transfer_rf.csv')
    
    clf = xgboost.XGBRegressor(random_state=42, **xgb_args)
    res = pool.starmap(eval_transfer_notrain, itertools.product([clf], DATASETS, SEEDS, [True, False], [True, False], [True, False], [None, 'minmax', 'percentile']))
    res += pool.starmap(eval_transfer_train, itertools.product([clf], DATASETS, SEEDS, [True, False], [True, False], [True, False], [None, 'minmax', 'percentile']))
    df = pd.DataFrame(sum(res, start=[]))
    df.to_csv('results_transfer_xgb.csv')
