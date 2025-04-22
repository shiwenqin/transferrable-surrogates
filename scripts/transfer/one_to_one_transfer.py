from sklearn import ensemble
import functools
import pandas as pd
import scipy.stats as ss
import glob
import numpy as np
import xgboost as xgb


@functools.lru_cache(maxsize=None)
def load_data(file, remove_zero_acc=True):
    data = pd.read_csv(file)
    data = data.set_index('id')
    #synflow can sometimes be larger than float32.max
    data['synflow'] = data['synflow'].clip(np.finfo(np.float32).min, np.finfo(np.float32).max)
    
    y = data['accuracy']
    X = data[[c for c in data.columns if c != 'accuracy']]
    
    if remove_zero_acc:
        ixs = y > 0
        y = y[ixs]
        X = X[ixs]
        
    return X, y

def eval_fit(real, predicted):
    return {'tau': ss.kendalltau(real, predicted).statistic, 
            'rho': ss.spearmanr(real, predicted).statistic}

DATASETS = ['addnist', 'chesseract', 'cifartile', 'geoclassing', 'gutenberg', 'isabella', 'language', 'multnist', 'cifar10']

xgb_args = {
    "tree_method": "hist",
    "subsample": 0.9,
    "n_estimators": 10000,
    "learning_rate": 0.01,
    "n_jobs": 1
}   
        
clf = xgb.XGBRegressor(random_state=42, **xgb_args)

res = []
for sd in DATASETS + ['cifar10all']:
    if sd != 'cifar10all':
        train_X, train_y = load_data(f'transfer_data/{sd}_seed=1.csv')
    else:
        train_files = [load_data(f) for f in glob.glob('transfer_data/cifar10_*') if 'seed=0' not in f and 'seed=4' not in f]
        train_X = pd.concat([f[0] for f in train_files])
        train_y = pd.concat([f[1] for f in train_files])
    clf.fit(train_X, train_y)
    for td in DATASETS:
        # do not remove zero accuracy networks for test data - we would still get them in search
        test_X, test_y = load_data(f'transfer_data/{td}_seed=0.csv', remove_zero_acc=False)
        py = clf.predict(test_X)
        r = {'cls': 'xgb', 'source': sd, 'target': td} | eval_fit(test_y, py)
        print(r)
        res.append(r)

pd.DataFrame(res).to_csv('one_to_one_transfer_xgb.csv')


clf = ensemble.RandomForestRegressor(random_state=42)
res = []
for sd in DATASETS + ['cifar10all']:
    if sd != 'cifar10all':
        train_X, train_y = load_data(f'transfer_data/{sd}_seed=1.csv')
    else:
        train_files = [load_data(f) for f in glob.glob('transfer_data/cifar10_*') if 'seed=0' not in f and 'seed=4' not in f]
        train_X = pd.concat([f[0] for f in train_files])
        train_y = pd.concat([f[1] for f in train_files])
    clf.fit(train_X, train_y)
    for td in DATASETS:
        test_X, test_y = load_data(f'transfer_data/{td}_seed=0.csv', remove_zero_acc=False)
        py = clf.predict(test_X)
        r = {'cls': 'rf', 'source': sd, 'target': td} | eval_fit(test_y, py)
        print(r)
        res.append(r)

pd.DataFrame(res).to_csv('one_to_one_transfer_rf.csv')
