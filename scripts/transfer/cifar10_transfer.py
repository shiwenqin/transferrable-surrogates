import xgboost as xgb
from sklearn import ensemble
import glob
import pandas as pd
import numpy as np
import scipy.stats as ss

def load_data(file):
    data = pd.read_csv(file)
    data = data.set_index('id')
    
    y = data['accuracy']
    X = data[[c for c in data.columns if c != 'accuracy']]

    return X, y

cifar_files = glob.glob('transfer_data/*cifar10_*')
train_data = [file for file in cifar_files if "seed=4" not in file and "seed=0" not in file]
val_data = [file for file in cifar_files if "seed=4" in file]

train_data = [load_data(file) for file in train_data]
val_data = [load_data(file) for file in val_data]

train_X = pd.concat(d[0] for d in train_data)
train_y = pd.concat(d[1] for d in train_data)

ixs = train_y > 0
train_X = train_X[ixs]
train_y = train_y[ixs]

val_X = pd.concat(d[0] for d in val_data)
val_y = pd.concat(d[1] for d in val_data)

ZCPS = ['grad_norm', 'snip', 'grasp', 'fisher', 'jacob_cov', 'plain', 'synflow']

for zcp in ZCPS:
    pred = val_X[zcp]
    if zcp == 'jacob_cov':
        pred.fillna(pred.median(skipna=True), inplace=True)
    tau = ss.kendalltau(pred, val_y)
    rho = ss.spearmanr(pred, val_y)
    print(f'{zcp} & & {rho.statistic:.3f} & {tau.statistic:.3f} \\\\ ')


clf = ensemble.RandomForestRegressor(random_state=42)
clf.fit(train_X, train_y)
pred = clf.predict(val_X)
tau = ss.kendalltau(pred, val_y)
rho = ss.spearmanr(pred, val_y)
print(f'RF & & {rho.statistic:.3f} & {tau.statistic:.3f} \\\\ ')

xgb_args = {
    "tree_method": "hist",
    "subsample": 0.9,
    "n_estimators": 10000,
    "learning_rate": 0.01
}

clf = xgb.XGBRegressor(random_state=42, **xgb_args)

clf.fit(train_X, train_y)
pred = clf.predict(val_X)
tau = ss.kendalltau(pred, val_y)
rho = ss.spearmanr(pred, val_y)
print(f'XGB & &  {rho.statistic:.3f} & {tau.statistic:.3f} \\\\ ')
