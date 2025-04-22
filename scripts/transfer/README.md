# Transfer Learning Evaluation Scripts

This directory contains the scripts, data, and outputs for the experiments with the transfer learning of the `RandomForestRegressor` and `XGBoostRegressor` models. 

`leave_one_out_transfer.py` runs the experiments in the leave-one-out mode (for each dataset D, train on all other datasets and predict the values on D). Tests different types of aggregations (`minmax`, `percentile`, and `None`). Produces `leave_one_out.csv`.

`evaluate_transfer.py` evaluates different settings of the models and simulates the optimization runs using the static data, it tests multiple different configurations:
    - `remove_zero_acc` - should zero accuracy networks be removed from the data before training?
    - `train` - should the model be re-fit during the optimization runs? In this setting, the model is trained (apart from the selected transfer data) also on the data from the current run - always data up to `limit` are used for training and data between `limit` and `limit+100` are used as test data, the limits can be set in the `LIMITS` constant in the code
    - `use_cifar`/`use_unseen` - should we use data from the cifar10 dataset runs for transfer learning? Should we use data from the other unseen datasets?
    - `agg_type` - what is the effect of different aggregation types? Possible types are `None`, `minmax`, and `percentile`
    - By default, the script tries all these combinations and produces `results_transfer_xgb.csv` and `results_transfer_rf.csv` with results.

`one_to_one_transfer.py` evaluates the transfer learning between different datasets one-to-one - train on one dataset (data from evolutionary run with `seed=1` is used for training) and predict on another one (`seed=0` is used for testing). This script produces the `one_on_one_transfer_rf.csv` and `one_on_one_transfer_xgb.csv` files.

`cifar10_transfer.py` evaluates the transfer between different runs on cifar10, seed=4 is used for testing, seed=0 is unused (reserved for validation), other seeds are used for training. It also evaluates the ZCPs on cifar10 seed=4

All the scripts expect to find the data in the `transfer_data` directory. It should contain one file for each dataset and each seed of the evolutionary run. The file names should be `<dataset>_seed=<seed>_feats.csv`. The scripts produce csv files with the results in the directory, where they are run. These files can then be processed using the `Tranfer_eval.ipynb` notebook to create the tables used in the paper.
