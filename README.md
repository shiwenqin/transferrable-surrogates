# Transferrable Surrogates in Expressive Neural Architecture Search Spaces

Please first follow the instructions on [einspace](https://github.com/linusericsson/einspace) to set up environment and data.

## Environment Setup

```bash
conda activate einspace
pip install torch torchvision torchaudio -U
pip install xgboost lightgbm pympler
```

## Running Experiments

Examples are given for Dataset AddNIST and seed 42

**Baseline runs**:

```bash
python main.py --config configs/tasks/addnist.yaml --device cuda:0 --seed 42 --first_gen_path first_gen_addnist_42.pkl --surrogate_start_iter 100 
```

**LM Surrogate runs**:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/bert_tasks/addnist.yaml --device cuda:0 --seed 42 --first_gen_path first_gen_addnist_42.pkl --surrogate_start_iter 100 --model_ckp MODEL_CKP_PATH_HERE --model_device cuda:0 --refit_steps 100
```

**RF Surrogate runs**:

```bash
python main.py --config configs/tasks/addnist.yaml --seed 42 --first_gen_path first_gen_addnist_42.pkl --surrogate_start_iter 100 --surrogate rf --refit_steps 20 --fit_on_cached True
```

**RF Transfer runs**:

```bash
python main.py --config configs/tasks/addnist.yaml --seed 42 --first_gen_path first_gen_addnist_42.pkl --surrogate_start_iter 100 --surrogate tr_rf --refit_steps 20 --fit_on_cached True
```

For the default setting search output can be found in `./results` and figures can be found in `./figures`.

## Test best architecture

```bash
python test.py --config configs/tasks/addnist.yaml --device cuda:0 --seed 42 --result_path SEARCH_OUTPUT_PATH_HERE
```

## Notes

1. Bert surrogate checkpoints & search results can be downloaded from [link](https://figshare.com/s/7df3e41015b341f7326b?file=53351423) 
2. The first_gen pickle file can be shared across baselines & surrogates to save compute.
3. If a run is crashed & killed, re-runing the same command will by default resume from the last checkpoint.