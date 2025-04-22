# Transferrable Surrogates in Expressive Neural Architecture Search Spaces


This is the official codebase for our paper:

[Transferrable Surrogates in Expressive Neural Architecture Search Spaces, Shiwen Qin, Gabriela Kadlecová, Martin Pilát, Shay B. Cohen, Roman Neruda, Elliot J. Crowley, Jovita Lukasik, Linus Ericsson](https://arxiv.org/abs/2504.12971)

Check out our [project page](https://shiwenqin.github.io/TransferrableSurrogate/) too!

## Environment Setup

Please first follow the instructions on [einspace](https://github.com/linusericsson/einspace) to set up environment and data.

```bash
conda activate einspace
pip install torch torchvision torchaudio -U
pip install xgboost lightgbm pympler
```

## Important Folders

```
.
├── configs/                # Config files for searching experiments
├── lm_tuning/              # LM surrogates finetune and inference codes
├── scripts/
│   ├── feature_extraction/ # Scripts for ZCP & GRAF feature extraction
│   └── transfer/           # RF surrogates transfer Learning Evaluation Scripts
├── surrogates/
│   ├── encodings/          # Encoding implementations (str, zcp etc.)
│   └── predictors/         # Surrogate implementations (LM, RF etc.)
```

## Running Experiments

BERT-based surrogate checkpoints & search results can be downloaded from [link](https://figshare.com/s/7df3e41015b341f7326b?file=53351423) 

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

**Test best architecture**:

```bash
python test.py --config configs/tasks/addnist.yaml --device cuda:0 --seed 42 --result_path SEARCH_OUTPUT_PATH_HERE
```

## Notes

1. The first_gen pickle file can be shared across baselines & surrogates to save compute.
2. If a run is crashed & killed, re-runing the same command will by default resume from the last checkpoint.

## Cite us

```
@misc{qin2025transferrablesurrogatesexpressiveneural,
      title={Transferrable Surrogates in Expressive Neural Architecture Search Spaces}, 
      author={Shiwen Qin and Gabriela Kadlecová and Martin Pilát and Shay B. Cohen and Roman Neruda and Elliot J. Crowley and Jovita Lukasik and Linus Ericsson},
      year={2025},
      eprint={2504.12971},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.12971}, 
}
```