# Transferrable Surrogates in Expressive Neural Architecture Search Spaces

This branch is for Surrogate as NAS objective experiments.

## Environment Setup

Please follow the instruction in branch surrogate_exp to set up the environment.

## Running Experiments

Examples are given for Dataset AddNIST and seed 42, on device cuda:0.

```bash
python main.py --config configs/tasks/addnist.yaml --first_gen_path first_gen_addnist_42.pkl --seed 42 --device cuda:0 --surrogate_start_iter 100 --prefix NASobj --model_device cuda:0 --model_ckp MODEL_CKP_PATH
```

For the default setting search output can be found in `./results` and figures can be found in `./figures`.

## Evaluating Search Results

1. Get running best models and train & evaluate them on validation set.

```bash
python eval_bert_search.py --config configs/tasks/addnist.yaml --device cuda:0 --path SEARCH_OUTPUT_DIR/search_results.pkl --output_path search_results.pkl
```

2. Choose the architecture with highest validation accuracy and evaluate it on test set.

```bash
python test.py --config configs/tasks/addnist.yaml --device cuda:0 --seed 42 --result_path SEARCH_OUTPUT_DIR/
```

To evaluate the raw results in `./search_obj_results`, the script `eval.sh` might come in handy.