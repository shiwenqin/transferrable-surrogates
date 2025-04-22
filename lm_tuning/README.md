# Surrogate LMs Finetune & Inference

## Introduction

This repository contains codes used to finetune LMs and use them for inference.

## Folder Structure

```
.
├── src/                  # Contains all main source code
│   ├── fine_tuning/                 # Contains codes for all fine tuning codes
│   │   ├── bert_loo_tuning.py       # Code for Leave_one_out tuning
│   │   ├── bert_on_the_fly.py       # Code for continous learning
│   │   ├── bert_tuning.py           # Code for fine tuning bert models
│   │   └── llm_tuning_reg.py        # Code for fine tuning llms
│   ├── inference/        # Contains codes for all inferencing codes
│   │   ├── bert_inference_task.py   # Code for inference for a specific task in a mixed data file
│   │   ├── bert_inference.py        # Code for inference for a specific task
│   │   └── few_shot_inference.py    # Code for few-shot prompting
│   └── prompts/          # Contains prompts for few shot prompting
├── scripts/         # Contains scripts for running experiments
│   ├── fine_tuning/
│   │   ├── bert_encoding_study.sh   # Script for bert finetuning and encoding ablation (cifar10)
│   │   ├── bert_loo_tuning.sh       # Script for Leave_one_out tuning
│   │   ├── bert_on_the_fly.sh       # Script for continous learning 
│   │   ├── bert_tuning_task.sh      # Script for bert finetuning on various tasks
│   │   └── llm_tune_reg.sh          # Script for fine tuning llms
│   ├── inferences/
│   │   ├── bert_inference.sh        # Script for inferencing bert tuned models
│   │   └── llm_inference.sh         # Script for llm few-shot inferencing
├── .gitignore         
├── requirements.txt   
└── README.md          

```

## Installation

```bash
pip install -r requirements.txt
```