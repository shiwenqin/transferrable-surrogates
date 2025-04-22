cuda=$1
model=$2
encoding=$3

# Model options:
# FacebookAI/roberta-base
# FacebookAI/roberta-large
# answerdotai/ModernBERT-base
# answerdotai/ModernBERT-large

# Encoding options:
# arch
# arch_long
# arch_pytorch

CUDA_VISIBLE_DEVICES=$cuda python bert_tuning.py \
    --model_name $model \
    --data_path DATAPATHHERE/cifar10_v2.21 \
    --output_path OUTPUTPATHHERE \
    --encoding $encoding \
    --wandb_project WANDBPROJECTNAME \
    --batch_size 2 \
    --epochs 5 \
    --discard_zero True