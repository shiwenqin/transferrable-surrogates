cuda=$1

tasks=(addnist chesseract cifartile geoclassing gutenberg isabella language multnist)

# Standardization options:
# ecdf
# minmax

for task in "${tasks[@]}"; do
    CUDA_VISIBLE_DEVICES=$cuda python bert_loo_tuning.py \
        --model_name answerdotai/ModernBERT-large \
        --data_path DATAPATHHERE/multitask_v2.21.csv \
        --output_path OUTPUTPATHHERE \
        --eval_task $task \
        --encoding arch_long \
        --standardize ecdf \
        --batch_size 2
done