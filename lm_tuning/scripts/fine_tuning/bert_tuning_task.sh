cuda=$1
tasks=(addnist chesseract cifartile geoclassing gutenberg isabella language multnist)

for task in "${tasks[@]}"; do
    CUDA_VISIBLE_DEVICES=$cuda python bert_tuning.py \
        --model_name answerdotai/ModernBERT-large \
        --data_path DATAPATHHERE/multitask_v2.21/${task} \
        --output_path OUTPUTPATHHERE/${task} \
        --encoding arch_long \
        --wandb_project WANDBPROJECTNAME \
        --task $task \
        --batch_size 2 \
        --epochs 15 \
        --discard_zero True
done