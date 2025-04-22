cuda=$1
seed=$2

datasets=(addnist chesseract cifartile geoclassing gutenberg isabella language multnist)

# Pretrained model + training
for dataset in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=$cuda python bert_on_the_fly.py \
        --model_name answerdotai/ModernBERT-large \
        --data_path DATAPATHHERE/multitask_v2.21.csv \
        --target_seed $seed \
        --output_path OUTPUTPATHHERE/${dataset} \
        --task $dataset \
        --batch_size 2 \
        --finetune True 
done

# Tuned model + training
for dataset in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=$cuda python bert_on_the_fly.py \
        --model_name MODELCHECKPOINTHERE \
        --data_path DATAPATHHERE/multitask_v2.21.csv \
        --target_seed $seed \
        --output_path OUTPUTPATHHERE/${dataset} \
        --task $dataset \
        --batch_size 2 \
        --finetune True 
done

# Tuned model + without training
for dataset in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=$cuda python bert_on_the_fly.py \
        --model_name MODELCHECKPOINTHERE \
        --data_path DATAPATHHERE/multitask_v2.21.csv \
        --target_seed $seed \
        --output_path OUTPUTPATHHERE/${dataset} \
        --task $dataset \
        --batch_size 2
done

# Experiments for loo models will be the same as above, just change the model checkpoints