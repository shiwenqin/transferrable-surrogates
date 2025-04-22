cuda=$1

# Model options:
# meta-llama/Llama-3.2-1B-Instruct
# meta-llama/Llama-3.2-1B
# meta-llama/Llama-3.2-3B

CUDA_VISIBLE_DEVICES=$cuda python llm_tuning_reg.py \
    --model_name meta-llama/Llama-3.2-1B \
    --data_path DATAPATHHERE/cifar10_v2.21 \
    --output_path OUTPUTPATHHERE \
    --wandb_project WANDBPROJECTNAME \
    --batch_size 1 \
    --epochs 5 \