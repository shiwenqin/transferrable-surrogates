cuda=$1
seed=$2

# Model options:
# microsoft/phi-4
# Qwen/Qwen2.5-7B-Instruct
# Qwen/Qwen2.5-14B-Instruct
# meta-llama/Llama-3.2-3B-Instruct

CUDA_VISIBLE_DEVICES=$cuda python few_shot_inference.py \
    --model_name microsoft/phi-4 \
    --data_path DATAPATHHERE/cifar10_v2.21 \
    --num_shots 3 \
    --output_path OUTPUTPATHHERE \
    --random_seed $seed
