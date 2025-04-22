cuda=$1\

# inference on unseennas

tasks=(addnist chesseract cifartile geoclassing gutenberg isabella language multnist)

for task in "${tasks[@]}"; do
    CUDA_VISIBLE_DEVICES=$cuda python bert_inference.py \
        --model_path MODELCHECKPOINTHERE \
        --data_path DATAPATHHERE/multitask_v2.21/$task \
        --output_path OUTPUTPATHHERE/$task \
        --encoding arch_long
done

# inference on cifar10

CUDA_VISIBLE_DEVICES=$cuda python bert_inference.py \
    --model_path MODELCHECKPOINTHERE \
    --data_path DATAPATHHERE/cifar10_v2.21 \
    --output_path OUTPUTPATHHERE/cifar10 \
    --encoding arch_long