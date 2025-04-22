task=$1
cuda=$2
seed=$3

python eval_bert_search.py \
    --config configs/tasks/${task,,}.yaml \
    --device cuda:$cuda \
    --path search_obj_results/${task}/seed_${seed}/search_results.pkl \
    --output_path search_obj_results/${task}/seed_${seed}/

python test.py \
    --device cuda:$cuda \
    --config configs/tasks/${task,,}.yaml \
    --seed $seed \
    --results_path search_obj_results/${task}/seed_${seed}/
