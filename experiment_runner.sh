#!/bin/bash

usage() {
    echo "Usage: $0 [-c <config>] [-s <seed>] [-p <prefix>] [-o <out_dir>] [-d <dataset>] [-i <start_row>] [-j <end_row>]"
    exit 1
}

while getopts ":f:c:s:p:o:d:g:i:j:" opt; do
    case $opt in
        f) csv_file="$OPTARG" ;;
        c) config="$OPTARG" ;;
        s) seed="$OPTARG" ;;
        p) prefix="$OPTARG" ;;
        o) out_dir="$OPTARG" ;;
        d) dataset="$OPTARG" ;;
        g) first_gen_path="$OPTARG" ;;
        i) start_row="$OPTARG" ;;
        j) end_row="$OPTARG" ;;
        *) usage ;;
    esac
done

if [[ -z "$csv_file" ]]; then
    usage
fi

if ! [[ -f "$csv_file" ]]; then
    echo "Error: File '$csv_file' not found."
    exit 1
fi

header_read=false
row_num=0
while IFS=',' read -r line; do
    ((row_num++))
    if [[ "$header_read" == false ]]; then
        IFS=',' read -r -a keys <<< "$line"
        header_read=true
        continue
    fi
    if [[ -n "$start_row" && "$row_num" -lt "$start_row" ]]; then
        continue
    fi
    if [[ -n "$end_row" && "$row_num" -gt "$end_row" ]]; then
        break
    fi
    IFS=',' read -r -a values <<< "$line"
    args=()
    for i in "${!keys[@]}"; do
        args+=("--${keys[i]}" "${values[i]}")
    done

    echo "--config $config --seed $seed --prefix $prefix ${args[@]}"

    python3 main.py \
        --config $config \
        --seed $seed \
        --dataset $dataset \
        --prefix $prefix \
        --first_gen_path "$first_gen_path"_"$dataset"_seed_"$seed".pkl \
        --figures_path $out_dir/figures \
        --results_path $out_dir/results \
	"${args[@]}" >> /dev/null 2>> $SCRATCHDIR/joberr.txt

    cp $SCRATCHDIR/joberr.txt $out_dir/$PBS_JOBID

done < "$csv_file"
