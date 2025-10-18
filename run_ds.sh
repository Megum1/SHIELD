#!/bin/bash

# GPU configuration
GPU_IDs="1"
export CUDA_VISIBLE_DEVICES=$GPU_IDs

# Parameters
phases=("direct" "cot")
modes=("greedy" "sampling")
image_filepath="data/example.png"

# Activate conda environment
conda activate shield_ds

# List of models to test
models=(
    "deepseek-vl2-tiny"
    "deepseek-vl2-small"
    "deepseek-vl2"
)

# Loop over all combinations
for phase in "${phases[@]}"; do
    for mode in "${modes[@]}"; do
        for model in "${models[@]}"; do
            echo "Running $model | phase=$phase | mode=$mode"
            python test.py \
                --model_id "$model" \
                --phase "$phase" \
                --mode "$mode" \
                --image_filepath "$image_filepath"
            echo "---------------------------------------------"
        done
    done
done
