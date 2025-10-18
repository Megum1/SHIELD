#!/bin/bash

# GPU configuration
GPU_IDs="0"
export CUDA_VISIBLE_DEVICES=$GPU_IDs

# Parameters
phases=("direct" "cot")
modes=("greedy" "sampling")
image_filepath="data/example.png"

# Activate conda environment
conda activate shield

# List of models to test
models=(
    "llava-1.5-7b"
    "llava-1.5-13b"
    "llava-1.6-7b"
    "llava-1.6-13b"
    "llava-1.6-34b"
    "llama-3.2-11b"
    "internvl-3-1b"
    "internvl-3-2b"
    "internvl-3-8b"
    "internvl-3-14b"
    "internvl-3-38b"
    "qwen2.5-vl-3b"
    "qwen2.5-vl-7b"
    "qwen2.5-vl-32b"
    "kimi-vl-a3b"
    "ovis2-1b"
    "ovis2-2b"
    "ovis2-4b"
    "ovis2-8b"
    "ovis2-16b"
    "ovis2-34b"
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
