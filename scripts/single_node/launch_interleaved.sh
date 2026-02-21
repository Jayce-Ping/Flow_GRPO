#!/bin/bash
# Launch script for interleaved GRPO training on multi-GPU

GPUS_PER_NODE=8
NUM_MACHINES=1
NUM_PROCESSES=$((NUM_MACHINES * GPUS_PER_NODE))
MASTER_PORT=19001

accelerate launch --config_file scripts/accelerate_configs/fsdp.yaml \
    --num_machines ${NUM_MACHINES} --num_processes ${NUM_PROCESSES} \
    --main_process_port ${MASTER_PORT} \
    scripts/train_interleaved_grpo.py \
    --config config/interleaved_grpo.py