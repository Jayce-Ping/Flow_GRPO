#!/bin/bash

GPUS_PER_NODE=${HOST_GPU_NUM:-8}
NUM_MACHINES=${HOST_NUM:-1}
NUM_PROCESSES=${NODE_NUM:-$((NUM_MACHINES * GPUS_PER_NODE))}
MASTER_PORT=${MASTER_PORT:-19001}
MASTER_ADDR=${CHIEF_IP:-${LOCAL_IP:-127.0.0.1}}
RANK=${INDEX:-0}

accelerate launch \
    --config_file scripts/accelerate_configs/fsdp.yaml \
    --num_machines ${NUM_MACHINES} \
    --num_processes ${NUM_PROCESSES} \
    --machine_rank ${RANK} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    scripts/train_bagel.py \
    --config config/grpo.py:pickscore_bagel