#!/bin/bash
# filepath: vllm/run_vllm_server.sh

export CUDA_VISIBLE_DEVICES=0,1

# MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
# MODEL_NAME="Qwen2.5-VL-7B-Instruct"
MODEL_PATH="/root/siton-tmp/models/EditScore-7B"
MODEL_NAME="EditScore-7B"
# MODEL_PATH="/root/siton-tmp/models/ConsistencyReward-7B"
# MODEL_NAME="ConsistencyReward-7B"
# MODEL_PATH="zai-org/GLM-4.1V-9B-Thinking"
# MODEL_NAME="GLM-4.1V-9B-Thinking"
LOG_FILE="vllm.log"
VLLM_PORT=${VLLM_PORT:-8000}

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
fi

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Launching vLLM on GPU: $CUDA_VISIBLE_DEVICES (num=$NUM_GPUS)"

vllm serve $MODEL_PATH \
    --served-model-name "$MODEL_NAME" \
    --gpu-memory-utilization 0.2 \
    --max-model-len 4096  \
    --host 0.0.0.0 \
    --port $VLLM_PORT \
    --tensor-parallel-size $NUM_GPUS \
    > $LOG_FILE 2>&1 &

echo $! > vllm.pid
echo "vLLM server launched (PID=$(cat vllm.pid))"

READY=0
TIMEOUT=3000
START=$(date +%s)
while [ $(( $(date +%s) - START )) -lt $TIMEOUT ]; do
    if curl -s -f http://127.0.0.1:$VLLM_PORT/v1/models > /dev/null 2>&1; then
        READY=1
        echo "vLLM server Ready!"
        break
    fi
    sleep 1
done

if [ $READY -eq 0 ]; then
    echo "vLLM server launch failed"
    bash stop_vllm_server.sh
    exit 1
fi
