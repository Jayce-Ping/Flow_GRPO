#!/bin/bash
# filepath: vllm/run_vllm_server.sh

# export CUDA_VISIBLE_DEVICES=1

# VLLM_MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
# VLLM_MODEL_NAME="Qwen2.5-VL-7B-Instruct"
Default_model_path="Qwen/Qwen2.5-VL-7B-Instruct"
Default_model_name="ConsistencyReward-7B"
Default_log_file="vllm.log"

VLLM_MODEL_PATH=${VLLM_MODEL_PATH:-$Default_model_path}
VLLM_MODEL_NAME=${VLLM_MODEL_NAME:-$Default_model_name}
LOG_FILE=${LOG_FILE:-$Default_log_file}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.2}
VLLM_PORT=${VLLM_PORT:-8000}

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
fi

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Launching vLLM on GPU: $CUDA_VISIBLE_DEVICES (num=$NUM_GPUS)"

vllm serve $VLLM_MODEL_PATH \
    --served-model-name "$VLLM_MODEL_NAME" \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len 4096  \
    --host 0.0.0.0 \
    --port $VLLM_PORT \
    --tensor-parallel-size $NUM_GPUS \
    > $LOG_FILE 2>&1 &

echo $! > vllm.pid
echo "vLLM server launched (PID=$(cat vllm.pid))"

READY=0
TIMEOUT=600
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
