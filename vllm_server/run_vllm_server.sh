#!/bin/bash
# filepath: vllm/run_vllm_server.sh

# Usage:
# Single model: bash run_vllm_server.sh
# Multi-models: VLLM_MODELS="model1:name1,model2:name2" bash run_vllm_server.sh

Default_model_path="Qwen/Qwen2.5-VL-7B-Instruct"
Default_model_name="ConsistencyReward-7B"
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.2}
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_LABEL=${VLLM_LABEL:-"vllm"}

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
fi

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Parse model
if [ -z "$VLLM_MODELS" ]; then
    # Single model
    MODEL_PATHS="$Default_model_path"
    MODEL_NAMES="$Default_model_name"
else
    # Multi-models: "path1:name1,path2:name2,path3:name3"
    MODEL_PATHS=$(echo "$VLLM_MODELS" | tr ',' '\n' | cut -d':' -f1 | tr '\n' ' ')
    MODEL_NAMES=$(echo "$VLLM_MODELS" | tr ',' '\n' | cut -d':' -f2 | tr '\n' ',' | sed 's/,$//')
fi

echo "Launching vLLM on GPU: $CUDA_VISIBLE_DEVICES (num=$NUM_GPUS)"
echo "Models: $MODEL_PATHS"
echo "Names: $MODEL_NAMES"

vllm serve $MODEL_PATHS \
    --served-model-name "$MODEL_NAMES" \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len 4096 \
    --host 0.0.0.0 \
    --port $VLLM_PORT \
    --tensor-parallel-size $NUM_GPUS \
    > ${VLLM_LABEL}.log 2>&1 &

echo $! > ${VLLM_LABEL}.pid
echo "vLLM server launched (PID=$(cat ${VLLM_LABEL}.pid))"

READY=0
TIMEOUT=1800
START=$(date +%s)
while [ $(( $(date +%s) - START )) -lt $TIMEOUT ]; do
    if curl -s -f http://127.0.0.1:$VLLM_PORT/v1/models > /dev/null 2>&1; then
        READY=1
        echo "vLLM server Ready!"
        curl -s http://127.0.0.1:$VLLM_PORT/v1/models | python3 -m json.tool
        break
    fi
    sleep 1
done

if [ $READY -eq 0 ]; then
    echo "vLLM server launch failed"
    [ -f ${VLLM_LABEL}.pid ] && kill $(cat ${VLLM_LABEL}.pid) 2>/dev/null
    exit 1
fi