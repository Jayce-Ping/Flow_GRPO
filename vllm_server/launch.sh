#!/bin/bash
# Unified startup script for vLLM multi-model servers with FastAPI gateway
# Usage: bash start_all.sh

set -e

#=============================================================================
# Configuration via environment variables
#=============================================================================

# Models (comma-separated)
VLLM_MODEL_PATHS=${VLLM_MODEL_PATHS:-"Qwen/Qwen2.5-VL-7B-Instruct,OpenGVLab/InternVL3_5-8B"}
VLLM_MODEL_NAMES=${VLLM_MODEL_NAMES:-"Qwen2.5-VL-7B-Instruct,InternVL3_5-8B"}

# Server settings
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.48}
VLLM_PORT=${VLLM_PORT:-8000}  # Public-facing port
VLLM_LABEL=${VLLM_LABEL:-"vllm"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}

# Internal settings
BACKEND_BASE_PORT=18001
GATEWAY_WORKERS=1

#=============================================================================
# Auto-detect GPUs
#=============================================================================

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
fi

AVAILABLE_GPUS=(${CUDA_VISIBLE_DEVICES//,/ })
NUM_GPUS=${#AVAILABLE_GPUS[@]}

echo "🔍 GPUs: ${AVAILABLE_GPUS[*]} (Total: $NUM_GPUS)"

#=============================================================================
# Parse models
#=============================================================================

IFS=',' read -ra MODEL_PATHS <<< "$VLLM_MODEL_PATHS"
IFS=',' read -ra MODEL_NAMES <<< "$VLLM_MODEL_NAMES"

NUM_MODELS=${#MODEL_PATHS[@]}

if [ $NUM_MODELS -ne ${#MODEL_NAMES[@]} ]; then
    echo "❌ Error: Number of model paths and names must match"
    exit 1
fi

echo "📦 Models: $NUM_MODELS"

#=============================================================================
# Calculate GPU allocation
#=============================================================================

if [ $NUM_MODELS -eq 1 ]; then
    # 单模型：使用所有GPU
    GPUS_PER_MODEL=$NUM_GPUS
    echo "💡 Single model mode: using all $NUM_GPUS GPUs with tensor parallelism"
else
    # 多模型：平均分配GPU
    GPUS_PER_MODEL=$((NUM_GPUS / NUM_MODELS))
    
    if [ $GPUS_PER_MODEL -eq 0 ]; then
        echo "❌ Error: Not enough GPUs ($NUM_GPUS) for $NUM_MODELS models"
        echo "   Each model needs at least 1 GPU"
        exit 1
    fi
    
    echo "💡 Multi-model mode: $GPUS_PER_MODEL GPU(s) per model"
    
    # 为每个模型分配GPU
    GPU_ASSIGNMENT=()
    TP_SIZES=()
    
    for i in "${!MODEL_PATHS[@]}"; do
        START_GPU=$((i * GPUS_PER_MODEL))
        MODEL_GPUS=()
        
        for ((j=0; j<GPUS_PER_MODEL; j++)); do
            GPU_INDEX=$((START_GPU + j))
            if [ $GPU_INDEX -lt $NUM_GPUS ]; then
                MODEL_GPUS+=("${AVAILABLE_GPUS[$GPU_INDEX]}")
            fi
        done
        
        # 将GPU列表转为逗号分隔的字符串
        GPU_STR=$(IFS=,; echo "${MODEL_GPUS[*]}")
        GPU_ASSIGNMENT+=("$GPU_STR")
        TP_SIZES+=(${#MODEL_GPUS[@]})
    done
fi

#=============================================================================
# Cleanup
#=============================================================================

echo "🧹 Cleanup..."

for PID_FILE in ${VLLM_LABEL}_*.pid; do
    [ -f "$PID_FILE" ] && kill $(cat "$PID_FILE") 2>/dev/null || true && rm -f "$PID_FILE"
done

GATEWAY_PID=$(lsof -ti:$VLLM_PORT 2>/dev/null || true)
[ -n "$GATEWAY_PID" ] && kill -9 $GATEWAY_PID 2>/dev/null || true

tmux kill-session -t ${VLLM_LABEL}_gateway 2>/dev/null || true

rm -f ${VLLM_LABEL}_*.log gateway.log gateway.pid gateway.tmux ${VLLM_LABEL}_servers.json
sleep 2

#=============================================================================
# Start vLLM servers
#=============================================================================

echo ""
echo "🚀 Starting vLLM servers..."

PIDS=()
PORTS=()

# 单模型模式：使用所有GPU
if [ $NUM_MODELS -eq 1 ]; then
    MODEL_PATH="${MODEL_PATHS[0]}"
    MODEL_NAME="${MODEL_NAMES[0]}"
    PORT=$VLLM_PORT
    LOG_FILE="${VLLM_LABEL}_${MODEL_NAME}.log"
    PID_FILE="${VLLM_LABEL}_${MODEL_NAME}.pid"
    
    echo "   [Single Model] $MODEL_NAME -> GPUs [${AVAILABLE_GPUS[*]}] (TP=$NUM_GPUS)"
    
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES vllm serve "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        --max-model-len $MAX_MODEL_LEN \
        --host 0.0.0.0 \
        --port $PORT \
        --tensor-parallel-size $NUM_GPUS \
        --trust-remote-code \
        > "$LOG_FILE" 2>&1 &
    
    PID=$!
    echo $PID > "$PID_FILE"
    PIDS+=($PID)
    PORTS+=($PORT)
else
    # 多模型模式：每个模型使用分配的GPU
    for i in "${!MODEL_PATHS[@]}"; do
        MODEL_PATH="${MODEL_PATHS[$i]}"
        MODEL_NAME="${MODEL_NAMES[$i]}"
        GPU_STR="${GPU_ASSIGNMENT[$i]}"
        TP_SIZE="${TP_SIZES[$i]}"
        PORT=$((BACKEND_BASE_PORT + i))
        LOG_FILE="${VLLM_LABEL}_${MODEL_NAME}.log"
        PID_FILE="${VLLM_LABEL}_${MODEL_NAME}.pid"
        
        echo "   [$((i+1))/$NUM_MODELS] $MODEL_NAME -> GPUs [$GPU_STR] (TP=$TP_SIZE) :$PORT"
        
        CUDA_VISIBLE_DEVICES=$GPU_STR vllm serve "$MODEL_PATH" \
            --served-model-name "$MODEL_NAME" \
            --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
            --max-model-len $MAX_MODEL_LEN \
            --host 127.0.0.1 \
            --port $PORT \
            --tensor-parallel-size $TP_SIZE \
            --trust-remote-code \
            > "$LOG_FILE" 2>&1 &
        
        PID=$!
        echo $PID > "$PID_FILE"
        PIDS+=($PID)
        PORTS+=($PORT)
    done
fi

#=============================================================================
# Wait for servers
#=============================================================================

echo ""
echo "⏳ Waiting for servers..."

TIMEOUT=600
START=$(date +%s)
ALL_READY=1

for i in "${!PORTS[@]}"; do
    PORT="${PORTS[$i]}"
    MODEL_NAME="${MODEL_NAMES[$i]}"
    echo -n "   $MODEL_NAME... "
    
    READY=0
    while [ $(( $(date +%s) - START )) -lt $TIMEOUT ]; do
        if curl -s -f http://127.0.0.1:$PORT/v1/models > /dev/null 2>&1; then
            READY=1
            echo "✅"
            break
        fi
        sleep 1
    done
    
    if [ $READY -eq 0 ]; then
        echo "❌"
        ALL_READY=0
    fi
done

if [ $ALL_READY -eq 0 ]; then
    echo "❌ Failed"
    for PID in "${PIDS[@]}"; do kill $PID 2>/dev/null || true; done
    exit 1
fi

#=============================================================================
# Save server info
#=============================================================================

INFO_FILE="${VLLM_LABEL}_servers.json"
echo "[" > "$INFO_FILE"

for i in "${!MODEL_NAMES[@]}"; do
    [ $i -gt 0 ] && echo "," >> "$INFO_FILE"
    PORT_VALUE=$VLLM_PORT
    [ $NUM_MODELS -gt 1 ] && PORT_VALUE=$((BACKEND_BASE_PORT + i))
    
    if [ $NUM_MODELS -eq 1 ]; then
        GPU_INFO="${AVAILABLE_GPUS[*]}"
        TP_VALUE=$NUM_GPUS
    else
        GPU_INFO="${GPU_ASSIGNMENT[$i]}"
        TP_VALUE="${TP_SIZES[$i]}"
    fi
    
    cat >> "$INFO_FILE" << EOF
  {
    "model_name": "${MODEL_NAMES[$i]}",
    "model_path": "${MODEL_PATHS[$i]}",
    "gpu_ids": "$GPU_INFO",
    "port": $PORT_VALUE,
    "pid": ${PIDS[$i]},
    "tensor_parallel_size": $TP_VALUE
  }
EOF
done
echo "]" >> "$INFO_FILE"

#=============================================================================
# Start gateway (only for multi-model mode)
#=============================================================================

if [ $NUM_MODELS -gt 1 ]; then
    echo ""
    echo "🌐 Starting gateway on port $VLLM_PORT..."

    if ! command -v tmux &> /dev/null; then
        echo "⚠️  tmux not found, using nohup instead"
        nohup python vllm_server/gateway_fastapi.py \
            --port "$VLLM_PORT" \
            --label "$VLLM_LABEL" \
            --workers "$GATEWAY_WORKERS" \
            > gateway_${VLLM_LABEL}.log 2>&1 &
        GATEWAY_PID=$!
        echo $GATEWAY_PID > gateway_${VLLM_LABEL}.pid
    else
        tmux kill-session -t ${VLLM_LABEL}_gateway 2>/dev/null || true
        
        tmux new-session -d -s ${VLLM_LABEL}_gateway \
            "python vllm_server/gateway_fastapi.py \
            --port $VLLM_PORT \
            --label $VLLM_LABEL \
            --workers $GATEWAY_WORKERS \
            2>&1 | tee gateway_${VLLM_LABEL}.log"
        
        echo "${VLLM_LABEL}_gateway" > gateway_${VLLM_LABEL}.tmux
    fi

    sleep 3

    if curl -s -f http://127.0.0.1:$VLLM_PORT/ > /dev/null 2>&1; then
        echo "✅ Gateway ready"
    else
        echo "❌ Gateway failed"
        exit 1
    fi
fi

#=============================================================================
# Summary
#=============================================================================

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✅ All services started"
echo "═══════════════════════════════════════════════════════════════"
echo ""

if [ $NUM_MODELS -eq 1 ]; then
    echo "🔗 Direct vLLM Endpoint: http://localhost:$VLLM_PORT"
    echo "📚 API Docs: http://localhost:$VLLM_PORT/docs"
    echo "🎯 Tensor Parallel: $NUM_GPUS GPUs"
else
    echo "🔗 Gateway Endpoint: http://localhost:$VLLM_PORT"
    echo "📚 Gateway Docs: http://localhost:$VLLM_PORT/docs"
    echo "🎯 GPU Allocation: $GPUS_PER_MODEL GPU(s) per model"
fi

echo ""
echo "📊 Models: ${MODEL_NAMES[*]}"
echo ""
echo "🛑 Stop: bash stop_all.sh"
echo ""