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
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.48} # 0.45 will cause OOM issue to load 2 models at once on 80GB GPU
VLLM_PORT=${VLLM_PORT:-8000}  # Gateway port (public-facing)
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

# Calculate how many models can fit per GPU
MODELS_PER_GPU=$(awk "BEGIN {print int(1.0 / $GPU_MEMORY_UTILIZATION)}")
echo "💡 Models per GPU: $MODELS_PER_GPU (based on ${GPU_MEMORY_UTILIZATION} utilization)"

# Build GPU assignment array
GPU_ASSIGNMENT=()
for i in "${!MODEL_PATHS[@]}"; do
    GPU_INDEX=$((i / MODELS_PER_GPU % NUM_GPUS))
    GPU_ID="${AVAILABLE_GPUS[$GPU_INDEX]}"
    GPU_ASSIGNMENT+=($GPU_ID)
done

#=============================================================================
# Cleanup
#=============================================================================

echo "🧹 Cleanup..."

for PID_FILE in ${VLLM_LABEL}_*.pid; do
    [ -f "$PID_FILE" ] && kill $(cat "$PID_FILE") 2>/dev/null || true && rm -f "$PID_FILE"
done

GATEWAY_PID=$(lsof -ti:$VLLM_PORT 2>/dev/null || true)
[ -n "$GATEWAY_PID" ] && kill -9 $GATEWAY_PID 2>/dev/null || true

# Kill existing tmux session if exists
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

for i in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH="${MODEL_PATHS[$i]}"
    MODEL_NAME="${MODEL_NAMES[$i]}"
    GPU_ID="${GPU_ASSIGNMENT[$i]}"
    PORT=$((BACKEND_BASE_PORT + i))
    LOG_FILE="${VLLM_LABEL}_${MODEL_NAME}.log"
    PID_FILE="${VLLM_LABEL}_${MODEL_NAME}.pid"
    
    echo "   [$((i+1))/$NUM_MODELS] $MODEL_NAME -> GPU$GPU_ID:$PORT"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID vllm serve "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        --max-model-len $MAX_MODEL_LEN \
        --host 127.0.0.1 \
        --port $PORT \
        --tensor-parallel-size 1 \
        --trust-remote-code \
        > "$LOG_FILE" 2>&1 &
    
    PID=$!
    echo $PID > "$PID_FILE"
    PIDS+=($PID)
    PORTS+=($PORT)
done

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
    cat >> "$INFO_FILE" << EOF
  {
    "model_name": "${MODEL_NAMES[$i]}",
    "model_path": "${MODEL_PATHS[$i]}",
    "gpu_id": ${GPU_ASSIGNMENT[$i]},
    "port": $((BACKEND_BASE_PORT + i)),
    "pid": ${PIDS[$i]}
  }
EOF
done
echo "]" >> "$INFO_FILE"

#=============================================================================
# Start gateway
#=============================================================================

echo ""
echo "🌐 Starting gateway on port $VLLM_PORT..."

# Check if tmux is available
if ! command -v tmux &> /dev/null; then
    echo "⚠️  tmux not found, using nohup instead"
    nohup python3 gateway_fastapi.py \
        --port "$VLLM_PORT" \
        --label "$VLLM_LABEL" \
        --workers "$GATEWAY_WORKERS" \
        > gateway.log 2>&1 &
    GATEWAY_PID=$!
    echo $GATEWAY_PID > gateway.pid
else
    # Kill existing tmux session if exists
    tmux kill-session -t ${VLLM_LABEL}_gateway 2>/dev/null || true
    
    # Start gateway in tmux session
    tmux new-session -d -s ${VLLM_LABEL}_gateway \
        "python3 gateway_fastapi.py \
        --port $VLLM_PORT \
        --label $VLLM_LABEL \
        --workers $GATEWAY_WORKERS \
        2>&1 | tee gateway.log"
    
    # Save tmux session name for later cleanup
    echo "${VLLM_LABEL}_gateway" > gateway.tmux
fi

sleep 3

if curl -s -f http://127.0.0.1:$VLLM_PORT/ > /dev/null 2>&1; then
    echo "✅ Gateway ready"
else
    echo "❌ Gateway failed"
    exit 1
fi

#=============================================================================
# Summary
#=============================================================================

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✅ All services started"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "🔗 Endpoint: http://localhost:$VLLM_PORT"
echo "📚 Docs: http://localhost:$VLLM_PORT/docs"
echo ""
echo "📊 Models: ${MODEL_NAMES[*]}"
echo ""
echo "🛑 Stop: bash stop_all.sh"
echo ""