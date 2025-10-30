#!/bin/bash
# Stop all services
# Usage: bash stop_all.sh

VLLM_LABEL=${VLLM_LABEL:-"vllm"}
VLLM_PORT=${VLLM_PORT:-8000}

echo "🛑 Stopping all services..."

# Stop vLLM servers
for PID_FILE in ${VLLM_LABEL}_*.pid; do
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        MODEL_NAME=$(basename "$PID_FILE" .pid | sed "s/${VLLM_LABEL}_//")
        echo -n "   $MODEL_NAME (PID $PID)... "
        
        if kill -0 $PID 2>/dev/null; then
            kill $PID 2>/dev/null || kill -9 $PID 2>/dev/null
            echo "✅"
        else
            echo "⚠️  Not running"
        fi
        rm -f "$PID_FILE"
    fi
done

# Stop gateway
if [ -f "gateway_${VLLM_LABEL}.tmux" ]; then
    TMUX_SESSION=$(cat gateway_${VLLM_LABEL}.tmux)
    echo -n "   Gateway (tmux: $TMUX_SESSION)... "
    
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        tmux kill-session -t "$TMUX_SESSION" 2>/dev/null
        echo "✅"
    else
        echo "⚠️  Not running"
    fi
    rm -f gateway_${VLLM_LABEL}.tmux
elif [ -f "gateway_${VLLM_LABEL}.pid" ]; then
    GATEWAY_PID=$(cat gateway_${VLLM_LABEL}.pid)
    echo -n "   Gateway (PID $GATEWAY_PID)... "
    
    if kill -0 $GATEWAY_PID 2>/dev/null; then
        kill $GATEWAY_PID 2>/dev/null || kill -9 $GATEWAY_PID 2>/dev/null
        echo "✅"
    else
        echo "⚠️  Not running"
    fi
    rm -f gateway_${VLLM_LABEL}.pid
else
    GATEWAY_PID=$(lsof -ti:$VLLM_PORT 2>/dev/null || true)
    if [ -n "$GATEWAY_PID" ]; then
        echo -n "   Gateway (port $VLLM_PORT)... "
        kill -9 $GATEWAY_PID 2>/dev/null
        echo "✅"
    fi
fi

# Cleanup
rm -f ${VLLM_LABEL}_*.log gateway_${VLLM_LABEL}.log gateway_${VLLM_LABEL}.tmux ${VLLM_LABEL}_servers.json

echo "✅ All services stopped"