#!/bin/bash
# filepath: vllm/stop_vllm_server.sh

VLLM_LABEL=${VLLM_LABEL:-"vllm"}

if [ -f ${VLLM_LABEL}.pid ]; then
    PID=$(cat ${VLLM_LABEL}.pid)
    echo "Shutting down vLLM server (PID=$PID)..."
    kill $PID 2>/dev/null
    
    # Wait for process to end (with timeout)
    TIMEOUT=30
    for i in $(seq 1 $TIMEOUT); do
        if ! kill -0 $PID 2>/dev/null; then
            echo "vLLM server stopped successfully."
            rm ${VLLM_LABEL}.pid
            exit 0
        fi
        sleep 1
    done
    
    # Force kill if still running
    echo "Force killing vLLM server..."
    kill -9 $PID 2>/dev/null
    rm ${VLLM_LABEL}.pid
else
    echo "vLLM server (${VLLM_LABEL}) is not running."
fi