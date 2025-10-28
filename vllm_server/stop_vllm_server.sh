#!/bin/bash
# filepath: vllm/stop_vllm_server.sh

# Find all .pid files and stop the corresponding vLLM servers
if [ -f ${VLLM_LABEL}.pid ]; then
    PID=$(cat ${VLLM_LABEL}.pid)
    echo "Shutting down vLLM server (PID=$PID)..."
    pkill -P $PID
    # Wait for process to end
    wait $PID 2>/dev/null
    rm ${VLLM_LABEL}.pid
else
    echo "vLLM server is not running."
fi
