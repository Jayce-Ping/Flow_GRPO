#!/bin/bash
# filepath: vllm/stop_vllm_server.sh

if [ -f vllm.pid ]; then
    PID=$(cat vllm.pid)
    echo "Shutting down vLLM server (PID=$PID)..."
    pkill -P $PID
    # Wait for process to end
    wait $PID 2>/dev/null
    rm vllm.pid
else
    echo "vLLM server is not running."
fi
