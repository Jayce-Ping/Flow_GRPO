#!/usr/bin/env python3
"""
Multi-model vLLM server launcher
Usage: python run_vllm_servers.py --model_paths path1 path2 --model_names name1 name2
"""

import argparse
import subprocess
import json
import time
import sys
from pathlib import Path
import requests


def parse_args():
    parser = argparse.ArgumentParser(description='Launch multiple vLLM servers')
    parser.add_argument('--gpu_ids', type=str, default=None, 
                        help='Comma-separated GPU IDs (e.g., "0,1,2,3"). Auto-detect if not provided')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                        help='GPU memory utilization per model')
    parser.add_argument('--max_model_len', type=int, default=4096,
                        help='Max model length')
    parser.add_argument('--port', type=int, default=8001,
                        help='Base port for first model (others will increment)')
    parser.add_argument('--label', type=str, default='vllm',
                        help='Label prefix for log files and PID tracking')
    parser.add_argument('--model_names', nargs='+', required=True,
                        help='Model names (e.g., model1 model2)')
    parser.add_argument('--model_paths', nargs='+', required=True,
                        help='Model paths (e.g., path/to/model1 path/to/model2)')
    parser.add_argument('--timeout', type=int, default=1800,
                        help='Timeout for server startup (seconds)')
    return parser.parse_args()


def get_available_gpus():
    """Auto-detect available GPUs"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
            capture_output=True, text=True, check=True
        )
        return [int(x.strip()) for x in result.stdout.strip().split('\n')]
    except Exception as e:
        print(f"Error detecting GPUs: {e}")
        return []


def wait_for_server(port, timeout=1800):
    """Wait for vLLM server to be ready"""
    url = f"http://127.0.0.1:{port}/v1/models"
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False


def launch_vllm_server(model_path, model_name, gpu_id, port, args):
    """Launch a single vLLM server instance"""
    log_file = f"{args.label}_{model_name}.log"
    pid_file = f"{args.label}_{model_name}.pid"
    
    env = {**subprocess.os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu_id)}
    
    cmd = [
        'vllm', 'serve', model_path,
        '--served-model-name', model_name,
        '--gpu-memory-utilization', str(args.gpu_memory_utilization),
        '--max-model-len', str(args.max_model_len),
        '--host', '0.0.0.0',
        '--port', str(port),
        '--tensor-parallel-size', '1'
    ]
    
    print(f"\nLaunching {model_name} on GPU {gpu_id}, port {port}")
    print(f"Command: {' '.join(cmd)}")
    
    with open(log_file, 'w') as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=log, env=env)
    
    # Save PID
    with open(pid_file, 'w') as f:
        f.write(str(process.pid))
    
    return process.pid, port


def main():
    args = parse_args()
    
    # Validate inputs
    if len(args.model_names) != len(args.model_paths):
        print("Error: Number of model names must match number of model paths")
        sys.exit(1)
    
    # Get GPU IDs
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    else:
        gpu_ids = get_available_gpus()
        if not gpu_ids:
            print("Error: No GPUs detected")
            sys.exit(1)
    
    num_models = len(args.model_names)
    if len(gpu_ids) < num_models:
        print(f"Warning: {num_models} models but only {len(gpu_ids)} GPUs. Models will share GPUs.")
    
    print(f"Available GPUs: {gpu_ids}")
    print(f"Launching {num_models} models...")
    
    # Launch all servers
    server_info = []
    for i, (model_path, model_name) in enumerate(zip(args.model_paths, args.model_names)):
        gpu_id = gpu_ids[i % len(gpu_ids)]  # Round-robin GPU assignment
        port = args.port + i
        
        pid, port = launch_vllm_server(model_path, model_name, gpu_id, port, args)
        server_info.append({
            'model_name': model_name,
            'model_path': model_path,
            'gpu_id': gpu_id,
            'port': port,
            'pid': pid
        })
    
    # Wait for all servers to be ready
    print("\nWaiting for servers to be ready...")
    all_ready = True
    for info in server_info:
        print(f"Checking {info['model_name']} on port {info['port']}...", end=' ')
        if wait_for_server(info['port'], args.timeout):
            print("✓ Ready")
        else:
            print("✗ Failed")
            all_ready = False
    
    # Save server info
    info_file = f"{args.label}_servers.json"
    with open(info_file, 'w') as f:
        json.dump(server_info, f, indent=2)
    
    if all_ready:
        print("\n✓ All servers launched successfully!")
        print(f"\nServer information saved to: {info_file}")
        print("\nEndpoint mapping:")
        for info in server_info:
            print(f"  {info['model_name']}: http://0.0.0.0:{info['port']}")
    else:
        print("\n✗ Some servers failed to launch. Check log files.")
        sys.exit(1)


if __name__ == '__main__':
    main()