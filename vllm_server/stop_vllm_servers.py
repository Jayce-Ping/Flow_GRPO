#!/usr/bin/env python3
"""
Stop vLLM servers launched by run_vllm_servers.py
Usage: python stop_vllM_servers.py --label vllm
"""

import argparse
import json
import os
import signal
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Stop vLLM servers')
    parser.add_argument('--label', type=str, default='vllm',
                        help='Label prefix used when launching servers')
    parser.add_argument('--force', action='store_true',
                        help='Force kill (SIGKILL) if graceful shutdown fails')
    parser.add_argument('--timeout', type=int, default=30,
                        help='Timeout for graceful shutdown (seconds)')
    return parser.parse_args()


def stop_process(pid, timeout=30, force=False):
    """Stop a process gracefully, or force kill if needed"""
    try:
        # Check if process exists
        os.kill(pid, 0)
    except OSError:
        return True  # Process already dead
    
    try:
        # Send SIGTERM for graceful shutdown
        print(f"  Sending SIGTERM to PID {pid}...", end=' ')
        os.kill(pid, signal.SIGTERM)
        
        # Wait for process to exit
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                os.kill(pid, 0)  # Check if still alive
                time.sleep(0.5)
            except OSError:
                print("✓ Stopped")
                return True
        
        # Process still alive after timeout
        if force:
            print("timeout, forcing SIGKILL...", end=' ')
            os.kill(pid, signal.SIGKILL)
            time.sleep(1)
            try:
                os.kill(pid, 0)
                print("✗ Failed to kill")
                return False
            except OSError:
                print("✓ Killed")
                return True
        else:
            print("✗ Timeout (use --force to kill)")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def cleanup_files(pattern, label):
    """Remove PID and log files"""
    for ext in ['pid', 'log']:
        files = list(Path('.').glob(f"{label}_*.{ext}"))
        for f in files:
            try:
                f.unlink()
                print(f"  Removed {f}")
            except Exception as e:
                print(f"  Failed to remove {f}: {e}")


def main():
    args = parse_args()
    
    info_file = f"{args.label}_servers.json"
    
    # Try to load server info
    if Path(info_file).exists():
        print(f"Loading server info from {info_file}")
        with open(info_file) as f:
            server_info = json.load(f)
        
        print(f"\nStopping {len(server_info)} servers...")
        all_stopped = True
        
        for info in server_info:
            print(f"\n{info['model_name']} (PID {info['pid']}, port {info['port']})")
            if not stop_process(info['pid'], args.timeout, args.force):
                all_stopped = False
        
        # Remove info file
        try:
            Path(info_file).unlink()
            print(f"\n  Removed {info_file}")
        except:
            pass
            
    else:
        # Fallback: try to stop from PID files
        print(f"Server info file not found, searching for PID files...")
        pid_files = list(Path('.').glob(f"{args.label}_*.pid"))
        
        if not pid_files:
            print(f"No PID files found with label '{args.label}'")
            return
        
        print(f"\nFound {len(pid_files)} PID files")
        all_stopped = True
        
        for pid_file in pid_files:
            try:
                with open(pid_file) as f:
                    pid = int(f.read().strip())
                print(f"\n{pid_file.stem} (PID {pid})")
                stop_process(pid, args.timeout, args.force)
            except Exception as e:
                print(f"  Error reading {pid_file}: {e}")
                all_stopped = False
    
    # Clean up files
    print("\nCleaning up files...")
    cleanup_files(f"{args.label}_*", args.label)
    
    if all_stopped:
        print("\n✓ All servers stopped successfully")
    else:
        print("\n✗ Some servers failed to stop")


if __name__ == '__main__':
    main()