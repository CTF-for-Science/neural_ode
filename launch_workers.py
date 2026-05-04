#!/usr/bin/env python3
"""
Launch multiple distributed workers, one per GPU.

Usage:
    python launch_workers.py --server http://128.59.145.47:5050 --run-opt run_opt.py --num-gpus 4
"""

import argparse
import os
import socket
import subprocess
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Launch multiple workers, one per GPU')
    parser.add_argument('--server', type=str, required=True, help='Server URL')
    parser.add_argument('--run-opt', type=str, required=True, help='Path to run_opt.py')
    parser.add_argument('--num-gpus', type=int, required=True, help='Number of GPUs/workers')
    parser.add_argument('--worker-id', type=str, default=None, help='Base worker ID')
    return parser.parse_args()


def main():
    args = parse_args()

    base_id = args.worker_id or socket.gethostname()

    print(f"Launching {args.num_gpus} workers...")

    processes = []
    for gpu_id in range(args.num_gpus):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        env['PYTHONUNBUFFERED'] = '1'

        worker_id = f"{base_id}_gpu{gpu_id}"

        cmd = [
            sys.executable, '-u', 'distributed_worker.py',
            '--server', args.server,
            '--run-opt', args.run_opt,
            '--worker-id', worker_id,
        ]

        p = subprocess.Popen(cmd, env=env, stdout=None, stderr=None)
        processes.append(p)
        print(f"Started {worker_id} (pid {p.pid}, GPU {gpu_id})")
        time.sleep(2)  # Stagger worker starts

    print(f"\nAll {args.num_gpus} workers running. Press Ctrl+C to stop.")

    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("\nStopping all workers...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.wait()
        print("Done.")


if __name__ == '__main__':
    main()
