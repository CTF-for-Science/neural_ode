#!/usr/bin/env python3
"""
Worker for distributed hyperparameter tuning.

Usage:
    # Single worker (1 GPU)
    python distributed_worker.py --server http://SERVER_IP:5050 --run-opt run_opt.py

    # Multiple workers (e.g., 4 GPUs on this machine)
    python distributed_worker.py --server http://SERVER_IP:5050 --run-opt run_opt.py --num-workers 4
"""

import argparse
import os
import socket
import sys
import tempfile
import threading
import time
import traceback

import requests
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description='Worker for distributed hyperparameter tuning')
    parser.add_argument('--server', type=str, required=True, help='Server URL (e.g., http://localhost:5050)')
    parser.add_argument('--run-opt', type=str, required=True, help='Path to run_opt.py')
    parser.add_argument('--worker-id', type=str, default=None, help='Worker ID (auto-generated if not provided)')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of parallel workers/GPUs (default: 1)')
    parser.add_argument('--heartbeat-interval', type=int, default=60)
    parser.add_argument('--retry-delay', type=int, default=30)
    parser.add_argument('--max-retries', type=int, default=3)
    return parser.parse_args()


class HeartbeatThread(threading.Thread):
    """Background thread for heartbeats."""

    def __init__(self, server_url, worker_id, interval=60):
        super().__init__(daemon=True)
        self.server_url = server_url
        self.worker_id = worker_id
        self.interval = interval
        self.trial_id = None
        self.progress = {}
        self.running = True

    def set_trial(self, trial_id):
        self.trial_id = trial_id
        self.progress = {}

    def update_progress(self, info):
        self.progress = info

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            if self.trial_id is not None:
                try:
                    response = requests.post(
                        f"{self.server_url}/heartbeat",
                        json={
                            'trial_id': self.trial_id,
                            'worker_id': self.worker_id,
                            'progress': self.progress,
                        },
                        timeout=10
                    )
                    if response.status_code != 200:
                        print(f"Heartbeat warning: {response.text}")
                except Exception as e:
                    print(f"Heartbeat error: {e}")

            time.sleep(self.interval)


def get_trial(server_url, worker_id, max_retries=3):
    """Get a trial from server."""
    for attempt in range(max_retries):
        try:
            response = requests.get(
                f"{server_url}/get_trial",
                params={'worker_id': worker_id},
                timeout=30
            )
            return response.json()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error getting trial (attempt {attempt + 1}): {e}")
                time.sleep(5)
            else:
                raise


def report_result(server_url, worker_id, trial_id, success, result, max_retries=3):
    """Report result to server."""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{server_url}/report",
                json={
                    'trial_id': trial_id,
                    'worker_id': worker_id,
                    'success': success,
                    'result': result,
                },
                timeout=60
            )
            return response.json()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error reporting (attempt {attempt + 1}): {e}")
                time.sleep(5)
            else:
                raise


def run_trial(run_opt_path, trial_config):
    """Run a trial using run_opt.py and return results."""

    # Import run_opt.main
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_opt", run_opt_path)
    run_opt = importlib.util.module_from_spec(spec)
    sys.modules["run_opt"] = run_opt
    spec.loader.exec_module(run_opt)

    # Write config to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(trial_config, f)
        config_path = f.name

    try:
        # Run the optimization
        run_opt.main(config_path)

        # Read results
        batch_id = trial_config['model'].get('batch_id', '')
        run_opt_dir = os.path.dirname(os.path.abspath(run_opt_path))
        results_path = os.path.join(run_opt_dir, f'results_{batch_id}.yaml')

        if not os.path.exists(results_path):
            raise RuntimeError(f"Results file not found: {results_path}")

        with open(results_path, 'r') as f:
            results = yaml.safe_load(f)

        # Clean up results file
        os.unlink(results_path)

        # Sum scores from all pairs
        total_score = 0
        for pair in results.get('pairs', []):
            metrics = pair.get('metrics', {})
            for metric_value in metrics.values():
                total_score += metric_value

        return {
            'score': total_score,
            'raw_results': results,
        }

    finally:
        # Clean up config file
        if os.path.exists(config_path):
            os.unlink(config_path)


def run_worker(server, run_opt, worker_id, gpu_id, heartbeat_interval, retry_delay, max_retries):
    """Run a single worker loop."""
    # Set GPU for this worker
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print(f"[{worker_id}] GPU: {gpu_id}")
    print(f"[{worker_id}] Server: {server}")
    print(f"[{worker_id}] run_opt: {run_opt}")
    print()

    # Start heartbeat
    heartbeat = HeartbeatThread(server, worker_id, heartbeat_interval)
    heartbeat.start()

    while True:
        try:
            print(f"[{worker_id}] Requesting trial...")
            trial_info = get_trial(server, worker_id, max_retries)

            if trial_info.get('status') == 'done':
                print(f"[{worker_id}] All trials completed. Exiting.")
                break

            if trial_info.get('status') != 'ok':
                print(f"[{worker_id}] Unexpected response: {trial_info}")
                time.sleep(retry_delay)
                continue

            trial_id = trial_info['trial_id']
            params = trial_info['params']
            trial_config = trial_info['config']

            print(f"\n[{worker_id}] {'='*50}")
            print(f"[{worker_id}] Trial {trial_id}")
            print(f"[{worker_id}] Params: {params}")
            print(f"[{worker_id}] {'='*50}\n")

            heartbeat.set_trial(trial_id)

            try:
                result = run_trial(run_opt, trial_config)
                print(f"\n[{worker_id}] Trial {trial_id} completed: score={result['score']:.4f}")
                report_result(server, worker_id, trial_id, True, result, max_retries)

            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                print(f"\n[{worker_id}] Trial {trial_id} failed: {error_msg}")
                report_result(server, worker_id, trial_id, False, {'error': error_msg}, max_retries)

            heartbeat.set_trial(None)

        except KeyboardInterrupt:
            print(f"\n[{worker_id}] Interrupted. Exiting.")
            break

        except Exception as e:
            print(f"[{worker_id}] Error: {e}")
            traceback.print_exc()
            time.sleep(retry_delay)

    heartbeat.stop()


def main():
    args = parse_args()

    # Verify run_opt.py exists
    if not os.path.exists(args.run_opt):
        print(f"Error: run_opt.py not found: {args.run_opt}")
        sys.exit(1)

    hostname = socket.gethostname()
    base_worker_id = args.worker_id or hostname

    if args.num_workers == 1:
        # Single worker
        worker_id = f"{base_worker_id}_gpu0"
        run_worker(args.server, args.run_opt, worker_id, 0,
                   args.heartbeat_interval, args.retry_delay, args.max_retries)
    else:
        # Multiple workers - spawn processes
        import multiprocessing
        print(f"Spawning {args.num_workers} workers...")

        processes = []
        for gpu_id in range(args.num_workers):
            worker_id = f"{base_worker_id}_gpu{gpu_id}"
            p = multiprocessing.Process(
                target=run_worker,
                args=(args.server, args.run_opt, worker_id, gpu_id,
                      args.heartbeat_interval, args.retry_delay, args.max_retries)
            )
            p.start()
            processes.append(p)
            print(f"Started {worker_id}")

        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            print("\nInterrupted. Terminating all workers...")
            for p in processes:
                p.terminate()
            for p in processes:
                p.join()


if __name__ == '__main__':
    main()
