#!/usr/bin/env python3
"""
Worker for distributed hyperparameter tuning.

Usage:
    python distributed_worker.py --server http://SERVER_IP:5000 --run-opt path/to/run_opt.py
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
    parser.add_argument('--server', type=str, required=True, help='Server URL (e.g., http://localhost:5000)')
    parser.add_argument('--run-opt', type=str, required=True, help='Path to run_opt.py')
    parser.add_argument('--worker-id', type=str, default=None, help='Worker ID (auto-generated if not provided)')
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


def main():
    args = parse_args()

    worker_id = args.worker_id or f"{socket.gethostname()}_{os.getpid()}"

    print(f"Worker: {worker_id}")
    print(f"Server: {args.server}")
    print(f"run_opt: {args.run_opt}")
    print()

    # Verify run_opt.py exists
    if not os.path.exists(args.run_opt):
        print(f"Error: run_opt.py not found: {args.run_opt}")
        sys.exit(1)

    # Start heartbeat
    heartbeat = HeartbeatThread(args.server, worker_id, args.heartbeat_interval)
    heartbeat.start()

    while True:
        try:
            print(f"Requesting trial...")
            trial_info = get_trial(args.server, worker_id, args.max_retries)

            if trial_info.get('status') == 'done':
                print("All trials completed. Exiting.")
                break

            if trial_info.get('status') != 'ok':
                print(f"Unexpected response: {trial_info}")
                time.sleep(args.retry_delay)
                continue

            trial_id = trial_info['trial_id']
            params = trial_info['params']
            trial_config = trial_info['config']

            print(f"\n{'='*50}")
            print(f"Trial {trial_id}")
            print(f"Params: {params}")
            print(f"{'='*50}\n")

            heartbeat.set_trial(trial_id)

            try:
                result = run_trial(args.run_opt, trial_config)
                print(f"\nTrial {trial_id} completed: score={result['score']:.4f}")
                report_result(args.server, worker_id, trial_id, True, result, args.max_retries)

            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                print(f"\nTrial {trial_id} failed: {error_msg}")
                report_result(args.server, worker_id, trial_id, False, {'error': error_msg}, args.max_retries)

            heartbeat.set_trial(None)

        except KeyboardInterrupt:
            print("\nInterrupted. Exiting.")
            break

        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            time.sleep(args.retry_delay)

    heartbeat.stop()


if __name__ == '__main__':
    main()
