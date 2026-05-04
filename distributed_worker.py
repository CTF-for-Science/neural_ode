#!/usr/bin/env python3
"""
Worker for distributed hyperparameter tuning (single GPU).

Usage:
    python distributed_worker.py --server http://SERVER_IP:5050 --run-opt run_opt.py

For multiple GPUs, use launch_workers.py instead.
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

# Global cache for preloaded data
_data_cache = {}


def preload_dataset(dataset_name, pair_ids):
    """Preload dataset into memory cache to avoid I/O contention."""
    global _data_cache

    if dataset_name in _data_cache:
        return  # Already loaded

    print(f"Preloading {dataset_name} data into memory...", flush=True)

    # Import data loading functions
    import ctf4science.data_module as dm
    _original_load_dataset = dm.load_dataset
    _original_get_training_timesteps = dm.get_training_timesteps

    _data_cache[dataset_name] = {}
    for pair_id in pair_ids:
        try:
            train_data, init_data = _original_load_dataset(dataset_name, pair_id)
            training_ts = _original_get_training_timesteps(dataset_name, pair_id)
            _data_cache[dataset_name][pair_id] = {
                'train_data': train_data,
                'init_data': init_data,
                'training_timesteps': training_ts,
            }
        except Exception as e:
            print(f"Warning: Could not preload pair {pair_id}: {e}")

    print(f"Preloaded {len(_data_cache[dataset_name])} pairs", flush=True)

    # Monkey-patch the data module to use cached data
    def cached_load_dataset(name, pair_id, transpose=False):
        if name in _data_cache and pair_id in _data_cache[name]:
            cached = _data_cache[name][pair_id]
            train_data = cached['train_data']
            init_data = cached['init_data']
            if transpose:
                train_data = [td.T for td in train_data]
                init_data = init_data.T if init_data is not None else None
            # Return deep copies to avoid mutation issues
            import copy
            return copy.deepcopy(train_data), copy.deepcopy(init_data) if init_data is not None else None
        return _original_load_dataset(name, pair_id, transpose)

    def cached_get_training_timesteps(name, pair_id):
        if name in _data_cache and pair_id in _data_cache[name]:
            import copy
            return copy.deepcopy(_data_cache[name][pair_id]['training_timesteps'])
        return _original_get_training_timesteps(name, pair_id)

    dm.load_dataset = cached_load_dataset
    dm.get_training_timesteps = cached_get_training_timesteps
    print("Data loading functions patched to use cache", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Worker for distributed hyperparameter tuning')
    parser.add_argument('--server', type=str, required=True, help='Server URL (e.g., http://localhost:5050)')
    parser.add_argument('--run-opt', type=str, required=True, help='Path to run_opt.py')
    parser.add_argument('--worker-id', type=str, default=None, help='Worker ID (auto-generated if not provided)')
    parser.add_argument('--heartbeat-interval', type=int, default=60)
    parser.add_argument('--retry-delay', type=int, default=30)
    parser.add_argument('--max-retries', type=int, default=3)
    parser.add_argument('--preload-dataset', type=str, default=None, help='Dataset name to preload at startup')
    parser.add_argument('--preload-pairs', type=str, default=None, help='Comma-separated pair IDs to preload (e.g., 1,2,3)')
    return parser.parse_args()


class HeartbeatThread(threading.Thread):
    """Background thread for heartbeats."""

    def __init__(self, server_url, worker_id, run_opt_dir, interval=60):
        super().__init__(daemon=True)
        self.server_url = server_url
        self.worker_id = worker_id
        self.run_opt_dir = run_opt_dir
        self.interval = interval
        self.trial_id = None
        self.batch_id = None
        self.running = True

    def set_trial(self, trial_id, batch_id=None):
        self.trial_id = trial_id
        self.batch_id = batch_id

    def stop(self):
        self.running = False

    def read_progress(self):
        """Read progress from progress file."""
        if not self.batch_id:
            return {}
        try:
            progress_file = os.path.join(self.run_opt_dir, f"progress_{self.batch_id}.yaml")
            if os.path.exists(progress_file):
                with open(progress_file, 'r') as f:
                    return yaml.safe_load(f) or {}
        except:
            pass
        return {}

    def run(self):
        while self.running:
            if self.trial_id is not None:
                try:
                    progress = self.read_progress()
                    response = requests.post(
                        f"{self.server_url}/heartbeat",
                        json={
                            'trial_id': self.trial_id,
                            'worker_id': self.worker_id,
                            'progress': progress,
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

    # Preload dataset if specified (staggers I/O across workers)
    if args.preload_dataset and args.preload_pairs:
        pair_ids = [int(p.strip()) for p in args.preload_pairs.split(',')]
        preload_dataset(args.preload_dataset, pair_ids)

    # Start heartbeat
    run_opt_dir = os.path.dirname(os.path.abspath(args.run_opt))
    heartbeat = HeartbeatThread(args.server, worker_id, run_opt_dir, args.heartbeat_interval)
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

            batch_id = trial_config['model'].get('batch_id', '')
            heartbeat.set_trial(trial_id, batch_id)

            try:
                result = run_trial(args.run_opt, trial_config)
                print(f"\nTrial {trial_id} completed: score={result['score']:.4f}")
                report_result(args.server, worker_id, trial_id, True, result, args.max_retries)

            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                print(f"\nTrial {trial_id} failed: {error_msg}")
                report_result(args.server, worker_id, trial_id, False, {'error': error_msg}, args.max_retries)

            heartbeat.set_trial(None, None)

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
