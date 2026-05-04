#!/usr/bin/env python3
"""
Worker that spawns one process per GPU.
Usage: python worker.py --server http://SERVER:5000 --num-gpus 8 --name worker-1
"""
import argparse
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
import requests
import yaml

def run_gpu_worker(gpu_id, server_url, worker_name, repo_path):
    """Worker loop for a single GPU."""
    print(f"[GPU {gpu_id}] Worker ready, connecting to {server_url}")
    while True:
        # Request job from server
        try:
            resp = requests.post(
                f"{server_url}/get_job",
                json={"worker": worker_name, "gpu": gpu_id},
                timeout=10
            )
            config = resp.json()
        except Exception as e:
            print(f"[GPU {gpu_id}] Error getting job: {e}")
            time.sleep(5)
            continue

        if config is None:
            print(f"[GPU {gpu_id}] No more jobs")
            break

        job_id = config["model"]["batch_id"]
        print(f"[GPU {gpu_id}] Starting {job_id}")

        # Save config
        config_file = Path(f"/tmp/{job_id}_config.yaml")
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        # Run training as subprocess with proper env
        log_file = Path(f"/tmp/{job_id}.log")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        with open(log_file, "w") as log_f:
            proc = subprocess.Popen(
                [sys.executable, "run_opt.py", str(config_file)],
                cwd=repo_path,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env
            )

            # Check if process crashed immediately
            time.sleep(2)
            if proc.poll() is not None:
                print(f"[GPU {gpu_id}] Process exited immediately with code {proc.returncode}")
                print(f"[GPU {gpu_id}] Log contents:")
                try:
                    print(log_file.read_text())
                except:
                    pass
                continue

            # Monitor progress by tailing log file
            last_status = "starting"
            while proc.poll() is None:
                if log_file.exists():
                    try:
                        text = log_file.read_text()
                        # Find last epoch line: "Epoch   50 | Train Loss: ..."
                        matches = re.findall(r"Epoch\s+(\d+)", text)
                        if matches:
                            epoch = matches[-1]
                            new_status = f"epoch {epoch}"
                            if new_status != last_status:
                                last_status = new_status
                                try:
                                    requests.post(
                                        f"{server_url}/progress",
                                        json={"job_id": job_id, "status": new_status},
                                        timeout=5
                                    )
                                except:
                                    pass
                    except:
                        pass
                time.sleep(3)

        # Job finished - collect results
        results_file = Path(repo_path) / f"results_{job_id}.yaml"
        result = {"job_id": job_id, "score": -999}

        if results_file.exists():
            try:
                with open(results_file) as f:
                    results = yaml.safe_load(f)
                score = 0
                for pair in results.get("pairs", []):
                    pair_total = sum(pair.get("metrics", {}).values())
                    result[f"pair_{pair['pair_id']}_total"] = pair_total
                    score += pair_total
                result["score"] = score
                result["config"] = config["model"]
                results_file.unlink(missing_ok=True)
            except Exception as e:
                result["error"] = str(e)

        # Send result
        try:
            requests.post(f"{server_url}/result", json=result, timeout=30)
            print(f"[GPU {gpu_id}] Completed {job_id}: score={result['score']:.2f}")
        except Exception as e:
            print(f"[GPU {gpu_id}] Error sending result: {e}")

        # Cleanup
        config_file.unlink(missing_ok=True)
        log_file.unlink(missing_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--name", default="worker")
    parser.add_argument("--repo-path", default=str(Path(__file__).parent.parent))
    args = parser.parse_args()

    print(f"Starting {args.num_gpus} GPU workers connecting to {args.server}")

    threads = []
    for gpu_id in range(args.num_gpus):
        t = threading.Thread(
            target=run_gpu_worker,
            args=(gpu_id, args.server, args.name, args.repo_path)
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("All workers finished")

if __name__ == "__main__":
    main()
