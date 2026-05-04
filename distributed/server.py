#!/usr/bin/env python3
"""
Central server for distributed hyperparameter tuning.
Usage: python server.py --config tuning_config/config_msfr.yaml --trials 200
"""
import argparse
import json
import random
import threading
import yaml
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
lock = threading.Lock()

state = {
    "queue": [],
    "running": {},     # job_id -> {worker, gpu, status}
    "results": [],
}

@app.route('/get_job', methods=['POST'])
def get_job():
    data = request.json
    worker = data.get("worker", "unknown")
    gpu = data.get("gpu", 0)

    with lock:
        if state["queue"]:
            job = state["queue"].pop(0)
            job_id = job["model"]["batch_id"]
            state["running"][job_id] = {"worker": worker, "gpu": gpu, "status": "starting"}
            return jsonify(job)
        return jsonify(None)

@app.route('/progress', methods=['POST'])
def progress():
    data = request.json
    job_id = data["job_id"]
    with lock:
        if job_id in state["running"]:
            state["running"][job_id]["status"] = data.get("status", "running")
    return jsonify({"status": "ok"})

@app.route('/result', methods=['POST'])
def result():
    data = request.json
    job_id = data.get("job_id")
    with lock:
        if job_id in state["running"]:
            del state["running"][job_id]
        state["results"].append(data)
        with open("results.json", "w") as f:
            json.dump(state["results"], f, indent=2)
        done = len(state["results"])
        total = done + len(state["queue"]) + len(state["running"])
        print(f"[{done}/{total}] {job_id} score={data.get('score', 'N/A'):.2f}")
    return jsonify({"status": "ok"})

@app.route('/status')
def status():
    with lock:
        best = max(state["results"], key=lambda x: x.get("score", -999)) if state["results"] else None
        return jsonify({
            "queue": len(state["queue"]),
            "running": state["running"],
            "done": len(state["results"]),
            "total": len(state["queue"]) + len(state["running"]) + len(state["results"]),
            "best": best,
        })

@app.route('/')
def dashboard():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Distributed Tuning</title>
    <style>
        body { font-family: monospace; background: #1a1a2e; color: #eee; padding: 20px; }
        h1 { color: #00d9ff; }
        .stats { display: flex; gap: 20px; margin: 20px 0; }
        .stat { background: #16213e; padding: 15px; border-radius: 8px; }
        .stat-value { font-size: 24px; color: #00d9ff; }
        table { border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #444; padding: 8px; }
        th { background: #16213e; }
    </style>
    <script>
        function refresh() {
            fetch('/status').then(r => r.json()).then(d => {
                document.getElementById('queue').textContent = d.queue;
                document.getElementById('running').textContent = Object.keys(d.running).length;
                document.getElementById('done').textContent = d.done;
                document.getElementById('best').textContent = d.best ? d.best.score.toFixed(2) : '-';
                let html = '';
                for (let [job, info] of Object.entries(d.running || {})) {
                    html += `<tr><td>${info.worker}</td><td>${info.gpu}</td><td>${job}</td><td>${info.status}</td></tr>`;
                }
                document.getElementById('jobs').innerHTML = html;
            });
        }
        setInterval(refresh, 2000);
    </script>
</head>
<body onload="refresh()">
    <h1>Distributed Tuning</h1>
    <div class="stats">
        <div class="stat"><div>Queue</div><div class="stat-value" id="queue">-</div></div>
        <div class="stat"><div>Running</div><div class="stat-value" id="running">-</div></div>
        <div class="stat"><div>Done</div><div class="stat-value" id="done">-</div></div>
        <div class="stat"><div>Best</div><div class="stat-value" id="best">-</div></div>
    </div>
    <table>
        <thead><tr><th>Worker</th><th>GPU</th><th>Job</th><th>Status</th></tr></thead>
        <tbody id="jobs"></tbody>
    </table>
</body>
</html>
    ''')

def generate_hyperparams(hp_config):
    params = {}
    for name, spec in hp_config.items():
        if spec["type"] == "randint":
            params[name] = random.randint(spec["lower_bound"], spec["upper_bound"] - 1)
        elif spec["type"] == "uniform":
            params[name] = random.uniform(spec["lower_bound"], spec["upper_bound"])
        elif spec["type"] == "loguniform":
            log_low, log_high = np.log(spec["lower_bound"]), np.log(spec["upper_bound"])
            params[name] = float(np.exp(random.uniform(log_low, log_high)))
        elif spec["type"] == "choice":
            params[name] = random.choice(spec["choices"])
    return params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    hp_config = config.pop("hyperparameters")
    base_config = config

    for i in range(args.trials):
        params = generate_hyperparams(hp_config)
        job = {
            "dataset": base_config["dataset"].copy(),
            "model": {**base_config["model"], **params, "batch_id": f"trial_{i:04d}"}
        }
        state["queue"].append(job)

    print(f"Generated {args.trials} jobs, server on port {args.port}")
    app.run(host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
