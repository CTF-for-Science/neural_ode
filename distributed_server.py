#!/usr/bin/env python3
"""
Central server for distributed hyperparameter tuning.

Usage:
    python distributed_server.py --config path/to/config.yaml --port 5000
"""

import argparse
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string

import optuna
from optuna.samplers import TPESampler
import yaml

app = Flask(__name__)

# Global state
study = None
config = None
search_space_config = None
output_dir = None
active_trials = {}  # trial_id -> {worker_id, start_time, last_heartbeat, params}
lock = threading.Lock()

HEARTBEAT_TIMEOUT = 300
TRIAL_TIMEOUT = 7200


def parse_args():
    parser = argparse.ArgumentParser(description='Central server for distributed hyperparameter tuning')
    parser.add_argument('--config', type=str, required=True, help='Path to tuning config YAML')
    parser.add_argument('--port', type=int, default=5050)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--trials', type=int, default=None, help='Override n_trials from config')
    parser.add_argument('--metric', type=str, default='score')
    parser.add_argument('--mode', type=str, default='max', choices=['min', 'max'])
    parser.add_argument('--resume', type=str, default=None, help='Resume from existing study directory')
    parser.add_argument('--heartbeat-timeout', type=int, default=300)
    parser.add_argument('--trial-timeout', type=int, default=7200)
    return parser.parse_args()


def load_config(config_path):
    """Load and validate config file."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    required = ['dataset', 'model', 'hyperparameters']
    for section in required:
        if section not in cfg:
            raise ValueError(f"Missing required section: {section}")

    return cfg


def suggest_params(trial, hp_config):
    """Suggest hyperparameters for a trial based on config."""
    params = {}

    for name, spec in hp_config.items():
        param_type = spec['type']

        if param_type == 'uniform':
            params[name] = trial.suggest_float(name, spec['lower_bound'], spec['upper_bound'])
        elif param_type == 'loguniform':
            params[name] = trial.suggest_float(name, spec['lower_bound'], spec['upper_bound'], log=True)
        elif param_type == 'quniform':
            params[name] = trial.suggest_float(name, spec['lower_bound'], spec['upper_bound'], step=spec['q'])
        elif param_type == 'randint':
            params[name] = trial.suggest_int(name, spec['lower_bound'], spec['upper_bound'])
        elif param_type == 'lograndint':
            params[name] = trial.suggest_int(name, spec['lower_bound'], spec['upper_bound'], log=True)
        elif param_type == 'qrandint':
            params[name] = trial.suggest_int(name, spec['lower_bound'], spec['upper_bound'], step=spec['q'])
        elif param_type == 'choice':
            params[name] = trial.suggest_categorical(name, spec['choices'])
        elif param_type == 'grid_search':
            params[name] = trial.suggest_categorical(name, spec['grid'])
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    return params


def create_study(args, cfg):
    """Create Optuna study."""
    global study, config, search_space_config, output_dir

    config = cfg
    search_space_config = cfg['hyperparameters']

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = cfg.get('model', {}).get('name', 'model')
    dataset_name = cfg['dataset']['name']

    output_dir = Path(f"results/distributed_tune/{model_name}/{dataset_name}/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(cfg, f)

    # Create study with SQLite storage
    storage_path = output_dir / 'study.db'
    storage = f"sqlite:///{storage_path}"

    n_trials = args.trials or cfg.get('model', {}).get('n_trials', 50)

    sampler = TPESampler(seed=42, constant_liar=True)
    study = optuna.create_study(
        study_name=f"{model_name}_{dataset_name}_{timestamp}",
        storage=storage,
        direction='maximize' if args.mode == 'max' else 'minimize',
        sampler=sampler,
        load_if_exists=True,
    )

    # Save study config
    study_config = {
        'study_name': study.study_name,
        'output_dir': str(output_dir),
        'target_trials': n_trials,
        'metric': args.metric,
        'mode': args.mode,
        'config': cfg,
    }
    with open(output_dir / 'study_config.json', 'w') as f:
        json.dump(study_config, f, indent=2)

    print(f"Study: {study.study_name}")
    print(f"Output: {output_dir}")
    print(f"Target trials: {n_trials}")

    return study, n_trials


def resume_study(study_dir):
    """Resume existing study."""
    global study, config, search_space_config, output_dir

    output_dir = Path(study_dir)

    with open(output_dir / 'study_config.json', 'r') as f:
        study_config = json.load(f)

    config = study_config['config']
    search_space_config = config['hyperparameters']

    storage = f"sqlite:///{output_dir / 'study.db'}"
    study = optuna.load_study(study_name=study_config['study_name'], storage=storage)

    print(f"Resumed: {study.study_name}")
    print(f"Completed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")

    return study, study_config['target_trials']


def check_stale_trials():
    """Check for timed out trials."""
    global active_trials

    now = time.time()
    stale = []

    with lock:
        for trial_id, info in list(active_trials.items()):
            if now - info['last_heartbeat'] > HEARTBEAT_TIMEOUT:
                print(f"Trial {trial_id} timed out (no heartbeat)")
                stale.append(trial_id)
            elif now - info['start_time'] > TRIAL_TIMEOUT:
                print(f"Trial {trial_id} exceeded max duration")
                stale.append(trial_id)

        for trial_id in stale:
            del active_trials[trial_id]
            try:
                study.tell(trial_id, state=optuna.trial.TrialState.FAIL)
            except:
                pass


def print_status_table():
    """Print a formatted status table."""
    with lock:
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])

        best = None
        try:
            if study.best_trial:
                best = study.best_value
        except ValueError:
            pass

        # Header
        print("\n" + "="*80)
        parts = [f"Completed: {completed}", f"Running: {len(active_trials)}", f"Failed: {failed}"]
        if best is not None:
            parts.append(f"Best: {best:.4f}")
        print(f"[Status] {' | '.join(parts)}")
        print("-"*80)
        print(f"{'Worker':<25} {'Trial':<8} {'Status':<20} {'Score':<10}")
        print("-"*80)

        # Active workers
        for trial_id, info in active_trials.items():
            worker = info['worker_id']
            progress = info.get('progress', {})
            epoch = progress.get('epoch', '?')
            pair_id = progress.get('pair_id', '?')
            status = f"Epoch {epoch}, Pair {pair_id}"
            print(f"{worker:<25} {trial_id:<8} {status:<20} {'...':<10}")

        # Recent completed (last 5)
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        for t in completed_trials[-5:]:
            worker = t.user_attrs.get('worker_id', 'unknown')
            score = t.value if t.value else 0
            print(f"{worker:<25} {t.number:<8} {'Completed':<20} {score:<10.4f}")

        print("="*80 + "\n")


def background_checker():
    """Background thread for status and stale trial checking."""
    while True:
        time.sleep(30)
        check_stale_trials()
        print_status_table()


DASHBOARD_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Distributed Tuning Dashboard</title>
    <meta http-equiv="refresh" content="10">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }
        h1 { color: #00d4ff; }
        .stats { display: flex; gap: 20px; margin-bottom: 20px; }
        .stat-box { background: #16213e; padding: 20px; border-radius: 8px; min-width: 120px; }
        .stat-box h3 { margin: 0; color: #888; font-size: 14px; }
        .stat-box .value { font-size: 32px; font-weight: bold; margin-top: 5px; }
        .completed .value { color: #00ff88; }
        .running .value { color: #ffaa00; }
        .failed .value { color: #ff4444; }
        .best .value { color: #00d4ff; }
        table { width: 100%; border-collapse: collapse; background: #16213e; border-radius: 8px; overflow: hidden; }
        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #2a2a4a; }
        th { background: #0f3460; color: #00d4ff; }
        tr:hover { background: #1f4068; }
        .status-running { color: #ffaa00; }
        .status-completed { color: #00ff88; }
        .status-failed { color: #ff4444; }
        .progress-bar { background: #333; border-radius: 4px; height: 8px; width: 100px; display: inline-block; }
        .progress-fill { background: #00d4ff; height: 100%; border-radius: 4px; }
        .section { margin-top: 30px; }
        .section h2 { color: #00d4ff; border-bottom: 1px solid #2a2a4a; padding-bottom: 10px; }
        .auto-refresh { color: #666; font-size: 12px; }
    </style>
</head>
<body>
    <h1>Distributed Hyperparameter Tuning</h1>
    <p class="auto-refresh">Auto-refreshes every 10 seconds</p>

    <div class="stats">
        <div class="stat-box completed">
            <h3>Completed</h3>
            <div class="value">{{ completed }}</div>
        </div>
        <div class="stat-box running">
            <h3>Running</h3>
            <div class="value">{{ running }}</div>
        </div>
        <div class="stat-box failed">
            <h3>Failed</h3>
            <div class="value">{{ failed }}</div>
        </div>
        <div class="stat-box">
            <h3>Target</h3>
            <div class="value">{{ target }}</div>
        </div>
        <div class="stat-box best">
            <h3>Best Score</h3>
            <div class="value">{{ best_score }}</div>
        </div>
    </div>

    <div class="section">
        <h2>Active Workers ({{ active_workers|length }})</h2>
        {% if active_workers %}
        <table>
            <tr>
                <th>Worker</th>
                <th>Trial</th>
                <th>Progress</th>
                <th>Duration</th>
            </tr>
            {% for w in active_workers %}
            <tr>
                <td>{{ w.worker_id }}</td>
                <td>#{{ w.trial_id }}</td>
                <td class="status-running">{{ w.status }}</td>
                <td>{{ w.duration }}</td>
            </tr>
            {% endfor %}
        </table>
        {% else %}
        <p>No active workers</p>
        {% endif %}
    </div>

    <div class="section">
        <h2>Recent Completed Trials</h2>
        {% if recent_trials %}
        <table>
            <tr>
                <th>Trial</th>
                <th>Worker</th>
                <th>Score</th>
                <th>Parameters</th>
            </tr>
            {% for t in recent_trials %}
            <tr>
                <td>#{{ t.number }}</td>
                <td>{{ t.worker }}</td>
                <td class="status-completed">{{ t.score }}</td>
                <td>{{ t.params }}</td>
            </tr>
            {% endfor %}
        </table>
        {% else %}
        <p>No completed trials yet</p>
        {% endif %}
    </div>

    {% if best_params %}
    <div class="section">
        <h2>Best Parameters</h2>
        <table>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
            </tr>
            {% for k, v in best_params.items() %}
            <tr>
                <td>{{ k }}</td>
                <td>{{ v }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    {% endif %}
</body>
</html>
'''


@app.route('/')
def dashboard():
    """Web dashboard for monitoring."""
    with lock:
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
        running = len(active_trials)

        with open(output_dir / 'study_config.json', 'r') as f:
            study_config = json.load(f)
        target = study_config['target_trials']

        best_score = '-'
        best_params = {}
        try:
            if study.best_trial:
                best_score = f"{study.best_value:.4f}"
                best_params = study.best_params
        except ValueError:
            pass

        # Active workers
        active_workers = []
        now = time.time()
        for trial_id, info in active_trials.items():
            progress = info.get('progress', {})
            epoch = progress.get('epoch', '?')
            pair_id = progress.get('pair_id', '?')
            duration_sec = int(now - info['start_time'])
            duration = f"{duration_sec // 60}m {duration_sec % 60}s"
            active_workers.append({
                'worker_id': info['worker_id'],
                'trial_id': trial_id,
                'status': f"Epoch {epoch}, Pair {pair_id}",
                'duration': duration,
            })

        # Recent completed trials
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        recent_trials = []
        for t in completed_trials[-10:][::-1]:  # Last 10, reversed
            recent_trials.append({
                'number': t.number,
                'worker': t.user_attrs.get('worker_id', 'unknown'),
                'score': f"{t.value:.4f}" if t.value else '-',
                'params': ', '.join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in t.params.items()),
            })

        return render_template_string(
            DASHBOARD_HTML,
            completed=completed,
            running=running,
            failed=failed,
            target=target,
            best_score=best_score,
            best_params=best_params,
            active_workers=active_workers,
            recent_trials=recent_trials,
        )


@app.route('/get_trial', methods=['GET'])
def get_trial():
    """Worker requests a trial."""
    global active_trials

    worker_id = request.args.get('worker_id', 'unknown')

    with lock:
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

        # Load target_trials from study_config
        with open(output_dir / 'study_config.json', 'r') as f:
            study_config = json.load(f)
        target = study_config['target_trials']

        if completed >= target:
            return jsonify({'status': 'done', 'message': 'All trials completed'})

        try:
            trial = study.ask()
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

        params = suggest_params(trial, search_space_config)

        active_trials[trial.number] = {
            'worker_id': worker_id,
            'start_time': time.time(),
            'last_heartbeat': time.time(),
            'params': params,
        }

        # Build config for worker (full config with suggested params merged into model section)
        trial_config = {
            'dataset': config['dataset'].copy(),
            'model': config['model'].copy(),
        }
        for k, v in params.items():
            trial_config['model'][k] = v
        trial_config['model']['batch_id'] = str(trial.number)

        print(f"Assigned trial {trial.number} to {worker_id}: {params}")

        return jsonify({
            'status': 'ok',
            'trial_id': trial.number,
            'params': params,
            'config': trial_config,
        })


@app.route('/heartbeat', methods=['POST'])
def heartbeat():
    """Worker heartbeat."""
    data = request.get_json(silent=True) or {}
    trial_id = data.get('trial_id')

    with lock:
        if trial_id in active_trials:
            active_trials[trial_id]['last_heartbeat'] = time.time()
            active_trials[trial_id]['progress'] = data.get('progress', {})
            return jsonify({'status': 'ok'})
        return jsonify({'status': 'error', 'message': 'Trial not found'}), 404


@app.route('/report', methods=['POST'])
def report():
    """Worker reports results."""
    global active_trials

    data = request.get_json(silent=True) or {}
    trial_id = data.get('trial_id')
    success = data.get('success', False)
    result = data.get('result', {})
    worker_id = data.get('worker_id', 'unknown')

    with lock:
        if trial_id not in active_trials:
            return jsonify({'status': 'error', 'message': 'Trial not found'}), 404

        trial_info = active_trials.pop(trial_id)

        if success:
            score = result.get('score', 0)
            try:
                study.tell(trial_id, score)

                # Save trial results
                trial = study.trials[trial_id]
                trial.set_user_attr('worker_id', worker_id)
                for k, v in result.items():
                    if k != 'score':
                        trial.set_user_attr(k, v)

                # Save to file
                results_dir = output_dir / 'trial_results'
                results_dir.mkdir(exist_ok=True)
                with open(results_dir / f'trial_{trial_id}.json', 'w') as f:
                    json.dump({'params': trial_info['params'], 'result': result}, f, indent=2)

                print(f"Trial {trial_id} completed: score={score:.4f} (worker: {worker_id})")
                save_results()

            except Exception as e:
                print(f"Error recording trial {trial_id}: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        else:
            error = result.get('error', 'Unknown')
            print(f"Trial {trial_id} failed: {error} (worker: {worker_id})")
            try:
                study.tell(trial_id, state=optuna.trial.TrialState.FAIL)
            except:
                pass

        return jsonify({'status': 'ok'})


@app.route('/status', methods=['GET'])
def status():
    """Get current status."""
    with lock:
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
        running = len(active_trials)

        best_value = None
        best_params = None
        try:
            if study.best_trial:
                best_value = study.best_value
                best_params = study.best_params
        except ValueError:
            pass

        with open(output_dir / 'study_config.json', 'r') as f:
            study_config = json.load(f)

        return jsonify({
            'study_name': study.study_name,
            'target_trials': study_config['target_trials'],
            'completed': completed,
            'failed': failed,
            'running': running,
            'workers': list(set(t['worker_id'] for t in active_trials.values())),
            'best_value': best_value,
            'best_params': best_params,
        })


@app.route('/results', methods=['GET'])
def results():
    """Get all results."""
    trials = []
    for t in study.trials:
        trials.append({
            'number': t.number,
            'state': str(t.state),
            'value': t.value,
            'params': t.params,
        })

    best = None
    try:
        if study.best_trial:
            best = {
                'number': study.best_trial.number,
                'value': study.best_value,
                'params': study.best_params,
            }
    except ValueError:
        pass

    return jsonify({
        'study_name': study.study_name,
        'trials': trials,
        'best': best,
    })


def save_results():
    """Save current results to disk."""
    results = {
        'study_name': study.study_name,
        'completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'failed': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
        'best': None,
        'trials': [],
    }

    try:
        if study.best_trial:
            results['best'] = {
                'number': study.best_trial.number,
                'value': study.best_value,
                'params': study.best_params,
            }
    except ValueError:
        pass

    for t in study.trials:
        results['trials'].append({
            'number': t.number,
            'state': str(t.state),
            'value': t.value,
            'params': t.params,
        })

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)


def main():
    global HEARTBEAT_TIMEOUT, TRIAL_TIMEOUT

    args = parse_args()
    HEARTBEAT_TIMEOUT = args.heartbeat_timeout
    TRIAL_TIMEOUT = args.trial_timeout

    if args.resume:
        study, n_trials = resume_study(args.resume)
    else:
        cfg = load_config(args.config)
        study, n_trials = create_study(args, cfg)

    print(f"\nServer: {args.host}:{args.port}")
    print(f"\nDashboard: http://localhost:{args.port}/")
    print(f"\nAPI Endpoints:")
    print(f"  GET  /get_trial  - Request a trial")
    print(f"  POST /heartbeat  - Send heartbeat")
    print(f"  POST /report     - Report results")
    print(f"  GET  /status     - Get status (JSON)")
    print(f"  GET  /results    - Get all results (JSON)")
    print()

    checker = threading.Thread(target=background_checker, daemon=True)
    checker.start()

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == '__main__':
    main()
