import os
import sys
import yaml
import argparse
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from pathlib import Path
from typing import List, Dict, Any
import datetime
import numpy as np
from ctf4science.data_module import load_dataset, get_prediction_timesteps, parse_pair_ids, get_applicable_plots
from ctf4science.eval_module import evaluate, save_results
from ctf4science.visualization_module import Visualization
from neural_ode import NeuralOde

file_dir = Path(__file__).parent

sys.path.insert(0, str(file_dir))

from run_opt import main as run_opt_main

def extract_per_pair_scores(results):
    """
    Extract per-pair scores from results dictionary.
    Returns dict with aggregate score and per-pair scores.
    """
    metrics_dict = {"score": 0}

    for pair_dict in results["pairs"]:
        pair_id = pair_dict["pair_id"]
        pair_total = 0
        for metric_name, metric_value in pair_dict["metrics"].items():
            # Store individual metric
            metrics_dict[f"pair_{pair_id}_{metric_name}"] = metric_value
            pair_total += metric_value
        # Store pair total
        metrics_dict[f"pair_{pair_id}_total"] = pair_total
        metrics_dict["score"] += pair_total

    return metrics_dict

def create_search_space(tuning_config):
    search_space = {}
    for name in tuning_config.keys():
        param_dict = tuning_config[name]
        if "type" not in param_dict:
            raise Exception(f"'type' not in {param_dict} keys")

        if param_dict["type"] == "uniform":
            search_space[name] = tune.uniform(param_dict["lower_bound"], param_dict["upper_bound"])
        elif param_dict["type"] == "quniform":
            search_space[name] = tune.quniform(param_dict["lower_bound"], param_dict["upper_bound"], param_dict["q"])
        elif param_dict["type"] == "loguniform":
            search_space[name] = tune.loguniform(param_dict["lower_bound"], param_dict["upper_bound"])
        elif param_dict["type"] == "qloguniform":
            search_space[name] = tune.qloguniform(param_dict["lower_bound"], param_dict["upper_bound"], param_dict["q"])
        elif param_dict["type"] == "randn":
            search_space[name] = tune.randn(param_dict["lower_bound"], param_dict["upper_bound"])
        elif param_dict["type"] == "qrandn":
            search_space[name] = tune.qrandn(param_dict["lower_bound"], param_dict["upper_bound"], param_dict["q"])
        elif param_dict["type"] == "randint":
            search_space[name] = tune.randint(param_dict["lower_bound"], param_dict["upper_bound"])
        elif param_dict["type"] == "qrandint":
            search_space[name] = tune.qrandint(param_dict["lower_bound"], param_dict["upper_bound"], param_dict["q"])
        elif param_dict["type"] == "lograndint":
            search_space[name] = tune.lograndint(param_dict["lower_bound"], param_dict["upper_bound"])
        elif param_dict["type"] == "qlograndint":
            search_space[name] = tune.qlograndint(param_dict["lower_bound"], param_dict["upper_bound"], param_dict["q"])
        elif param_dict["type"] == "choice":
            search_space[name] = tune.choice(param_dict["choices"])
        elif param_dict["type"] == "grid":
            search_space[name] = tune.grid(param_dict["grid"])
        else:
            raise Exception(f"Parameter type {param_dict['type']} not supported.")

    return search_space

def generate_config(config, template, name):
    for blank_key in config.keys():
        template["model"][blank_key] = config[blank_key]
    config_path = file_dir / "config" / f"{name}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(template, f)
    return config_path

def main(config_path: str, save_config: bool = True) -> None:
    with open(config_path, "r") as f:
        hp_config = yaml.safe_load(f)

    hyperparameters = hp_config.pop("hyperparameters")
    blank_config = hp_config.copy()
    param_dict = create_search_space(hyperparameters)

    def objective(config):
        batch_id = str(tune.get_context().get_trial_id())
        blank_config["model"]["batch_id"] = batch_id
        config_path = generate_config(config, blank_config, f"hp_config_{batch_id}")
        run_opt_main(config_path)

        config_file = file_dir / "config" / f"hp_config_{batch_id}.yaml"
        results_file = file_dir / f"results_{batch_id}.yaml"
        with open(results_file, "r") as f:
            results = yaml.safe_load(f)
        results_file.unlink(missing_ok=True)
        config_file.unlink(missing_ok=True)

        # Return per-pair scores along with aggregate
        return extract_per_pair_scores(results)

    tuner = tune.Tuner(
                        tune.with_resources(objective, {"cpu": 1, "gpu": 1}),
                        param_space=param_dict,
                        tune_config=tune.TuneConfig(
                            max_concurrent_trials=8,
                            num_samples=blank_config["model"]["n_trials"],
                            metric="score",
                            mode="max",
                        ))

    results = tuner.fit()

    # Print best overall
    result = results.get_best_result(metric="score", mode="max")
    best_config = result.config
    best_score = result.metrics["score"]
    print(f"\nBest OVERALL score: {best_score}")
    print(f"Params: {best_config}")

    # Print best per pair
    print("\n=== Best hyperparameters PER PAIR ===")
    pair_ids = blank_config["dataset"]["pair_id"]
    for pair_id in pair_ids:
        metric_name = f"pair_{pair_id}_total"
        try:
            best_for_pair = results.get_best_result(metric=metric_name, mode="max")
            pair_score = best_for_pair.metrics[metric_name]
            pair_params = best_for_pair.config
            print(f"Pair {pair_id}: score={pair_score:.2f}, params={pair_params}")
        except Exception as e:
            print(f"Pair {pair_id}: Could not find best result - {e}")

    if save_config:
        pair_ids_str = "".join(map(str, blank_config["dataset"]["pair_id"]))
        blank_config["model"].pop("batch_id", None)
        blank_config["model"].pop("n_trials", None)
        blank_config["model"].pop("train_split", None)
        config_path = generate_config(best_config, blank_config, f"config_{blank_config['dataset']['name']}_{pair_ids_str}_optimized")
        print(f"\nFinal config file saved to: {config_path}")
        with open(config_path, "w") as f:
            yaml.dump(blank_config, f)

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the hyperparameter configuration file.")
    parser.add_argument("save_config", action="store_true", help="Save the final hyperparameter configuration file.")
    args = parser.parse_args()
    main(args.config, args.save_config)
