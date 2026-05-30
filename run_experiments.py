import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from config import get_config


DEFAULT_RISK_SCORE_MODELS = [
    "patch_mil_m30_pa",
    "patch_mil_m30_lat",
    "patch_mil_bl_pa",
    "patch_mil_bl_lat",
]


def run_command(args, run_id: str):
    env = os.environ.copy()
    env["KNEE_RUN_ID"] = run_id
    subprocess.run([sys.executable, *args], check=True, cwd=Path(__file__).parent, env=env)


def summarize_seed_metrics(model_name: str) -> None:
    cfg = get_config()
    model_dir = cfg.paths.metrics_dir / model_name
    seed_paths = sorted(model_dir.glob("seed_*_test_metrics.csv"))
    if not seed_paths:
        return
    df = pd.concat([pd.read_csv(path) for path in seed_paths], ignore_index=True)
    df.to_csv(model_dir / "all_seed_test_metrics.csv", index=False)
    numeric = df.select_dtypes(include="number")
    summary = pd.DataFrame({"mean": numeric.mean(), "sd": numeric.std(ddof=1)}).reset_index()
    summary = summary.rename(columns={"index": "metric"})
    summary.to_csv(model_dir / "mean_sd_test_metrics.csv", index=False)


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--make-splits", action="store_true")
    parser.add_argument("--force-recreate-participant-splits", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--clinical-benchmark", action="store_true")
    parser.add_argument("--risk-score-fusion", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--fusion-seed", type=int, default=None)
    parser.add_argument("--fusion-model-seeds", default=None)
    parser.add_argument("--skip-file-check", action="store_true")
    args = parser.parse_args()

    cfg = get_config()
    if args.run_id is not None:
        cfg.paths.run_id = args.run_id
        cfg.paths.__post_init__()
    os.environ["KNEE_RUN_ID"] = cfg.paths.run_id
    print(f"Run ID: {cfg.paths.run_id}")
    print(f"Output directory: {cfg.paths.output_dir}")
    if args.models is not None:
        models = args.models
    elif args.all_models:
        models = list(cfg.model_inputs.keys())
    else:
        models = DEFAULT_RISK_SCORE_MODELS
    if not models:
        raise ValueError("No models selected. Pass --models or omit it to use the default risk-score models.")
    seeds = args.seeds if args.seeds is not None else [cfg.training.seeds[0]]

    split_model = models[0]
    for index, model_name in enumerate(models):
        if args.make_splits:
            split_args = ["split_data.py", "--model", model_name]
            if args.force_recreate_participant_splits and index == 0:
                split_args.append("--force-recreate-participant-splits")
            if args.skip_file_check:
                split_args.append("--skip-file-check")
            run_command(split_args, cfg.paths.run_id)
    if args.clinical_benchmark:
        run_command(["clinical_benchmark.py", "--run-id", cfg.paths.run_id, "--split-model", split_model], cfg.paths.run_id)
    for model_name in models:
        if args.train:
            for seed in seeds:
                run_command(["train.py", "--model", model_name, "--seed", str(seed)], cfg.paths.run_id)
        if args.evaluate:
            for seed in seeds:
                run_command(["evaluate.py", "--model", model_name, "--seed", str(seed)], cfg.paths.run_id)
            summarize_seed_metrics(model_name)
    if args.risk_score_fusion:
        seed = args.fusion_seed if args.fusion_seed is not None else seeds[0]
        fusion_args = ["risk_score_fusion.py", "--run-id", cfg.paths.run_id, "--seed", str(seed), "--split-model", split_model]
        if args.fusion_model_seeds is not None:
            fusion_args.extend(["--model-seeds", args.fusion_model_seeds])
        run_command(fusion_args, cfg.paths.run_id)


if __name__ == "__main__":
    main()
