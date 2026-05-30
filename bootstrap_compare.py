import argparse
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from config import Config, get_config
from evaluate import binary_metrics
from utils import set_seed


TASK_METRICS = {
    "incidence_auc": ("incidence", "auc"),
    "incidence_pr_auc": ("incidence", "pr_auc"),
    "incidence_brier": ("incidence", "brier"),
    "incidence_ece": ("incidence", "ece"),
    "progression_auc": ("progression", "auc"),
    "progression_pr_auc": ("progression", "pr_auc"),
    "progression_brier": ("progression", "brier"),
    "progression_ece": ("progression", "ece"),
}


def load_predictions(cfg: Config, model_name: str, seed: int, split: str) -> pd.DataFrame:
    path = cfg.paths.prediction_dir / model_name / f"seed_{seed}_{split}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file for {model_name}: {path}")
    df = pd.read_csv(path)
    rename = {
        f"{task}_logit": f"{task}_logit_{model_name}"
        for task in ["incidence", "progression"]
    }
    rename.update(
        {
            f"{task}_prob": f"{task}_prob_{model_name}"
            for task in ["incidence", "progression"]
        }
    )
    return df.rename(columns=rename)


def paired_test_dataframe(
    cfg: Config,
    model_a: str,
    model_b: str,
    seed: int,
    model_a_seed: int = None,
    model_b_seed: int = None,
) -> pd.DataFrame:
    model_a_seed = seed if model_a_seed is None else model_a_seed
    model_b_seed = seed if model_b_seed is None else model_b_seed
    a = load_predictions(cfg, model_a, model_a_seed, "test")
    b = load_predictions(cfg, model_b, model_b_seed, "test")
    keys = [
        "participant_id",
        "knee_id",
        "incidence_label",
        "incidence_mask",
        "progression_label",
        "progression_mask",
    ]
    merged = a.merge(b, on=keys, how="inner")
    if merged.empty:
        raise ValueError(f"No overlapping test knees between {model_a} and {model_b}.")
    return merged


def metric_from_predictions(df: pd.DataFrame, model_name: str, metric: str, cfg: Config) -> float:
    if metric not in TASK_METRICS:
        raise ValueError(f"Unknown metric '{metric}'. Choose one of: {', '.join(TASK_METRICS)}")
    task, metric_key = TASK_METRICS[metric]
    task_df = df[df[f"{task}_mask"] == 1]
    values = binary_metrics(
        task_df[f"{task}_label"].to_numpy(),
        task_df[f"{task}_prob_{model_name}"].to_numpy(),
        n_bins=cfg.classification.calibration_bins,
    )
    return float(values[metric_key])


def sample_participant_rows(df: pd.DataFrame, sampled_participants: np.ndarray) -> pd.DataFrame:
    pieces = []
    for draw_index, participant_id in enumerate(sampled_participants):
        piece = df[df["participant_id"] == participant_id].copy()
        piece["bootstrap_draw"] = draw_index
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def bootstrap_sample_participants(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    participants = df["participant_id"].drop_duplicates().to_numpy()
    sampled = rng.choice(participants, size=len(participants), replace=True)
    return sample_participant_rows(df, sampled)


def stratified_bootstrap_sample_participants(df: pd.DataFrame, metric: str, rng: np.random.Generator) -> pd.DataFrame:
    task, _ = TASK_METRICS[metric]
    task_df = df[df[f"{task}_mask"] == 1]
    participant_event = (
        task_df.groupby("participant_id")[f"{task}_label"]
        .max()
        .reset_index()
    )
    sampled_ids = []
    for event_value in [0, 1]:
        ids = participant_event.loc[
            participant_event[f"{task}_label"] == event_value,
            "participant_id",
        ].to_numpy()
        if len(ids) == 0:
            continue
        sampled_ids.extend(rng.choice(ids, size=len(ids), replace=True).tolist())
    sampled_ids = np.asarray(sampled_ids)
    rng.shuffle(sampled_ids)
    return sample_participant_rows(df, sampled_ids)


def participant_permuted_predictions(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = df.copy()
    participants = out["participant_id"].drop_duplicates().to_numpy()
    swap_ids = set(participants[rng.random(len(participants)) < 0.5])
    swap_mask = out["participant_id"].isin(swap_ids)
    for task in ["incidence", "progression"]:
        col_a = f"{task}_prob_{model_a}"
        col_b = f"{task}_prob_{model_b}"
        a_values = out.loc[swap_mask, col_a].copy()
        out.loc[swap_mask, col_a] = out.loc[swap_mask, col_b].to_numpy()
        out.loc[swap_mask, col_b] = a_values.to_numpy()
    return out


def paired_bootstrap_compare(
    cfg: Config,
    model_a: str,
    model_b: str,
    seed: int,
    metric: str,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 2026,
    stratified: bool = True,
    model_a_seed: int = None,
    model_b_seed: int = None,
) -> Dict[str, float]:
    set_seed(bootstrap_seed)
    rng = np.random.default_rng(bootstrap_seed)
    paired = paired_test_dataframe(cfg, model_a, model_b, seed, model_a_seed=model_a_seed, model_b_seed=model_b_seed)

    observed_a = metric_from_predictions(paired, model_a, metric, cfg)
    observed_b = metric_from_predictions(paired, model_b, metric, cfg)
    observed_diff = observed_a - observed_b

    diffs = []
    for _ in range(n_bootstrap):
        boot = (
            stratified_bootstrap_sample_participants(paired, metric, rng)
            if stratified
            else bootstrap_sample_participants(paired, rng)
        )
        try:
            value_a = metric_from_predictions(boot, model_a, metric, cfg)
            value_b = metric_from_predictions(boot, model_b, metric, cfg)
            if np.isfinite(value_a) and np.isfinite(value_b):
                diffs.append(value_a - value_b)
        except Exception:
            continue

    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0:
        raise RuntimeError("All bootstrap replicates failed; check event counts.")

    p_value = 2.0 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return {
        "seed": seed,
        "model_a_seed": int(seed if model_a_seed is None else model_a_seed),
        "model_b_seed": int(seed if model_b_seed is None else model_b_seed),
        "model_a": model_a,
        "model_b": model_b,
        "metric": metric,
        "resampling": "participant_stratified_bootstrap" if stratified else "participant_bootstrap",
        "n_bootstrap_requested": n_bootstrap,
        "n_bootstrap_used": int(len(diffs)),
        "model_a_value": float(observed_a),
        "model_b_value": float(observed_b),
        "observed_difference_a_minus_b": float(observed_diff),
        "bootstrap_mean_difference": float(np.mean(diffs)),
        "ci95_lower": float(np.percentile(diffs, 2.5)),
        "ci95_upper": float(np.percentile(diffs, 97.5)),
        "bootstrap_p_value": float(min(p_value, 1.0)),
        "paired_test_knees": int(len(paired)),
        "paired_test_participants": int(paired["participant_id"].nunique()),
    }


def paired_permutation_compare(
    cfg: Config,
    model_a: str,
    model_b: str,
    seed: int,
    metric: str,
    n_permutations: int = 1000,
    permutation_seed: int = 2027,
    model_a_seed: int = None,
    model_b_seed: int = None,
) -> Dict[str, float]:
    rng = np.random.default_rng(permutation_seed)
    paired = paired_test_dataframe(cfg, model_a, model_b, seed, model_a_seed=model_a_seed, model_b_seed=model_b_seed)

    observed_a = metric_from_predictions(paired, model_a, metric, cfg)
    observed_b = metric_from_predictions(paired, model_b, metric, cfg)
    observed_diff = observed_a - observed_b

    null_diffs = []
    for _ in range(n_permutations):
        permuted = participant_permuted_predictions(paired, model_a, model_b, rng)
        try:
            value_a = metric_from_predictions(permuted, model_a, metric, cfg)
            value_b = metric_from_predictions(permuted, model_b, metric, cfg)
            if np.isfinite(value_a) and np.isfinite(value_b):
                null_diffs.append(value_a - value_b)
        except Exception:
            continue

    null_diffs = np.asarray(null_diffs, dtype=float)
    if len(null_diffs) == 0:
        raise RuntimeError("All permutation replicates failed; check event counts.")

    p_value = (np.sum(np.abs(null_diffs) >= abs(observed_diff)) + 1.0) / (len(null_diffs) + 1.0)
    return {
        "seed": seed,
        "model_a_seed": int(seed if model_a_seed is None else model_a_seed),
        "model_b_seed": int(seed if model_b_seed is None else model_b_seed),
        "model_a": model_a,
        "model_b": model_b,
        "metric": metric,
        "resampling": "participant_paired_permutation",
        "n_permutations_requested": n_permutations,
        "n_permutations_used": int(len(null_diffs)),
        "model_a_value": float(observed_a),
        "model_b_value": float(observed_b),
        "observed_difference_a_minus_b": float(observed_diff),
        "permutation_p_value": float(min(p_value, 1.0)),
        "null_diff_mean": float(np.mean(null_diffs)),
        "null_diff_ci95_lower": float(np.percentile(null_diffs, 2.5)),
        "null_diff_ci95_upper": float(np.percentile(null_diffs, 97.5)),
        "paired_test_knees": int(len(paired)),
        "paired_test_participants": int(paired["participant_id"].nunique()),
    }


DEFAULT_COMPARISONS = [
    ("model2_current_pa_lat", "model1_current_pa", "value of current lateral view"),
    ("model3_current_pa_history_pa", "model1_current_pa", "value of historical PA X-ray"),
    ("model4_full_multiview_history", "model1_current_pa", "value of full multi-view + historical imaging"),
    ("model4_full_multiview_history", "model3_current_pa_history_pa", "additional value of lateral view when history is included"),
]


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", choices=list(get_config().model_inputs.keys()))
    parser.add_argument("--model-b", choices=list(get_config().model_inputs.keys()))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-a-seed", type=int, default=None)
    parser.add_argument("--model-b-seed", type=int, default=None)
    parser.add_argument("--metric", default="incidence_auc", choices=list(TASK_METRICS.keys()))
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--include-permutation", action="store_true")
    parser.add_argument("--unstratified-bootstrap", action="store_true")
    parser.add_argument("--all-default-comparisons", action="store_true")
    args = parser.parse_args()

    cfg = get_config()
    comparisons = DEFAULT_COMPARISONS if args.all_default_comparisons else [(args.model_a, args.model_b, "custom")]
    rows = []
    for model_a, model_b, label in comparisons:
        try:
            row = paired_bootstrap_compare(
                cfg,
                model_a,
                model_b,
                args.seed,
                args.metric,
            args.n_bootstrap,
            stratified=not args.unstratified_bootstrap,
            model_a_seed=args.model_a_seed,
            model_b_seed=args.model_b_seed,
        )
            row["comparison_label"] = label
            rows.append(row)
            if args.include_permutation:
                perm_row = paired_permutation_compare(
                    cfg,
                    model_a,
                    model_b,
                    args.seed,
                    args.metric,
                    args.n_permutations,
                    model_a_seed=args.model_a_seed,
                    model_b_seed=args.model_b_seed,
                )
                perm_row["comparison_label"] = label
                rows.append(perm_row)
        except FileNotFoundError as exc:
            print(f"Skipping {model_a} vs {model_b}: {exc}")

    out_dir = cfg.paths.metrics_dir / "bootstrap"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / f"seed_{args.seed}_{args.metric}_bootstrap_comparisons.csv", index=False)


if __name__ == "__main__":
    main()
