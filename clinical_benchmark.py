import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import Config, configure_run_paths, get_config
from utils import require_columns, save_metrics_csv


TASKS = ("incidence", "progression")

FEATURE_SETS: Dict[str, List[str]] = {
    "m30_kl": ["m30_kl"],
    "kl": ["bl_kl", "m30_kl"],
    "pfoa": ["bl_pfoa", "m30_pfoa"],
    "m30_kl_pfoa": ["m30_kl", "m30_pfoa"],
    "kl_pfoa": ["bl_kl", "m30_kl", "bl_pfoa", "m30_pfoa"],
}


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    if len(y_true) == 0:
        return np.nan
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        else:
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if not mask.any():
            continue
        ece += mask.mean() * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return float(ece)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    has_both_classes = len(np.unique(y_true)) == 2
    return {
        "auc": float(roc_auc_score(y_true, y_prob)) if has_both_classes else np.nan,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if has_both_classes else np.nan,
        "brier": float(brier_score_loss(y_true, y_prob)) if len(y_true) else np.nan,
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
        "n": int(len(y_true)),
        "events": int(y_true.sum()),
    }


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, path: Path, n_bins: int = 10) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    observed = []
    predicted = []
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        else:
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.any():
            observed.append(float(y_true[mask].mean()))
            predicted.append(float(y_prob[mask].mean()))

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    if observed:
        plt.plot(predicted, observed, marker="o", linewidth=2)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed event rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def build_model():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def task_frame(df: pd.DataFrame, task: str, features: List[str], cfg: Config) -> Tuple[pd.DataFrame, pd.Series]:
    label_col = getattr(cfg.columns, f"{task}_label")
    mask_col = getattr(cfg.columns, f"{task}_mask")
    out = df[df[mask_col] == 1].copy()
    y = out[label_col].astype(int)
    x = out[features].apply(pd.to_numeric, errors="coerce")
    return x, y


def evaluate_feature_set(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    task: str,
    feature_set: str,
    features: List[str],
    split: str,
    cfg: Config,
) -> Tuple[Dict[str, float], pd.DataFrame, object]:
    x_train, y_train = task_frame(train_df, task, features, cfg)
    x_eval, y_eval = task_frame(eval_df, task, features, cfg)
    if y_train.nunique() < 2:
        raise ValueError(f"Training split has fewer than 2 classes for task={task}, feature_set={feature_set}.")

    model = build_model()
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_eval)[:, 1]
    metrics = binary_metrics(y_eval.to_numpy(), prob, n_bins=cfg.classification.calibration_bins)
    metrics.update(
        {
            "task": task,
            "feature_set": feature_set,
            "features": ";".join(features),
            "split": split,
        }
    )
    pred = eval_df.loc[x_eval.index, ["participant_id", "knee_id", "side"]].copy()
    pred["task"] = task
    pred["feature_set"] = feature_set
    pred["split"] = split
    pred["label"] = y_eval.to_numpy()
    pred["prob"] = prob
    return metrics, pred, model


def run_clinical_benchmarks(
    cfg: Config,
    split_model: str,
    feature_sets: Dict[str, List[str]] = None,
) -> pd.DataFrame:
    feature_sets = feature_sets or FEATURE_SETS
    split_dir = cfg.paths.split_dir / split_model
    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "val.csv")
    test_df = pd.read_csv(split_dir / "test.csv")

    required = ["participant_id", "knee_id", "side"]
    for features in feature_sets.values():
        required.extend(features)
    for task in TASKS:
        required.extend([getattr(cfg.columns, f"{task}_label"), getattr(cfg.columns, f"{task}_mask")])
    require_columns(train_df, sorted(set(required)))
    require_columns(val_df, sorted(set(required)))
    require_columns(test_df, sorted(set(required)))

    out_dir = cfg.paths.metrics_dir / "clinical_benchmark"
    pred_dir = cfg.paths.prediction_dir / "clinical_benchmark"
    plot_dir = out_dir / "calibration_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    pred_rows = []
    for feature_set, features in feature_sets.items():
        for task in TASKS:
            for split, eval_df in [("val", val_df), ("test", test_df)]:
                metrics, pred, _ = evaluate_feature_set(
                    train_df,
                    eval_df,
                    task,
                    feature_set,
                    features,
                    split,
                    cfg,
                )
                metric_rows.append(metrics)
                pred_rows.append(pred)
                plot_calibration(
                    pred["label"].to_numpy(),
                    pred["prob"].to_numpy(),
                    plot_dir / f"{split}_{task}_{feature_set}_calibration.png",
                    n_bins=cfg.classification.calibration_bins,
                )

    metrics_df = pd.DataFrame(metric_rows)
    preds_df = pd.concat(pred_rows, ignore_index=True)
    metrics_df.to_csv(out_dir / "clinical_benchmark_metrics.csv", index=False)
    preds_df.to_csv(pred_dir / "clinical_benchmark_predictions.csv", index=False)

    test_wide = metrics_df[metrics_df["split"] == "test"].copy()
    save_metrics_csv(
        {
            "split_model": split_model,
            "n_feature_sets": int(len(feature_sets)),
            "n_rows": int(len(metrics_df)),
        },
        out_dir / "clinical_benchmark_summary.csv",
    )
    return test_wide


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--split-model", default=None, choices=list(get_config().model_inputs.keys()))
    args = parser.parse_args()

    cfg = get_config()
    if args.run_id is not None:
        configure_run_paths(cfg.paths, args.run_id)
    split_model = args.split_model or cfg.split.reference_model
    test_metrics = run_clinical_benchmarks(cfg, split_model)
    print(test_metrics.sort_values(["task", "feature_set"]).to_string(index=False))


if __name__ == "__main__":
    main()
