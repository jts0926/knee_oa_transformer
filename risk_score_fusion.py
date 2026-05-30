import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import Config, configure_run_paths, get_config


TASKS = ("incidence", "progression")

SINGLE_VIEW_MODELS = {
    "m30_pa": "patch_mil_m30_pa",
    "m30_lat": "patch_mil_m30_lat",
    "bl_pa": "patch_mil_bl_pa",
    "bl_lat": "patch_mil_bl_lat",
}

CLINICAL_FEATURES = ["bl_kl", "m30_kl", "bl_pfoa", "m30_pfoa"]

FUSION_FEATURE_SETS = {
    "score_m30_pa": ["score_m30_pa"],
    "score_m30_lat": ["score_m30_lat"],
    "score_bl_pa": ["score_bl_pa"],
    "score_bl_lat": ["score_bl_lat"],
    "scores_current_pa_lat": ["score_m30_pa", "score_m30_lat"],
    "scores_pa_history": ["score_m30_pa", "score_bl_pa"],
    "scores_all_views": ["score_m30_pa", "score_m30_lat", "score_bl_pa", "score_bl_lat"],
    "clinical_kl_pfoa": CLINICAL_FEATURES,
    "score_m30_pa_clinical": ["score_m30_pa", *CLINICAL_FEATURES],
    "scores_pa_history_clinical": ["score_m30_pa", "score_bl_pa", *CLINICAL_FEATURES],
    "scores_all_views_clinical": ["score_m30_pa", "score_m30_lat", "score_bl_pa", "score_bl_lat", *CLINICAL_FEATURES],
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
        if mask.any():
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


def build_model(c_value: float):
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
                    C=c_value,
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def split_base_dataframe(cfg: Config, split_model: str, split: str) -> pd.DataFrame:
    path = cfg.paths.split_dir / split_model / f"{split}.csv"
    df = pd.read_csv(path)
    keep = [
        "participant_id",
        "knee_id",
        "side",
        "incidence_label",
        "incidence_mask",
        "progression_label",
        "progression_mask",
        *CLINICAL_FEATURES,
    ]
    missing = [column for column in keep if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required split columns in {path}: {missing}")
    return df[keep].copy()


def load_prediction_scores(cfg: Config, seed: int, split: str, model_alias: str, model_name: str) -> pd.DataFrame:
    path = cfg.paths.prediction_dir / model_name / f"seed_{seed}_{split}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing prediction file: {path}. Run evaluate.py for {model_name} before risk_score_fusion.py."
        )
    pred = pd.read_csv(path)
    keep = ["participant_id", "knee_id"]
    for task in TASKS:
        keep.append(f"{task}_prob")
    pred = pred[keep].copy()
    return pred.rename(columns={f"{task}_prob": f"score_{model_alias}_{task}" for task in TASKS})


def assemble_split_scores(
    cfg: Config,
    seed: int,
    split_model: str,
    split: str,
    model_seeds: Dict[str, int] = None,
) -> pd.DataFrame:
    model_seeds = model_seeds or {}
    df = split_base_dataframe(cfg, split_model, split)
    for alias, model_name in SINGLE_VIEW_MODELS.items():
        score_seed = int(model_seeds.get(alias, model_seeds.get(model_name, seed)))
        scores = load_prediction_scores(cfg, score_seed, split, alias, model_name)
        df = df.merge(scores, on=["participant_id", "knee_id"], how="inner")
    return df


def task_columns(task: str, feature_names: List[str]) -> List[str]:
    out = []
    for name in feature_names:
        if name.startswith("score_"):
            out.append(f"{name}_{task}")
        else:
            out.append(name)
    return out


def task_xy(df: pd.DataFrame, task: str, feature_names: List[str], cfg: Config) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    label_col = getattr(cfg.columns, f"{task}_label")
    mask_col = getattr(cfg.columns, f"{task}_mask")
    task_df = df[df[mask_col] == 1].copy()
    columns = task_columns(task, feature_names)
    x = task_df[columns].apply(pd.to_numeric, errors="coerce")
    y = task_df[label_col].astype(int)
    return x, y, task_df


def choose_c_by_validation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    task: str,
    feature_names: List[str],
    cfg: Config,
    c_grid: List[float],
) -> float:
    x_train, y_train, _ = task_xy(train_df, task, feature_names, cfg)
    x_val, y_val, _ = task_xy(val_df, task, feature_names, cfg)
    best_c = c_grid[0]
    best_score = -np.inf
    for c_value in c_grid:
        model = build_model(c_value)
        model.fit(x_train, y_train)
        prob = model.predict_proba(x_val)[:, 1]
        metrics = binary_metrics(y_val.to_numpy(), prob, n_bins=cfg.classification.calibration_bins)
        score = metrics["pr_auc"]
        if not np.isfinite(score):
            score = metrics["auc"]
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_c = c_value
    return best_c


def fit_predict_feature_set(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    task: str,
    feature_set: str,
    feature_names: List[str],
    cfg: Config,
    c_grid: List[float],
) -> Tuple[List[Dict[str, float]], pd.DataFrame, Dict[str, object]]:
    selected_c = choose_c_by_validation(train_df, val_df, task, feature_names, cfg, c_grid)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    x_train_val, y_train_val, _ = task_xy(train_val_df, task, feature_names, cfg)
    model = build_model(selected_c)
    model.fit(x_train_val, y_train_val)

    metric_rows = []
    pred_rows = []
    for split, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        x, y, task_df = task_xy(split_df, task, feature_names, cfg)
        prob = model.predict_proba(x)[:, 1]
        metrics = binary_metrics(y.to_numpy(), prob, n_bins=cfg.classification.calibration_bins)
        metrics.update(
            {
                "task": task,
                "feature_set": feature_set,
                "features": ";".join(task_columns(task, feature_names)),
                "split": split,
                "selected_c": selected_c,
                "stack_training": "train_val_after_val_tuning",
            }
        )
        metric_rows.append(metrics)

        pred = task_df[["participant_id", "knee_id", "side"]].copy()
        pred["task"] = task
        pred["feature_set"] = feature_set
        pred["split"] = split
        pred["label"] = y.to_numpy()
        pred["prob"] = prob
        pred_rows.append(pred)

    logistic = model.named_steps["logistic"]
    coef_row = {
        "task": task,
        "feature_set": feature_set,
        "selected_c": selected_c,
        "intercept": float(logistic.intercept_[0]),
    }
    for column, coef in zip(task_columns(task, feature_names), logistic.coef_[0]):
        coef_row[f"coef_{column}"] = float(coef)
    return metric_rows, pd.concat(pred_rows, ignore_index=True), coef_row


def run_risk_score_fusion(
    cfg: Config,
    seed: int,
    split_model: str,
    feature_sets: Dict[str, List[str]] = None,
    model_seeds: Dict[str, int] = None,
) -> pd.DataFrame:
    feature_sets = feature_sets or FUSION_FEATURE_SETS
    model_seeds = model_seeds or {}
    train_df = assemble_split_scores(cfg, seed, split_model, "train", model_seeds=model_seeds)
    val_df = assemble_split_scores(cfg, seed, split_model, "val", model_seeds=model_seeds)
    test_df = assemble_split_scores(cfg, seed, split_model, "test", model_seeds=model_seeds)

    out_dir = cfg.paths.metrics_dir / "risk_score_fusion"
    pred_dir = cfg.paths.prediction_dir / "risk_score_fusion"
    plot_dir = out_dir / "calibration_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    c_grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    metric_rows = []
    pred_rows = []
    coef_rows = []
    for feature_set, features in feature_sets.items():
        for task in TASKS:
            rows, preds, coef = fit_predict_feature_set(
                train_df,
                val_df,
                test_df,
                task,
                feature_set,
                features,
                cfg,
                c_grid,
            )
            metric_rows.extend(rows)
            pred_rows.append(preds)
            coef["model_seeds"] = format_model_seeds(seed, model_seeds)
            coef_rows.append(coef)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df["model_seeds"] = format_model_seeds(seed, model_seeds)
    preds_df = pd.concat(pred_rows, ignore_index=True)
    preds_df["model_seeds"] = format_model_seeds(seed, model_seeds)
    coefs_df = pd.DataFrame(coef_rows)
    suffix = fusion_suffix(seed, model_seeds)
    metrics_df.to_csv(out_dir / f"{suffix}_risk_score_fusion_metrics.csv", index=False)
    preds_df.to_csv(pred_dir / f"{suffix}_risk_score_fusion_predictions.csv", index=False)
    coefs_df.to_csv(out_dir / f"{suffix}_risk_score_fusion_coefficients.csv", index=False)

    for _, row in metrics_df[metrics_df["split"] == "test"].iterrows():
        pred = preds_df[
            (preds_df["split"] == "test")
            & (preds_df["task"] == row["task"])
            & (preds_df["feature_set"] == row["feature_set"])
        ]
        plot_calibration(
            pred["label"].to_numpy(),
            pred["prob"].to_numpy(),
            plot_dir / f"{suffix}_test_{row['task']}_{row['feature_set']}_calibration.png",
            n_bins=cfg.classification.calibration_bins,
        )

    return metrics_df[metrics_df["split"] == "test"].copy()


def parse_model_seeds(value: str) -> Dict[str, int]:
    if value is None or str(value).strip() == "":
        return {}
    out = {}
    for item in str(value).split(","):
        if not item.strip():
            continue
        key, seed = item.split("=", 1)
        out[key.strip()] = int(seed)
    return out


def format_model_seeds(default_seed: int, model_seeds: Dict[str, int]) -> str:
    if not model_seeds:
        return f"default={default_seed}"
    pieces = [f"default={default_seed}"]
    pieces.extend(f"{key}={value}" for key, value in sorted(model_seeds.items()))
    return ";".join(pieces)


def fusion_suffix(default_seed: int, model_seeds: Dict[str, int]) -> str:
    if not model_seeds:
        return f"seed_{default_seed}"
    pieces = [f"default{default_seed}"]
    pieces.extend(f"{key}{value}" for key, value in sorted(model_seeds.items()))
    return "seedmix_" + "_".join(pieces)


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split-model", default=None, choices=list(get_config().model_inputs.keys()))
    parser.add_argument(
        "--model-seeds",
        default=None,
        help="Optional comma-separated alias/model seed map, e.g. m30_pa=11,m30_lat=37,bl_pa=23,bl_lat=51.",
    )
    args = parser.parse_args()

    cfg = get_config()
    if args.run_id is not None:
        configure_run_paths(cfg.paths, args.run_id)
    split_model = args.split_model or cfg.split.reference_model
    test_metrics = run_risk_score_fusion(cfg, args.seed, split_model, model_seeds=parse_model_seeds(args.model_seeds))
    print(test_metrics.sort_values(["task", "feature_set"]).to_string(index=False))


if __name__ == "__main__":
    main()
