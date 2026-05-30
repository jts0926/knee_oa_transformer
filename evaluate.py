import argparse
import gc
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config, ensure_output_dirs, get_config
from dataset import KneeXrayClassificationDataset, build_transforms, knee_collate_fn
from losses import TASKS, masked_bce_with_logits_loss
from model import KneeOAClassificationTransformer
from utils import move_batch_to_device, resolve_device, save_metrics_csv, set_seed


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def shifted_images(images: Dict[str, torch.Tensor], shift_x: float, shift_y: float) -> Dict[str, torch.Tensor]:
    if shift_x == 0.0 and shift_y == 0.0:
        return images
    shifted = {}
    for key, value in images.items():
        theta = torch.zeros((value.shape[0], 2, 3), dtype=value.dtype, device=value.device)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, 0, 2] = shift_x
        theta[:, 1, 2] = shift_y
        grid = F.affine_grid(theta, value.shape, align_corners=False)
        shifted[key] = F.grid_sample(value, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return shifted


def model_logits_with_tta(
    model: KneeOAClassificationTransformer,
    images: Dict[str, torch.Tensor],
    cfg: Config,
) -> Dict[str, torch.Tensor]:
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    if not cfg.training.use_eval_tta or getattr(raw_model, "architecture", "") == "patch_mil":
        return model(images)
    offset = float(cfg.training.eval_tta_translate_fraction) * 2.0
    shifts = [
        (0.0, 0.0),
        (offset, 0.0),
        (-offset, 0.0),
        (0.0, offset),
        (0.0, -offset),
    ]
    outputs = []
    for shift_x, shift_y in shifts:
        outputs.append(model(shifted_images(images, shift_x, shift_y)))
    return {
        task: torch.stack([output[task] for output in outputs], dim=0).mean(dim=0)
        for task in TASKS
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


@torch.no_grad()
def predict_on_loader(
    model: KneeOAClassificationTransformer,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows: List[pd.DataFrame] = []
    for batch in loader:
        device_batch = move_batch_to_device(batch, device)
        if cfg.training.channels_last and device.type == "cuda":
            device_batch["images"] = {
                key: value.contiguous(memory_format=torch.channels_last)
                for key, value in device_batch["images"].items()
            }
        with torch.cuda.amp.autocast(enabled=bool(cfg.training.use_amp and device.type == "cuda")):
            logits = model_logits_with_tta(model, device_batch["images"], cfg)
        row = {
            "participant_id": batch["participant_id"],
            "knee_id": batch["knee_id"],
        }
        for task in TASKS:
            task_logits = logits[task].detach().cpu().numpy()
            row[f"{task}_label"] = batch[f"{task}_label"].cpu().numpy().astype(int)
            row[f"{task}_mask"] = batch[f"{task}_mask"].cpu().numpy().astype(int)
            row[f"{task}_logit"] = task_logits
            row[f"{task}_prob"] = sigmoid(task_logits)
        rows.append(pd.DataFrame(row))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


@torch.no_grad()
def evaluate_model_on_loader(
    model: KneeOAClassificationTransformer,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    model.eval()
    losses = []
    task_loss_rows = []
    preds = []
    for batch in loader:
        device_batch = move_batch_to_device(batch, device)
        if cfg.training.channels_last and device.type == "cuda":
            device_batch["images"] = {
                key: value.contiguous(memory_format=torch.channels_last)
                for key, value in device_batch["images"].items()
            }
        with torch.cuda.amp.autocast(enabled=bool(cfg.training.use_amp and device.type == "cuda")):
            logits = model_logits_with_tta(model, device_batch["images"], cfg)
            loss, task_losses = masked_bce_with_logits_loss(logits, device_batch)
        losses.append(float(loss.detach().cpu()))
        task_loss_rows.append(task_losses)
        row = {
            "participant_id": batch["participant_id"],
            "knee_id": batch["knee_id"],
        }
        for task in TASKS:
            task_logits = logits[task].detach().cpu().numpy()
            row[f"{task}_label"] = batch[f"{task}_label"].cpu().numpy().astype(int)
            row[f"{task}_mask"] = batch[f"{task}_mask"].cpu().numpy().astype(int)
            row[f"{task}_logit"] = task_logits
            row[f"{task}_prob"] = sigmoid(task_logits)
        preds.append(pd.DataFrame(row))

    pred_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    metrics: Dict[str, float] = {"loss": float(np.mean(losses)) if losses else np.nan}
    if task_loss_rows:
        task_loss_df = pd.DataFrame(task_loss_rows)
        for column in task_loss_df.columns:
            metrics[column] = float(task_loss_df[column].mean(skipna=True))

    for task in TASKS:
        if pred_df.empty:
            task_metrics = {"auc": np.nan, "pr_auc": np.nan, "brier": np.nan, "ece": np.nan, "n": 0, "events": 0}
        else:
            task_df = pred_df[pred_df[f"{task}_mask"] == 1]
            task_metrics = binary_metrics(
                task_df[f"{task}_label"].to_numpy(),
                task_df[f"{task}_prob"].to_numpy(),
                n_bins=cfg.classification.calibration_bins,
            )
        metrics.update({f"{task}_{key}": value for key, value in task_metrics.items()})

    pr_values = [metrics[f"{task}_pr_auc"] for task in TASKS if np.isfinite(metrics[f"{task}_pr_auc"])]
    auc_values = [metrics[f"{task}_auc"] for task in TASKS if np.isfinite(metrics[f"{task}_auc"])]
    metrics["mean_pr_auc"] = float(np.mean(pr_values)) if pr_values else np.nan
    metrics["mean_auc"] = float(np.mean(auc_values)) if auc_values else np.nan
    return metrics, pred_df


def build_eval_loader(df: pd.DataFrame, cfg: Config, model_name: str) -> DataLoader:
    dataset = KneeXrayClassificationDataset(
        df,
        cfg,
        model_name,
        transform=build_transforms(cfg, train=False, model_name=model_name),
        check_image_files=True,
    )
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": cfg.training.patch_batch_size
        if cfg.model_inputs[model_name].get("architecture") == "patch_mil"
        else cfg.training.batch_size,
        "shuffle": False,
        "num_workers": cfg.training.num_workers,
        "collate_fn": knee_collate_fn,
        "pin_memory": cfg.training.pin_memory,
    }
    if cfg.training.num_workers > 0:
        loader_kwargs["persistent_workers"] = cfg.training.persistent_workers
        loader_kwargs["prefetch_factor"] = cfg.training.prefetch_factor
    return DataLoader(**loader_kwargs)


def load_model_from_checkpoint(cfg: Config, model_name: str, checkpoint_path: Path, device: torch.device):
    model = KneeOAClassificationTransformer(cfg, model_name).to(device)
    if cfg.training.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def save_calibration_plots(pred_df: pd.DataFrame, cfg: Config, model_name: str, seed: int, split: str = "test") -> None:
    plot_dir = cfg.paths.metrics_dir / model_name / "calibration_plots"
    for task in TASKS:
        task_df = pred_df[pred_df[f"{task}_mask"] == 1]
        if task_df.empty:
            continue
        plot_calibration(
            task_df[f"{task}_label"].to_numpy(),
            task_df[f"{task}_prob"].to_numpy(),
            plot_dir / f"seed_{seed}_{split}_{task}_calibration.png",
            n_bins=cfg.classification.calibration_bins,
        )


def evaluate_split(
    cfg: Config,
    model_name: str,
    seed: int,
    split: str,
    model: KneeOAClassificationTransformer,
    device: torch.device,
) -> Dict[str, float]:
    split_df = pd.read_csv(cfg.paths.split_dir / model_name / f"{split}.csv")
    loader = build_eval_loader(split_df, cfg, model_name)
    try:
        metrics, pred = evaluate_model_on_loader(model, loader, cfg, device)
    finally:
        del loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pred_dir = cfg.paths.prediction_dir / model_name
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(pred_dir / f"seed_{seed}_{split}_predictions.csv", index=False)
    if split == "test":
        save_calibration_plots(pred, cfg, model_name, seed, split=split)
    return {"model": model_name, "seed": seed, "split": split, **metrics}


def evaluate_checkpoint(cfg: Config, model_name: str, seed: int) -> Dict[str, float]:
    ensure_output_dirs(cfg)
    set_seed(seed)
    device = resolve_device(cfg.training.device)
    torch.backends.cudnn.benchmark = bool(cfg.training.cudnn_benchmark and device.type == "cuda")
    ckpt = cfg.paths.checkpoint_dir / model_name / f"seed_{seed}_best.pt"
    model = load_model_from_checkpoint(cfg, model_name, ckpt, device)

    rows = [evaluate_split(cfg, model_name, seed, split, model, device) for split in ["train", "val", "test"]]
    metrics = [row for row in rows if row["split"] == "test"][0]

    metric_path = cfg.paths.metrics_dir / model_name / f"seed_{seed}_test_metrics.csv"
    save_metrics_csv(metrics, metric_path)
    pd.DataFrame(rows).to_csv(cfg.paths.metrics_dir / model_name / f"seed_{seed}_all_split_metrics.csv", index=False)
    return metrics


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(get_config().model_inputs.keys()), required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = get_config()
    seeds = [args.seed] if args.seed is not None else cfg.training.seeds
    rows = [evaluate_checkpoint(cfg, args.model, seed) for seed in seeds]
    out_dir = cfg.paths.metrics_dir / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "all_seed_test_metrics.csv", index=False)


if __name__ == "__main__":
    main()
