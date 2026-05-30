import argparse
import gc
import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import Config, ensure_output_dirs, get_config
from dataset import KneeXrayClassificationDataset, build_transforms, knee_collate_fn
from evaluate import evaluate_model_on_loader
from losses import masked_bce_with_logits_loss
from model import KneeOAClassificationTransformer
from utils import EarlyStopping, move_batch_to_device, resolve_device, set_seed


def build_loader(df: pd.DataFrame, cfg: Config, model_name: str, train: bool) -> DataLoader:
    dataset = KneeXrayClassificationDataset(
        df,
        cfg,
        model_name,
        transform=build_transforms(cfg, train=train, model_name=model_name),
        check_image_files=True,
    )
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": cfg.training.patch_batch_size
        if cfg.model_inputs[model_name].get("architecture") == "patch_mil"
        else cfg.training.batch_size,
        "shuffle": train,
        "num_workers": cfg.training.num_workers,
        "collate_fn": knee_collate_fn,
        "pin_memory": cfg.training.pin_memory,
    }
    if cfg.training.num_workers > 0:
        loader_kwargs["persistent_workers"] = cfg.training.persistent_workers
        loader_kwargs["prefetch_factor"] = cfg.training.prefetch_factor
    return DataLoader(
        **loader_kwargs,
    )


def load_split_csvs(cfg: Config, model_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_dir = cfg.paths.split_dir / model_name
    return (
        pd.read_csv(split_dir / "train.csv"),
        pd.read_csv(split_dir / "val.csv"),
        pd.read_csv(split_dir / "test.csv"),
    )


def format_metric(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value:.4f}"


def monitored_metric(metrics: dict, metric_name: str) -> float:
    value = metrics.get(metric_name, float("nan"))
    if pd.isna(value) and metric_name == "mean_pr_auc":
        value = metrics.get("mean_auc", float("nan"))
    if pd.isna(value) and metric_name in {"mean_pr_auc", "mean_auc"}:
        value = -metrics.get("loss", float("inf"))
    return value


def set_backbone_trainable(model: KneeOAClassificationTransformer, trainable: bool) -> None:
    model = unwrap_model(model)
    for parameter in model.image_encoder.features.parameters():
        parameter.requires_grad = trainable


def unwrap_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def build_optimizer(cfg: Config, model: KneeOAClassificationTransformer) -> torch.optim.Optimizer:
    model = unwrap_model(model)
    backbone_params = list(model.image_encoder.features.parameters())
    gate_params = list(model.token_gate_logits.parameters())
    other_params = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("image_encoder.features.")
        and not name.startswith("token_gate_logits.")
    ]
    return torch.optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": cfg.training.lr * cfg.training.backbone_lr_multiplier,
            },
            {
                "params": gate_params,
                "lr": cfg.training.lr * cfg.training.gate_lr_multiplier,
                "weight_decay": 0.0,
            },
            {
                "params": other_params,
                "lr": cfg.training.lr,
            },
        ],
        weight_decay=cfg.training.weight_decay,
    )


def compute_pos_weights(train_df: pd.DataFrame, cfg: Config, device: torch.device) -> dict:
    if not cfg.classification.use_pos_weight:
        return {}
    weights = {}
    for task in ["incidence", "progression"]:
        label_col = getattr(cfg.columns, f"{task}_label")
        mask_col = getattr(cfg.columns, f"{task}_mask")
        task_df = train_df[train_df[mask_col] == 1]
        positives = float(task_df[label_col].sum())
        negatives = float(len(task_df) - positives)
        if positives <= 0 or negatives <= 0:
            continue
        value = negatives / positives
        value = min(max(value, cfg.classification.min_pos_weight), cfg.classification.max_pos_weight)
        weights[task] = torch.tensor(value, dtype=torch.float32, device=device)
    return weights


def train_one_seed(
    cfg: Config,
    model_name: str,
    seed: int,
    show_progress: Optional[bool] = None,
) -> Path:
    best_path: Optional[Path] = None
    set_seed(seed)
    ensure_output_dirs(cfg)
    device = resolve_device(cfg.training.device)
    torch.backends.cudnn.benchmark = bool(cfg.training.cudnn_benchmark and device.type == "cuda")
    train_df, val_df, _ = load_split_csvs(cfg, model_name)
    pos_weights = compute_pos_weights(train_df, cfg, device)
    train_loader = build_loader(train_df, cfg, model_name, train=True)
    val_loader = build_loader(val_df, cfg, model_name, train=False)

    model = KneeOAClassificationTransformer(cfg, model_name).to(device)
    if cfg.training.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    optimizer = build_optimizer(cfg, model)
    if cfg.training.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min" if cfg.training.early_stopping_metric == "loss" else "max",
        factor=cfg.training.lr_scheduler_factor,
        patience=cfg.training.lr_scheduler_patience,
    )
    metric_name = cfg.training.early_stopping_metric
    stopper = EarlyStopping(cfg.training.patience, mode="min" if metric_name == "loss" else "max")

    ckpt_dir = cfg.paths.checkpoint_dir / model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"seed_{seed}_best.pt"
    history = []
    show_progress = cfg.training.show_progress if show_progress is None else show_progress
    use_amp = bool(cfg.training.use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if show_progress:
        print(
            f"\nTraining {model_name} | seed={seed} | device={device} | "
            f"train knees={len(train_loader.dataset)} | val knees={len(val_loader.dataset)} | "
            f"lr={cfg.training.lr:g} | backbone lr={cfg.training.lr * cfg.training.backbone_lr_multiplier:g} | "
            f"batch={train_loader.batch_size} | workers={cfg.training.num_workers} | amp={use_amp}"
        )
        if pos_weights:
            weight_text = ", ".join(f"{task}={float(value.detach().cpu()):.3f}" for task, value in pos_weights.items())
            print(f"Using class-balanced positive weights: {weight_text}")

    try:
        for epoch in range(1, cfg.training.max_epochs + 1):
            set_backbone_trainable(model, epoch > cfg.training.freeze_backbone_epochs)
            model.train()
            train_loss_sum = 0.0
            train_batches = 0
            batch_iter = tqdm(
                train_loader,
                desc=f"{model_name} seed {seed} epoch {epoch}/{cfg.training.max_epochs}",
                leave=False,
                disable=not show_progress,
            )
            for batch in batch_iter:
                batch = move_batch_to_device(batch, device)
                if cfg.training.channels_last and device.type == "cuda":
                    batch["images"] = {
                        key: value.contiguous(memory_format=torch.channels_last)
                        for key, value in batch["images"].items()
                    }
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(batch["images"])
                    loss, _ = masked_bce_with_logits_loss(
                        logits,
                        batch,
                        pos_weights=pos_weights,
                        label_smoothing=cfg.classification.label_smoothing,
                    )
                scaler.scale(loss).backward()
                if cfg.training.gradient_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                train_loss_sum += float(loss.detach().cpu())
                train_batches += 1
                if show_progress and train_batches % cfg.training.log_every_n_batches == 0:
                    batch_iter.set_postfix(train_loss=f"{train_loss_sum / train_batches:.4f}")

            use_eval_tta = cfg.training.use_eval_tta
            cfg.training.use_eval_tta = False
            try:
                val_metrics, _ = evaluate_model_on_loader(model, val_loader, cfg, device)
            finally:
                cfg.training.use_eval_tta = use_eval_tta
            train_loss = train_loss_sum / max(train_batches, 1)
            raw_model = unwrap_model(model)
            gate_values = {
                f"gate_{token}": value
                for token, value in raw_model.token_gate_values().items()
            }
            attention_values = {
                f"attn_{task}_{token}": value
                for task, task_values in raw_model.token_attention_values().items()
                for token, value in task_values.items()
            }
            context_values = {
                f"context_scale_{task}": value
                for task, value in raw_model.context_scale_values().items()
            }
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_batches_used": train_batches,
                "lr_backbone": optimizer.param_groups[0]["lr"],
                "lr_gates": optimizer.param_groups[1]["lr"],
                "lr_other": optimizer.param_groups[2]["lr"],
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **gate_values,
                **attention_values,
                **context_values,
            }
            history.append(row)

            monitored = monitored_metric(val_metrics, metric_name)
            scheduler.step(monitored)
            raw_should_stop = stopper.step(monitored)
            should_stop = epoch >= cfg.training.min_epochs and raw_should_stop
            improved = stopper.best == monitored
            if improved:
                torch.save(
                    {
                        "model_state_dict": raw_model.state_dict(),
                        "model_name": model_name,
                        "seed": seed,
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                        "token_gate_values": raw_model.token_gate_values(),
                    },
                    best_path,
                )
            if show_progress:
                best_text = "saved best" if improved else f"no improvement {stopper.num_bad_epochs}/{cfg.training.patience}"
                lr_text = ",".join(f"{group['lr']:.2e}" for group in optimizer.param_groups)
                frozen_text = "backbone frozen" if epoch <= cfg.training.freeze_backbone_epochs else "backbone trainable"
                print(
                    f"Epoch {epoch:03d} | "
                    f"train loss={format_metric(train_loss)} | "
                    f"val loss={format_metric(val_metrics['loss'])} | "
                    f"val mean PR-AUC={format_metric(val_metrics['mean_pr_auc'])} | "
                    f"val mean AUC={format_metric(val_metrics['mean_auc'])} | "
                    f"used batches={train_batches} | "
                    f"lr={lr_text} | {frozen_text} | {best_text}"
                )
                if cfg.model.use_clinical_token_gates:
                    gate_text = ", ".join(
                        f"{token}={value:.4f}"
                        for token, value in raw_model.token_gate_values().items()
                    )
                    print(f"  token gates: {gate_text}")
                attention_values = raw_model.token_attention_values()
                if attention_values:
                    for task, task_weights in attention_values.items():
                        attention_text = ", ".join(
                            f"{token}={value:.4f}"
                            for token, value in task_weights.items()
                        )
                        print(f"  {task} attention: {attention_text}")
                context_values = raw_model.context_scale_values()
                if context_values:
                    context_text = ", ".join(
                        f"{task}={value:.4f}"
                        for task, value in context_values.items()
                    )
                    print(f"  context scales: {context_text}")
            if should_stop:
                if show_progress:
                    print(f"Early stopping at epoch {epoch}; best {metric_name}={format_metric(stopper.best)}")
                break
    finally:
        del train_loader, val_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pd.DataFrame(history).to_csv(ckpt_dir / f"seed_{seed}_training_history.csv", index=False)
    if show_progress:
        print(f"Training history saved: {ckpt_dir / f'seed_{seed}_training_history.csv'}")
        print(f"Best checkpoint saved: {best_path}")
    if best_path is None:
        raise RuntimeError("Training did not create a checkpoint.")
    return best_path


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(get_config().model_inputs.keys()), required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = get_config()
    seeds = [args.seed] if args.seed is not None else cfg.training.seeds
    for seed in seeds:
        train_one_seed(cfg, args.model, seed, show_progress=not args.quiet)


if __name__ == "__main__":
    main()
