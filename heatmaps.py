import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config, configure_run_paths, get_config
from dataset import KneeXrayClassificationDataset, build_transforms, knee_collate_fn
from evaluate import load_model_from_checkpoint, predict_on_loader
from losses import TASKS
from utils import move_batch_to_device, resolve_device, set_seed


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
TARGET_LAYER_INDEX = {
    "layer2": -4,
    "layer3": -3,
    "layer4": -2,
}


def build_loader(df: pd.DataFrame, cfg: Config, model_name: str) -> DataLoader:
    dataset = KneeXrayClassificationDataset(
        df,
        cfg,
        model_name,
        transform=build_transforms(cfg, train=False, model_name=model_name),
        check_image_files=True,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=knee_collate_fn,
        pin_memory=cfg.training.pin_memory,
    )


def prediction_path(cfg: Config, model_name: str, seed: int, split: str) -> Path:
    return cfg.paths.prediction_dir / model_name / f"seed_{seed}_{split}_predictions.csv"


def load_or_create_predictions(
    cfg: Config,
    model_name: str,
    seed: int,
    split: str,
    model,
    device: torch.device,
) -> pd.DataFrame:
    path = prediction_path(cfg, model_name, seed, split)
    if path.exists():
        return pd.read_csv(path)

    split_df = pd.read_csv(cfg.paths.split_dir / model_name / f"{split}.csv")
    loader = build_loader(split_df, cfg, model_name)
    pred_df = predict_on_loader(model, loader, cfg, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(path, index=False)
    return pred_df


def select_knees_for_task(
    pred_df: pd.DataFrame,
    task: str,
    selection: str,
    top_fraction: float,
    max_knees: int,
) -> pd.DataFrame:
    task_df = pred_df[pred_df[f"{task}_mask"] == 1].copy()
    if selection in {"positive", "positive_top_prob"}:
        task_df = task_df[task_df[f"{task}_label"] == 1].copy()
    if selection in {"positive_top_prob", "top_prob"}:
        task_df = task_df.sort_values(f"{task}_prob", ascending=False)
        keep_n = max(1, int(math.ceil(len(task_df) * top_fraction)))
        task_df = task_df.head(keep_n)
    if max_knees > 0:
        task_df = task_df.head(max_knees)
    if task_df.empty:
        raise ValueError(f"No knees selected for task={task}, selection={selection}.")
    return task_df.reset_index(drop=True)


class SharedEncoderGradCAM:
    def __init__(self, model, target_layer: str = "layer3"):
        self.model = model
        self.activations: List[torch.Tensor] = []
        self.gradients: List[torch.Tensor] = []
        if target_layer not in TARGET_LAYER_INDEX:
            raise ValueError(f"target_layer must be one of {sorted(TARGET_LAYER_INDEX)}")
        # Hook a ResNet block, not the full encoder output. The full `features`
        # module ends with avgpool, whose 1x1 output cannot produce spatially
        # meaningful Grad-CAM maps. layer3 is a good default for medical images:
        # finer than layer4, less noisy than layer2.
        self.target_layer_name = target_layer
        self.target_layer = model.image_encoder.features[TARGET_LAYER_INDEX[target_layer]]
        self.handles = [
            self.target_layer.register_forward_hook(self._forward_hook),
            self.target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def _forward_hook(self, module, inputs, output) -> None:
        self.activations.append(output)

    def _backward_hook(self, module, grad_input, grad_output) -> None:
        self.gradients.append(grad_output[0])

    def clear(self) -> None:
        self.activations = []
        self.gradients = []

    def heatmaps(self, images: Dict[str, torch.Tensor], task: str, size: int) -> Dict[str, torch.Tensor]:
        self.clear()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(images)
        score = logits[task].sum()
        score.backward()

        gradients = list(reversed(self.gradients))
        if len(self.activations) != len(self.model.required_images) or len(gradients) != len(self.model.required_images):
            raise RuntimeError(
                "Grad-CAM hook count did not match model inputs. "
                f"activations={len(self.activations)}, gradients={len(gradients)}, inputs={len(self.model.required_images)}"
            )

        cams = {}
        for index, image_key in enumerate(self.model.required_images):
            activation = self.activations[index]
            gradient = gradients[index]
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activation).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=(size, size), mode="bilinear", align_corners=False)
            cam = cam[0, 0]
            cam = cam - cam.min()
            denom = cam.max().clamp_min(1e-8)
            cams[image_key] = (cam / denom).detach().cpu()
        return cams


def shift_images(images: Dict[str, torch.Tensor], shift_x: float, shift_y: float) -> Dict[str, torch.Tensor]:
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


def add_noise(images: Dict[str, torch.Tensor], noise_std: float) -> Dict[str, torch.Tensor]:
    if noise_std <= 0:
        return images
    return {
        key: value + torch.randn_like(value) * noise_std
        for key, value in images.items()
    }


def smooth_heatmaps(
    cam: SharedEncoderGradCAM,
    images: Dict[str, torch.Tensor],
    task: str,
    size: int,
    samples: int,
    noise_std: float,
    translate_fraction: float,
) -> Dict[str, torch.Tensor]:
    samples = max(1, int(samples))
    sums: Dict[str, torch.Tensor] = {}
    for sample_idx in range(samples):
        if sample_idx == 0:
            augmented = images
        else:
            max_shift = float(translate_fraction) * 2.0
            shift_x = float(torch.empty((), device=next(iter(images.values())).device).uniform_(-max_shift, max_shift))
            shift_y = float(torch.empty((), device=next(iter(images.values())).device).uniform_(-max_shift, max_shift))
            augmented = add_noise(shift_images(images, shift_x, shift_y), noise_std)
        cams = cam.heatmaps(augmented, task, size)
        for image_key, heatmap in cams.items():
            sums[image_key] = sums.get(image_key, torch.zeros_like(heatmap)) + heatmap
    return {image_key: heatmap_sum / samples for image_key, heatmap_sum in sums.items()}


def patch_attention_heatmaps(model, images: Dict[str, torch.Tensor], task: str, size: int) -> Dict[str, torch.Tensor]:
    logits = model(images)
    _ = logits[task]
    attentions = model.token_attention_values().get(task, {})
    grid = int(model.patch_grid_size)
    values = []
    for row in range(grid):
        for col in range(grid):
            values.append(float(attentions.get(f"r{row}_c{col}", 0.0)))
    heatmap = torch.tensor(values, dtype=torch.float32).reshape(1, 1, grid, grid)
    heatmap = F.interpolate(heatmap, size=(size, size), mode="bicubic", align_corners=False)
    sigma = max(3, int(round(size / grid * 0.35)))
    heatmap = gaussian_blur_2d(heatmap, sigma=sigma)[0, 0]
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / heatmap.max().clamp_min(1e-8)
    return {model.required_images[0]: heatmap.cpu()}


def gaussian_blur_2d(tensor: torch.Tensor, sigma: int) -> torch.Tensor:
    if sigma <= 0:
        return tensor
    radius = int(max(1, round(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=tensor.dtype, device=tensor.device)
    kernel_1d = torch.exp(-(coords ** 2) / (2 * float(sigma) ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_x = kernel_1d.view(1, 1, 1, -1)
    kernel_y = kernel_1d.view(1, 1, -1, 1)
    out = F.pad(tensor, (radius, radius, 0, 0), mode="reflect")
    out = F.conv2d(out, kernel_x)
    out = F.pad(out, (0, 0, radius, radius), mode="reflect")
    out = F.conv2d(out, kernel_y)
    return out


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu()[0] * IMAGENET_STD + IMAGENET_MEAN
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return image


def foreground_mask(image: np.ndarray) -> np.ndarray:
    gray = image.mean(axis=2)
    border = max(3, int(round(min(gray.shape) * 0.03)))
    border_pixels = np.concatenate(
        [
            gray[:border, :].ravel(),
            gray[-border:, :].ravel(),
            gray[:, :border].ravel(),
            gray[:, -border:].ravel(),
        ]
    )
    background = float(np.median(border_pixels))
    mad = float(np.median(np.abs(border_pixels - background)))
    threshold = max(0.03, 4.0 * mad)
    mask = np.abs(gray - background) > threshold
    if mask.mean() < 0.02:
        low, high = np.percentile(gray, [5, 95])
        mask = (gray > low + 0.10 * (high - low)) & (gray < high - 0.02 * (high - low))
    return mask


def heatmap_qc(image: np.ndarray, heatmap: np.ndarray) -> Dict[str, float]:
    mask = foreground_mask(image)
    total = float(heatmap.sum()) + 1e-8
    inside = float(heatmap[mask].sum()) / total if mask.any() else np.nan
    outside = 1.0 - inside if np.isfinite(inside) else np.nan
    peak_y, peak_x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    height, width = heatmap.shape
    return {
        "foreground_fraction": float(mask.mean()),
        "heat_inside_foreground": inside,
        "heat_outside_foreground": outside,
        "peak_x_fraction": float((peak_x + 0.5) / width),
        "peak_y_fraction": float((peak_y + 0.5) / height),
    }


def save_overlay(path: Path, image: np.ndarray, heatmap: np.ndarray, title: str = None) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap="gray")
    plt.imshow(heatmap, cmap="jet", alpha=0.42, vmin=0, vmax=1)
    if title:
        plt.title(title, fontsize=8)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=250, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_heatmap_outputs(
    out_dir: Path,
    task: str,
    image_key: str,
    avg_image: np.ndarray,
    avg_heatmap: np.ndarray,
) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = out_dir / f"{task}_{image_key}_average_heatmap.png"
    overlay_path = out_dir / f"{task}_{image_key}_average_overlay.png"

    plt.figure(figsize=(4, 4))
    plt.imshow(avg_heatmap, cmap="jet", vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(heatmap_path, dpi=250, bbox_inches="tight", pad_inches=0)
    plt.close()

    save_overlay(overlay_path, avg_image, avg_heatmap)


def generate_task_heatmaps(
    cfg: Config,
    model_name: str,
    seed: int,
    split: str,
    task: str,
    selection: str,
    top_fraction: float,
    max_knees: int,
    target_layer: str = "layer3",
    smooth_samples: int = 12,
    smooth_noise: float = 0.02,
    smooth_translate: float = 0.02,
    save_individual: int = 16,
) -> Dict[str, Path]:
    set_seed(seed)
    device = resolve_device(cfg.training.device)
    ckpt = cfg.paths.checkpoint_dir / model_name / f"seed_{seed}_best.pt"
    model = load_model_from_checkpoint(cfg, model_name, ckpt, device)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    model.eval()

    pred_df = load_or_create_predictions(cfg, model_name, seed, split, model, device)
    selected = select_knees_for_task(pred_df, task, selection, top_fraction, max_knees)
    selected_ids = set(selected["knee_id"].astype(str))

    split_df = pd.read_csv(cfg.paths.split_dir / model_name / f"{split}.csv")
    selected_df = split_df[split_df[cfg.columns.knee_id].astype(str).isin(selected_ids)].copy()
    selected_df = selected_df.set_index(cfg.columns.knee_id).loc[selected["knee_id"].astype(str)].reset_index()
    loader = build_loader(selected_df, cfg, model_name)

    heat_sums: Dict[str, np.ndarray] = {}
    image_sums: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    qc_rows = []

    out_dir = (
        cfg.paths.metrics_dir
        / "heatmaps"
        / model_name
        / f"seed_{seed}_{split}_{selection}_{target_layer}_smooth{smooth_samples}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    individual_dir = out_dir / "individual"

    use_patch_attention = getattr(raw_model, "architecture", "") == "patch_mil"
    cam = None if use_patch_attention else SharedEncoderGradCAM(raw_model, target_layer=target_layer)
    try:
        for batch_index, batch in enumerate(loader):
            device_batch = move_batch_to_device(batch, device)
            if cfg.training.channels_last and device.type == "cuda":
                device_batch["images"] = {
                    key: value.contiguous(memory_format=torch.channels_last)
                    for key, value in device_batch["images"].items()
                }
            heatmap_size = next(iter(device_batch["images"].values())).shape[-1]
            if use_patch_attention:
                with torch.no_grad():
                    cams = patch_attention_heatmaps(raw_model, device_batch["images"], task, heatmap_size)
            else:
                cams = smooth_heatmaps(
                    cam,
                    device_batch["images"],
                    task,
                    heatmap_size,
                    samples=smooth_samples,
                    noise_std=smooth_noise,
                    translate_fraction=smooth_translate,
                )
            for image_key, heatmap_tensor in cams.items():
                heatmap = heatmap_tensor.numpy()
                image = denormalize_image(device_batch["images"][image_key])
                heat_sums[image_key] = heat_sums.get(image_key, np.zeros_like(heatmap)) + heatmap
                image_sums[image_key] = image_sums.get(image_key, np.zeros_like(image)) + image
                counts[image_key] = counts.get(image_key, 0) + 1
                qc = heatmap_qc(image, heatmap)
                qc.update(
                    {
                        "task": task,
                        "model": model_name,
                        "seed": seed,
                        "split": split,
                        "selection": selection,
                        "target_layer": target_layer,
                        "smooth_samples": smooth_samples,
                        "image_key": image_key,
                        "participant_id": batch["participant_id"][0],
                        "knee_id": batch["knee_id"][0],
                    }
                )
                qc_rows.append(qc)
                if save_individual > 0 and batch_index < save_individual:
                    save_overlay(
                        individual_dir / f"{task}_{image_key}_{batch_index:03d}_{batch['knee_id'][0]}_overlay.png",
                        image,
                        heatmap,
                        title=f"{task} {image_key} {batch['knee_id'][0]}",
                    )
    finally:
        if cam is not None:
            cam.close()

    selected.to_csv(out_dir / f"{task}_selected_knees.csv", index=False)
    pd.DataFrame(qc_rows).to_csv(out_dir / f"{task}_heatmap_qc.csv", index=False)

    outputs = {}
    for image_key, heat_sum in heat_sums.items():
        avg_heatmap = heat_sum / counts[image_key]
        avg_image = image_sums[image_key] / counts[image_key]
        avg_heatmap = avg_heatmap - avg_heatmap.min()
        avg_heatmap = avg_heatmap / max(avg_heatmap.max(), 1e-8)
        save_heatmap_outputs(out_dir, task, image_key, avg_image, avg_heatmap)
        outputs[image_key] = out_dir / f"{task}_{image_key}_average_overlay.png"
    return outputs


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(get_config().model_inputs.keys()), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--task", default="both", choices=["both", *TASKS])
    parser.add_argument("--selection", default="positive_top_prob", choices=["positive_top_prob", "positive", "top_prob"])
    parser.add_argument("--top-fraction", type=float, default=0.20)
    parser.add_argument("--max-knees", type=int, default=64)
    parser.add_argument("--target-layer", default="layer3", choices=sorted(TARGET_LAYER_INDEX))
    parser.add_argument("--smooth-samples", type=int, default=12)
    parser.add_argument("--smooth-noise", type=float, default=0.02)
    parser.add_argument("--smooth-translate", type=float, default=0.02)
    parser.add_argument("--save-individual", type=int, default=16)
    args = parser.parse_args()

    cfg = get_config()
    if args.run_id is not None:
        configure_run_paths(cfg.paths, args.run_id)

    tasks = TASKS if args.task == "both" else (args.task,)
    for task in tasks:
        outputs = generate_task_heatmaps(
            cfg,
            args.model,
            args.seed,
            args.split,
            task,
            args.selection,
            args.top_fraction,
            args.max_knees,
            target_layer=args.target_layer,
            smooth_samples=args.smooth_samples,
            smooth_noise=args.smooth_noise,
            smooth_translate=args.smooth_translate,
            save_individual=args.save_individual,
        )
        for image_key, path in outputs.items():
            print(f"{args.model} {task} {image_key}: {path}")


if __name__ == "__main__":
    main()
