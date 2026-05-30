import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from config import get_config
from model import KneeOAClassificationTransformer


def mask_for_model(model_name: str) -> np.ndarray:
    cfg = get_config()
    model = KneeOAClassificationTransformer(cfg, model_name)
    mask = model._patch_valid_mask(device="cpu").reshape(cfg.training.patch_grid_size, cfg.training.patch_grid_size)
    return mask.numpy().astype(int)


def plot_mask(model_name: str, out_dir: Path) -> Path:
    mask = mask_for_model(model_name)
    grid = mask.shape[0]
    path = out_dir / f"{model_name}_patch_mask.png"
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = np.zeros((grid, grid, 3), dtype=float)
    colors[mask == 1] = [0.25, 0.70, 0.35]
    colors[mask == 0] = [0.85, 0.20, 0.20]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(colors)
    ax.set_xticks(np.arange(grid))
    ax.set_yticks(np.arange(grid))
    ax.set_xticklabels([f"c{idx}" for idx in range(grid)])
    ax.set_yticklabels([f"r{idx}" for idx in range(grid)])
    ax.set_xticks(np.arange(-0.5, grid, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row in range(grid):
        for col in range(grid):
            label = "keep" if mask[row, col] else "exclude"
            ax.text(col, row, label, ha="center", va="center", color="white", fontsize=8, weight="bold")
    ax.set_title(model_name)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/patch_mask_preview")
    parser.add_argument("--models", nargs="*", default=["patch_mil_m30_pa", "patch_mil_m30_lat"])
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    for model_name in args.models:
        print(plot_mask(model_name, out_dir))


if __name__ == "__main__":
    main()
