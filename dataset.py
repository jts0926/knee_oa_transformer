from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    from torchvision import transforms
except ImportError:
    torch = None
    Image = None
    transforms = None
    Dataset = object

from config import Config, IMAGE_KEY_TO_COLUMN_ATTR
from utils import require_columns, safe_path_exists


class ForegroundCrop:
    """Crop away stable blank borders while leaving generous knee context."""

    def __init__(self, margin: float = 0.08, min_size_fraction: float = 0.35):
        self.margin = float(margin)
        self.min_size_fraction = float(min_size_fraction)

    def __call__(self, image: Image.Image) -> Image.Image:
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        height, width = gray.shape
        bbox = self._foreground_bbox(gray)
        if bbox is None:
            return image
        left, top, right, bottom = self._expand_bbox(bbox, width, height)
        cropped = image.crop((left, top, right, bottom))
        return self._pad_to_square(cropped, image)

    def _foreground_bbox(self, gray: np.ndarray) -> Tuple[int, int, int, int] | None:
        height, width = gray.shape
        border = max(3, int(round(min(height, width) * 0.03)))
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
        threshold = max(8.0, 4.0 * mad)
        mask = np.abs(gray - background) > threshold

        # If the border is not representative, fall back to a robust percentile mask.
        if mask.mean() < 0.02:
            low, high = np.percentile(gray, [5, 95])
            mask = (gray > low + 0.10 * (high - low)) & (gray < high - 0.02 * (high - low))

        row_fraction = mask.mean(axis=1)
        col_fraction = mask.mean(axis=0)
        row_threshold = max(0.01, min(0.10, mask.mean() * 0.50))
        col_threshold = max(0.01, min(0.10, mask.mean() * 0.50))
        rows = np.flatnonzero(row_fraction > row_threshold)
        cols = np.flatnonzero(col_fraction > col_threshold)
        if len(rows) == 0 or len(cols) == 0:
            return None

        top, bottom = int(rows[0]), int(rows[-1]) + 1
        left, right = int(cols[0]), int(cols[-1]) + 1
        min_height = height * self.min_size_fraction
        min_width = width * self.min_size_fraction
        if (bottom - top) < min_height or (right - left) < min_width:
            return None
        if (bottom - top) > 0.98 * height and (right - left) > 0.98 * width:
            return None
        return left, top, right, bottom

    def _expand_bbox(self, bbox: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
        left, top, right, bottom = bbox
        box_width = right - left
        box_height = bottom - top
        pad_x = int(round(box_width * self.margin))
        pad_y = int(round(box_height * self.margin))
        return (
            max(0, left - pad_x),
            max(0, top - pad_y),
            min(width, right + pad_x),
            min(height, bottom + pad_y),
        )

    def _pad_to_square(self, cropped: Image.Image, original: Image.Image) -> Image.Image:
        width, height = cropped.size
        side = max(width, height)
        if width == height:
            return cropped
        fill = self._border_fill(original)
        canvas = Image.new(cropped.mode, (side, side), color=fill)
        canvas.paste(cropped, ((side - width) // 2, (side - height) // 2))
        return canvas

    @staticmethod
    def _border_fill(image: Image.Image):
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
        border = max(3, int(round(min(gray.shape) * 0.03)))
        pixels = np.concatenate(
            [
                gray[:border, :].ravel(),
                gray[-border:, :].ravel(),
                gray[:, :border].ravel(),
                gray[:, -border:].ravel(),
            ]
        )
        value = int(np.median(pixels))
        if image.mode == "RGB":
            return (value, value, value)
        return value


def image_key_to_column(cfg: Config, image_key: str) -> str:
    return getattr(cfg.columns, IMAGE_KEY_TO_COLUMN_ATTR[image_key])


def required_columns_for_model(cfg: Config, model_name: str) -> List[str]:
    model_spec = cfg.model_inputs[model_name]
    image_columns = [
        image_key_to_column(cfg, key)
        for key in model_spec["required_images"]
    ]
    return [
        cfg.columns.participant_id,
        cfg.columns.knee_id,
        cfg.columns.incidence_label,
        cfg.columns.incidence_mask,
        cfg.columns.progression_label,
        cfg.columns.progression_mask,
        *image_columns,
    ]


def apply_model_filter(
    df: pd.DataFrame,
    cfg: Config,
    model_name: str,
    check_image_files: bool = True,
) -> pd.DataFrame:
    """Keep labeled knees with the images required by one model configuration."""
    require_columns(df, required_columns_for_model(cfg, model_name))
    c = cfg.columns

    out = df.copy()
    for column in [
        c.incidence_label,
        c.incidence_mask,
        c.progression_label,
        c.progression_mask,
    ]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(int)
    out = out[(out[c.incidence_mask] == 1) | (out[c.progression_mask] == 1)].copy()

    required_images = cfg.model_inputs[model_name]["required_images"]
    for image_key in required_images:
        column = image_key_to_column(cfg, image_key)
        out = out[out[column].notna() & (out[column].astype(str).str.strip() != "")]
        if check_image_files:
            out = out[out[column].apply(safe_path_exists)]

    return out.reset_index(drop=True)


def model_image_size(cfg: Config, model_name: str = None) -> int:
    if model_name is not None and cfg.model_inputs[model_name].get("architecture") == "patch_mil":
        return int(cfg.training.patch_image_size)
    return int(cfg.training.image_size)


def build_transforms(cfg: Config, train: bool = True, model_name: str = None) -> transforms.Compose:
    if transforms is None:
        raise ImportError("torchvision is required to build image transforms. Install project requirements first.")
    image_size = model_image_size(cfg, model_name)
    transform_steps = []
    if cfg.training.use_foreground_crop:
        transform_steps.append(
            ForegroundCrop(
                margin=cfg.training.foreground_crop_margin,
                min_size_fraction=cfg.training.foreground_crop_min_size_fraction,
            )
        )
    if train:
        transform_steps.append(transforms.Resize((image_size, image_size)))
        if cfg.training.use_horizontal_flip:
            transform_steps.append(transforms.RandomHorizontalFlip(p=0.5))
        if (
            cfg.training.random_rotation_degrees > 0
            or cfg.training.random_translate_fraction > 0
            or cfg.training.random_scale_min != 1.0
            or cfg.training.random_scale_max != 1.0
        ):
            transform_steps.append(
                transforms.RandomAffine(
                    degrees=cfg.training.random_rotation_degrees,
                    translate=(cfg.training.random_translate_fraction, cfg.training.random_translate_fraction),
                    scale=(cfg.training.random_scale_min, cfg.training.random_scale_max),
                    fill=0,
                )
            )
        transform_steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return transforms.Compose(transform_steps)
    transform_steps.extend(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(
        transform_steps
    )


class KneeXrayClassificationDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        cfg: Config,
        model_name: str,
        transform: transforms.Compose,
        check_image_files: bool = True,
    ):
        self.cfg = cfg
        self.model_name = model_name
        self.transform = transform
        self.df = apply_model_filter(dataframe, cfg, model_name, check_image_files=check_image_files)
        self.required_images: List[str] = list(cfg.model_inputs[model_name]["required_images"])

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: str) -> torch.Tensor:
        if torch is None or Image is None:
            raise ImportError("PyTorch and Pillow are required to load images. Install project requirements first.")
        with Image.open(Path(path).expanduser()) as img:
            img = img.convert("RGB")
            return self.transform(img)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.df.iloc[index]
        images = {}
        for image_key in self.required_images:
            column = image_key_to_column(self.cfg, image_key)
            images[image_key] = self._load_image(row[column])

        return {
            "participant_id": str(row[self.cfg.columns.participant_id]),
            "knee_id": str(row[self.cfg.columns.knee_id]),
            "images": images,
            "incidence_label": torch.tensor(float(row[self.cfg.columns.incidence_label]), dtype=torch.float32),
            "incidence_mask": torch.tensor(float(row[self.cfg.columns.incidence_mask]), dtype=torch.float32),
            "progression_label": torch.tensor(float(row[self.cfg.columns.progression_label]), dtype=torch.float32),
            "progression_mask": torch.tensor(float(row[self.cfg.columns.progression_mask]), dtype=torch.float32),
        }


def knee_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    if torch is None:
        raise ImportError("PyTorch is required to collate training batches. Install project requirements first.")
    image_keys = batch[0]["images"].keys()
    images = {
        key: torch.stack([item["images"][key] for item in batch])
        for key in image_keys
    }
    return {
        "participant_id": [item["participant_id"] for item in batch],
        "knee_id": [item["knee_id"] for item in batch],
        "images": images,
        "incidence_label": torch.stack([item["incidence_label"] for item in batch]),
        "incidence_mask": torch.stack([item["incidence_mask"] for item in batch]),
        "progression_label": torch.stack([item["progression_label"] for item in batch]),
        "progression_mask": torch.stack([item["progression_mask"] for item in batch]),
    }
