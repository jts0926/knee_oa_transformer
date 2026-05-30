import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    train_test_split = None

from config import Config, ensure_output_dirs, get_config
from dataset import apply_model_filter
from utils import require_columns, set_seed


def participant_event_table(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    c = cfg.columns
    table = (
        df.groupby(c.participant_id)
        .agg(
            participant_incidence_event=(c.incidence_label, "max"),
            participant_incidence_n=(c.incidence_mask, "sum"),
            participant_progression_event=(c.progression_label, "max"),
            participant_progression_n=(c.progression_mask, "sum"),
        )
        .reset_index()
    )
    table["participant_any_event"] = table[["participant_incidence_event", "participant_progression_event"]].max(axis=1)
    if cfg.split.stratify_by_m30_kl and cfg.split.m30_kl_col in df.columns:
        kl = df[[c.participant_id, cfg.split.m30_kl_col]].copy()
        kl[cfg.split.m30_kl_col] = pd.to_numeric(kl[cfg.split.m30_kl_col], errors="coerce")
        kl_table = (
            kl.groupby(c.participant_id)[cfg.split.m30_kl_col]
            .max()
            .reset_index()
            .rename(columns={cfg.split.m30_kl_col: "participant_m30_kl"})
        )
        table = table.merge(kl_table, on=c.participant_id, how="left")
        table["participant_m30_kl_group"] = table["participant_m30_kl"].apply(kl_group)
        table["event_kl_stratum"] = (
            table["participant_any_event"].astype(str) + "_kl_" + table["participant_m30_kl_group"].astype(str)
        )
    else:
        table["participant_m30_kl"] = np.nan
        table["participant_m30_kl_group"] = "missing"
        table["event_kl_stratum"] = table["participant_any_event"].astype(str)
    return table


def kl_group(value: object) -> str:
    if pd.isna(value):
        return "missing"
    grade = int(value)
    if grade <= 1:
        return "0_1"
    return str(grade)


def choose_stratify(participant_df: pd.DataFrame, test_size: float):
    candidates = ["event_kl_stratum", "participant_m30_kl_group", "participant_any_event"]
    n = len(participant_df)
    expected_test_n = max(1, int(round(n * test_size)))
    for column in candidates:
        if column not in participant_df.columns:
            continue
        values = participant_df[column]
        counts = values.value_counts()
        if values.nunique() < 2 or counts.min() < 2:
            continue
        if values.nunique() > expected_test_n:
            continue
        return values
    return None


def safe_stratified_split(
    participant_df: pd.DataFrame,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stratify = choose_stratify(participant_df, test_size)
    if train_test_split is None:
        return fallback_train_test_split(participant_df, test_size, seed, stratify)
    try:
        left, right = train_test_split(
            participant_df,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        left, right = train_test_split(
            participant_df,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )
    return left.reset_index(drop=True), right.reset_index(drop=True)


def fallback_train_test_split(
    participant_df: pd.DataFrame,
    test_size: float,
    seed: int,
    stratify,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    if stratify is None:
        indices = participant_df.index.to_numpy().copy()
        rng.shuffle(indices)
        n_test = max(1, int(round(len(indices) * test_size)))
        right_idx = indices[:n_test]
        left_idx = indices[n_test:]
        return participant_df.loc[left_idx].reset_index(drop=True), participant_df.loc[right_idx].reset_index(drop=True)

    right_indices = []
    for _, group in participant_df.groupby(stratify):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        n_test = int(round(len(indices) * test_size))
        if test_size > 0 and len(indices) > 1:
            n_test = min(max(1, n_test), len(indices) - 1)
        right_indices.extend(indices[:n_test].tolist())

    right_idx = np.asarray(right_indices, dtype=int)
    left_idx = participant_df.index.difference(right_idx).to_numpy()
    return participant_df.loc[left_idx].reset_index(drop=True), participant_df.loc[right_idx].reset_index(drop=True)


def make_participant_splits(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    set_seed(cfg.split.seed)
    participant_df = participant_event_table(df, cfg)
    train_val, test = safe_stratified_split(participant_df, cfg.split.test_frac, cfg.split.seed)
    val_relative = cfg.split.val_frac / (cfg.split.train_frac + cfg.split.val_frac)
    train, val = safe_stratified_split(train_val, val_relative, cfg.split.seed + 1)
    return train, val, test


def subset_by_participants(df: pd.DataFrame, cfg: Config, participants: pd.DataFrame) -> pd.DataFrame:
    ids = set(participants[cfg.columns.participant_id].astype(str))
    mask = df[cfg.columns.participant_id].astype(str).isin(ids)
    return df[mask].reset_index(drop=True)


def subset_by_knees(df: pd.DataFrame, cfg: Config, knees: pd.DataFrame) -> pd.DataFrame:
    ids = set(knees[cfg.columns.knee_id].astype(str))
    mask = df[cfg.columns.knee_id].astype(str).isin(ids)
    return df[mask].reset_index(drop=True)


def participant_split_dir(cfg: Config) -> Path:
    return cfg.paths.split_dir / "participants"


def complete_case_knee_path(cfg: Config) -> Path:
    return participant_split_dir(cfg) / "complete_case_knees.csv"


def reference_complete_case_df(
    cfg: Config,
    check_image_files: bool = True,
    reference_model: str = None,
) -> pd.DataFrame:
    reference_model = reference_model or cfg.split.reference_model
    raw = pd.read_csv(cfg.paths.data_csv)
    require_columns(
        raw,
        [
            cfg.columns.participant_id,
            cfg.columns.knee_id,
            cfg.columns.incidence_mask,
            cfg.columns.progression_mask,
        ],
    )
    return apply_model_filter(raw, cfg, reference_model, check_image_files=check_image_files)


def save_global_participant_splits(
    cfg: Config,
    reference_model: str = None,
    check_image_files: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_output_dirs(cfg)
    reference_model = reference_model or cfg.split.reference_model
    model_df = reference_complete_case_df(cfg, check_image_files=check_image_files, reference_model=reference_model)
    train_ids, val_ids, test_ids = make_participant_splits(model_df, cfg)

    out_dir = participant_split_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_df.to_csv(complete_case_knee_path(cfg), index=False)
    train_ids.to_csv(out_dir / "train_participants.csv", index=False)
    val_ids.to_csv(out_dir / "val_participants.csv", index=False)
    test_ids.to_csv(out_dir / "test_participants.csv", index=False)
    metadata = {
        "reference_model": reference_model,
        "seed": cfg.split.seed,
        "train_frac": cfg.split.train_frac,
        "val_frac": cfg.split.val_frac,
        "test_frac": cfg.split.test_frac,
        "stratify_by_m30_kl": cfg.split.stratify_by_m30_kl,
        "m30_kl_col": cfg.split.m30_kl_col,
        "complete_case_knees": int(len(model_df)),
        "complete_case_participants": int(model_df[cfg.columns.participant_id].nunique()),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return train_ids, val_ids, test_ids


def load_or_create_global_participant_splits(
    cfg: Config,
    check_image_files: bool = True,
    force_recreate: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir = participant_split_dir(cfg)
    paths = [out_dir / f"{name}_participants.csv" for name in ["train", "val", "test"]]
    complete_case_path = complete_case_knee_path(cfg)
    metadata_path = out_dir / "metadata.json"
    metadata_ok = False
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_ok = (
            metadata.get("reference_model") == cfg.split.reference_model
            and metadata.get("seed") == cfg.split.seed
            and metadata.get("stratify_by_m30_kl") == cfg.split.stratify_by_m30_kl
            and metadata.get("m30_kl_col") == cfg.split.m30_kl_col
        )
    if not force_recreate and all(path.exists() for path in paths) and complete_case_path.exists() and metadata_ok:
        return tuple(pd.read_csv(path) for path in paths)
    return save_global_participant_splits(cfg, check_image_files=check_image_files)


def save_splits_for_model(
    cfg: Config,
    model_name: str,
    check_image_files: bool = True,
    force_recreate_participant_splits: bool = False,
) -> None:
    ensure_output_dirs(cfg)
    raw = pd.read_csv(cfg.paths.data_csv)
    require_columns(raw, [cfg.columns.participant_id, cfg.columns.incidence_mask, cfg.columns.progression_mask])
    model_df = apply_model_filter(raw, cfg, model_name, check_image_files=check_image_files)
    train_ids, val_ids, test_ids = load_or_create_global_participant_splits(
        cfg,
        check_image_files=check_image_files,
        force_recreate=force_recreate_participant_splits,
    )
    complete_case_df = pd.read_csv(complete_case_knee_path(cfg))
    model_df = subset_by_knees(model_df, cfg, complete_case_df)

    model_split_dir = cfg.paths.split_dir / model_name
    model_split_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        split_df = subset_by_participants(model_df, cfg, ids)
        split_df.to_csv(model_split_dir / f"{name}.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "split": name,
                "participants": split_df[cfg.columns.participant_id].nunique(),
                "knees": len(split_df),
                "incidence_task_knees": int(split_df[cfg.columns.incidence_mask].sum()),
                "incidence_events": int(split_df.loc[split_df[cfg.columns.incidence_mask] == 1, cfg.columns.incidence_label].sum()),
                "progression_task_knees": int(split_df[cfg.columns.progression_mask].sum()),
                "progression_events": int(split_df.loc[split_df[cfg.columns.progression_mask] == 1, cfg.columns.progression_label].sum()),
                "m30_kl_nonmissing": int(split_df[cfg.split.m30_kl_col].notna().sum()) if cfg.split.m30_kl_col in split_df.columns else 0,
                "m30_kl_mean": float(pd.to_numeric(split_df[cfg.split.m30_kl_col], errors="coerce").mean()) if cfg.split.m30_kl_col in split_df.columns else np.nan,
            }
            for name, split_df in [
                ("train", subset_by_participants(model_df, cfg, train_ids)),
                ("val", subset_by_participants(model_df, cfg, val_ids)),
                ("test", subset_by_participants(model_df, cfg, test_ids)),
            ]
        ]
    )
    summary.to_csv(model_split_dir / "summary.csv", index=False)
    if cfg.split.m30_kl_col in model_df.columns:
        kl_rows = []
        for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
            split_df = subset_by_participants(model_df, cfg, ids)
            groups = pd.to_numeric(split_df[cfg.split.m30_kl_col], errors="coerce").apply(kl_group)
            counts = groups.value_counts(dropna=False).sort_index()
            for group, count in counts.items():
                kl_rows.append(
                    {
                        "split": name,
                        "m30_kl_group": group,
                        "knees": int(count),
                    }
                )
        pd.DataFrame(kl_rows).to_csv(model_split_dir / "m30_kl_distribution.csv", index=False)


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(get_config().model_inputs.keys()), required=True)
    parser.add_argument("--reference-model", choices=list(get_config().model_inputs.keys()), default=None)
    parser.add_argument("--force-recreate-participant-splits", action="store_true")
    parser.add_argument("--skip-file-check", action="store_true")
    args = parser.parse_args()
    cfg = get_config()
    if args.reference_model is not None:
        cfg.split.reference_model = args.reference_model
    save_splits_for_model(
        cfg,
        args.model,
        check_image_files=not args.skip_file_check,
        force_recreate_participant_splits=args.force_recreate_participant_splits,
    )


if __name__ == "__main__":
    main()
