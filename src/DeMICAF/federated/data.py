r"""Data loading and preparation for the FL simulation.

This module is the ONLY place that reads from disk or decides which rows go
into each split.  Changing how samples are selected (different subset columns,
etc.) means only touching this file — the training code stays stable.

Structure
---------
ClientConfig   dataclass — filesystem / CSV config for one FL client
ClientSplit    dataclass — pre-loaded DataFrames for one FL client
load_client_splits   load train / val / test DataFrames for one client
load_evaluation_set  load the CheXpert test set for final external evaluation
make_data_loader     wrap a DataFrame in a DatasetCXR → DataLoader
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from DeMICAF.classifier import train_tfm, val_tfm
from DeMICAF.data.datasets import DatasetCXR

if TYPE_CHECKING:
    from pathlib import Path

# ── Dataclasses ────────────────────────────────────────────────────────────────


@dataclass
class ClientConfig:
    """Filesystem and CSV configuration for one FL client.

    Attributes
    ----------
    name
        Dataset subfolder name, e.g. ``"ChestX-ray8"``.
    harmonized_csv
        Filename of the label CSV relative to ``{datasets_root}/{name}/``.
    subsets_csv
        Filename of the subset-mask CSV relative to ``{datasets_root}/{name}/``.
    training_col
        Column in ``subsets_csv`` that marks the annotated training rows
        (all positive rows become the training set — no size limit is applied).
    val_pool_col
        Column in ``subsets_csv`` marking the broader training-eligible pool
        from which validation rows are sampled (seeded).
    test_col
        Column in ``subsets_csv`` marking the fixed held-out test split.
        If ``fl_test_splits.csv`` exists in the dataset directory the test rows
        are read from that file instead (see :func:`_read_fl_test_splits`).
    """

    name: str
    harmonized_csv: str
    subsets_csv: str
    training_col: str
    val_pool_col: str
    test_col: str


@dataclass
class ClientSplit:
    """Pre-loaded DataFrames for one FL client.

    Attributes
    ----------
    name
        Dataset name, mirrors :attr:`ClientConfig.name`.
    train_df
        All rows where ``training_col == 1``.  Used as the client training set.
    val_df
        ``n_val`` seeded rows drawn from the val pool (``val_pool_col == 1``)
        after excluding test rows and training rows.
    test_df
        All rows in the fixed test split (``fl_test_splits.csv`` if present,
        otherwise ``test_col == 1`` from ``subsets_csv``).
    root_dir
        Absolute path to ``datasets_root``.  Image paths in the DataFrames
        include the dataset-name prefix (e.g. ``"ChestX-ray8/images/..."``),
        so the root is the common parent of all dataset directories.
    """

    name: str
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    root_dir: Path


# ── Private helpers ────────────────────────────────────────────────────────────


def _seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker from the current Torch seed."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


def _read_fl_test_splits(
    dataset_dir: Path,
    merged: pd.DataFrame,
    image_column: str,
    test_col: str,
) -> pd.DataFrame:
    """Return the test DataFrame, preferring ``fl_test_splits.csv`` over ``test_col``.

    If ``fl_test_splits.csv`` exists in ``dataset_dir`` it is used; otherwise
    falls back to filtering ``merged`` by ``test_col == 1``.
    """
    fl_test_path = dataset_dir / "fl_test_splits.csv"
    if fl_test_path.exists():
        fl_splits = pd.read_csv(fl_test_path)
        test_paths = set(fl_splits[fl_splits["fl_test"] == 1][image_column].tolist())
        return merged[merged[image_column].isin(test_paths)].copy()

    if test_col not in merged.columns:
        raise ValueError(
            f"Column '{test_col}' not found in subsets CSV for dataset at '{dataset_dir}'. "
            f"Run setup_splits.py first, or check the test_col config value."
        )
    return merged[merged[test_col] == 1].copy()


# ── Public loading functions ───────────────────────────────────────────────────


def load_client_splits(
    datasets_root: Path,
    client_cfg: ClientConfig,
    label_cols: list[str],
    *,
    n_val: int,
    image_column: str,
    seed: int,
) -> ClientSplit:
    """Load train, val, and test DataFrames for one FL client.

    Split logic
    -----------
    * **Train** — all rows where ``training_col == 1``.  No size limit; the
      entire annotated set is used.
    * **Val** — ``n_val`` rows sampled (seeded) from the val pool
      (``val_pool_col == 1``) after removing test rows and training rows.
      Identical for the same ``seed``; can change when the training set changes.
    * **Test** — all rows from ``fl_test_splits.csv`` (preferred) or
      ``test_col == 1`` in ``subsets_csv``.  Fixed across all experiments.

    Raises
    ------
    FileNotFoundError
        If ``harmonized_csv`` or ``subsets_csv`` is missing.
    ValueError
        If the required columns are absent, or the val pool is too small.
    """
    dataset_dir = datasets_root / client_cfg.name
    harm_path = dataset_dir / client_cfg.harmonized_csv
    subs_path = dataset_dir / client_cfg.subsets_csv

    if not harm_path.exists():
        raise FileNotFoundError(f"Harmonized CSV not found: {harm_path}")
    if not subs_path.exists():
        raise FileNotFoundError(f"Subsets CSV not found: {subs_path}")

    harm = pd.read_csv(harm_path)
    subs = pd.read_csv(subs_path)
    merged = harm.merge(subs, on=image_column, how="inner")

    # ── Training split — always the exact annotated set, no exclusions ───────────
    if client_cfg.training_col not in merged.columns:
        raise ValueError(
            f"training_col '{client_cfg.training_col}' not found in {subs_path}. "
            f"Available columns: {list(merged.columns)}"
        )
    train_df = merged[merged[client_cfg.training_col] == 1].copy()
    train_image_paths: set[str] = set(train_df[image_column].tolist())

    # ── Test split — must not contain any training rows ────────────────────────
    test_df = _read_fl_test_splits(dataset_dir, merged, image_column, client_cfg.test_col)
    test_image_paths: set[str] = set(test_df[image_column].tolist())

    overlap = train_image_paths & test_image_paths
    if overlap:
        raise ValueError(
            f"{client_cfg.name}: {len(overlap)} image(s) appear in both the training set "
            f"(training_col='{client_cfg.training_col}') and the test split."
        )

    # ── Validation split (seeded draw from wider pool) ────────────────────────
    if client_cfg.val_pool_col not in merged.columns:
        raise ValueError(
            f"val_pool_col '{client_cfg.val_pool_col}' not found in {subs_path}. "
            f"Available columns: {list(merged.columns)}"
        )
    val_pool = merged[
        (merged[client_cfg.val_pool_col] == 1)
        & (~merged[image_column].isin(test_image_paths))
        & (~merged[image_column].isin(train_image_paths))
    ].copy()

    if len(val_pool) < n_val:
        raise ValueError(
            f"{client_cfg.name}: val pool has {len(val_pool)} rows but n_val={n_val} requested. "
            f"Reduce n_val or widen val_pool_col."
        )
    val_df = val_pool.sample(n=n_val, random_state=seed).reset_index(drop=True)

    # Keep only relevant columns to avoid confusing DatasetCXR.
    keep_cols = [image_column] + [c for c in label_cols if c in merged.columns]
    # root_dir is the parent of all dataset directories because image_path values
    # in the CSV include the dataset-name prefix (e.g. "ChestX-ray8/images/...").
    return ClientSplit(
        name=client_cfg.name,
        train_df=train_df[keep_cols].reset_index(drop=True),
        val_df=val_df[keep_cols].reset_index(drop=True),
        test_df=test_df[keep_cols].reset_index(drop=True),
        root_dir=datasets_root,
    )


def load_evaluation_set(
    datasets_root: Path,
    dataset_name: str,
    label_cols: list[str],
    *,
    subsets_csv: str,
    test_col: str,
    image_column: str,
) -> tuple[pd.DataFrame, Path]:
    """Load the test split of an evaluation-only dataset (e.g. CheXpert).

    Unlike client splits, this function always reads ``test_col`` from
    ``subsets_csv`` directly — no ``fl_test_splits.csv`` fallback.

    Raises
    ------
    FileNotFoundError
        If the harmonized or subsets CSV is missing.
    ValueError
        If ``test_col`` is absent from the subsets CSV.
    """
    root_dir = datasets_root / dataset_name
    harm_path = root_dir / "harmonized.csv"
    subs_path = root_dir / subsets_csv

    if not harm_path.exists():
        raise FileNotFoundError(f"Harmonized CSV not found: {harm_path}")
    if not subs_path.exists():
        raise FileNotFoundError(f"Subsets CSV not found: {subs_path}")

    harm = pd.read_csv(harm_path)
    subs = pd.read_csv(subs_path)
    merged = harm.merge(subs, on=image_column, how="inner")

    if test_col not in merged.columns:
        raise ValueError(f"test_col '{test_col}' not found in {subs_path}.")

    test_df = merged[merged[test_col] == 1].copy()
    keep_cols = [image_column] + [c for c in label_cols if c in merged.columns]
    return test_df[keep_cols].reset_index(drop=True), datasets_root


# ── DataLoader factory ─────────────────────────────────────────────────────────


def make_data_loader(
    df: pd.DataFrame,
    root_dir: Path,
    label_cols: list[str],
    *,
    is_train: bool,
    batch_size: int,
    num_workers: int,
    image_column: str,
    uncertain_policy: str,
    seed: int,
) -> DataLoader[Any]:
    """Wrap a DataFrame in a DatasetCXR and return a configured DataLoader.

    Returns
    -------
    DataLoader
        Yields ``(filename, image_tensor, (labels_tensor, mask_tensor))`` batches.
    """
    transform = train_tfm if is_train else val_tfm
    dataset = DatasetCXR(
        root_dir=root_dir,
        dataframe=df,
        label_cols=label_cols,
        transform=transform,
        image_column=image_column,
        output_mode="labels_mask",
        uncertain_policy=uncertain_policy,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        generator=generator,
        pin_memory=True,
    )
