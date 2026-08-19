"""
Data pipeline for the plant-disease-detection project.

Responsibilities
----------------
1. Gather (image_path, label) pairs from data/raw/<Class>/<image>.
2. Split them into train/val/test with STRATIFICATION (each class keeps its
   proportion across the three splits) and save the splits as CSV manifests so
   the exact same test set can be reused during evaluation.
3. Build efficient tf.data pipelines (decode -> resize -> augment -> batch ->
   prefetch).
4. Compute class weights to counter class imbalance.

Design note
-----------
The datasets yield images resized to IMG_SIZE as float32 in the [0, 255] range.
Backbone-specific preprocessing (e.g. MobileNetV2 / ResNet `preprocess_input`)
is applied INSIDE the model in Step 3, so this module does not need to know
which backbone will be used.

Run as a script to build the splits and print a summary:
    python -m src.data_loader
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# --- Configuration ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

IMG_SIZE = (224, 224)          # input size expected by MobileNet/ResNet backbones
BATCH_SIZE = 32
SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
AUTOTUNE = tf.data.AUTOTUNE


# --- 1. Gather samples -----------------------------------------------------
def gather_samples(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, list[str]]:
    """Walk data/raw and return a DataFrame of (path, label, class_name)."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"{raw_dir} not found. Run: bash scripts/download_data.sh github-subset"
        )

    class_names = sorted(d.name for d in raw_dir.iterdir() if d.is_dir())
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    rows = []
    for c in class_names:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
            for p in (raw_dir / c).glob(ext):
                rows.append({"path": str(p), "class_name": c, "label": class_to_idx[c]})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No images found under {raw_dir}")
    return df, class_names


# --- 2. Stratified 3-way split --------------------------------------------
def make_splits(
    df: pd.DataFrame,
    val_frac: float = VAL_FRACTION,
    test_frac: float = TEST_FRACTION,
    seed: int = SEED,
    save: bool = True,
) -> dict[str, pd.DataFrame]:
    """Stratified split into train/val/test. Optionally save CSV manifests."""
    # First peel off the test set...
    train_val, test = train_test_split(
        df, test_size=test_frac, stratify=df["label"], random_state=seed
    )
    # ...then split the remainder into train / val (adjust val fraction).
    val_relative = val_frac / (1.0 - test_frac)
    train, val = train_test_split(
        train_val, test_size=val_relative, stratify=train_val["label"], random_state=seed
    )

    splits = {"train": train.reset_index(drop=True),
              "val": val.reset_index(drop=True),
              "test": test.reset_index(drop=True)}

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        for name, part in splits.items():
            part.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
    return splits


# --- 3. Augmentation -------------------------------------------------------
def build_augmenter() -> tf.keras.Sequential:
    """Geometric + photometric augmentation, applied to the training set only."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.15),
    ], name="augmentation")


# --- 4. tf.data pipeline ---------------------------------------------------
def _decode(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)          # [0, 255], backbone preprocessing later
    return img, label


def build_dataset(
    part: pd.DataFrame,
    training: bool,
    batch_size: int = BATCH_SIZE,
    augmenter: tf.keras.Sequential | None = None,
) -> tf.data.Dataset:
    """Turn a split DataFrame into a batched, prefetched tf.data.Dataset."""
    ds = tf.data.Dataset.from_tensor_slices(
        (part["path"].values, part["label"].values.astype("int32"))
    )
    if training:
        ds = ds.shuffle(buffer_size=min(len(part), 2000), seed=SEED,
                        reshuffle_each_iteration=True)
    ds = ds.map(_decode, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    if training and augmenter is not None:
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


# --- 5. Class weights ------------------------------------------------------
def compute_weights(train_df: pd.DataFrame, n_classes: int) -> dict[int, float]:
    """Inverse-frequency class weights for model.fit(class_weight=...)."""
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(n_classes),
        y=train_df["label"].values,
    )
    return {i: float(w) for i, w in enumerate(weights)}


# --- Convenience entry point ----------------------------------------------
def get_datasets(batch_size: int = BATCH_SIZE):
    """Build everything Step 3 needs: datasets, class names, and class weights."""
    df, class_names = gather_samples()
    splits = make_splits(df)
    augmenter = build_augmenter()

    train_ds = build_dataset(splits["train"], training=True,
                             batch_size=batch_size, augmenter=augmenter)
    val_ds = build_dataset(splits["val"], training=False, batch_size=batch_size)
    test_ds = build_dataset(splits["test"], training=False, batch_size=batch_size)

    class_weights = compute_weights(splits["train"], len(class_names))
    return train_ds, val_ds, test_ds, class_names, class_weights, splits


if __name__ == "__main__":
    df, class_names = gather_samples()
    splits = make_splits(df)

    print(f"Classes ({len(class_names)}): {class_names}\n")
    summary = pd.DataFrame({
        name: part["class_name"].value_counts() for name, part in splits.items()
    }).fillna(0).astype(int)
    summary["total"] = summary.sum(axis=1)
    print("Per-class split counts:")
    print(summary.to_string(), "\n")
    print("Split totals:", {k: len(v) for k, v in splits.items()})

    weights = compute_weights(splits["train"], len(class_names))
    print("\nClass weights (inverse frequency):")
    for i, c in enumerate(class_names):
        print(f"  {weights[i]:.3f}  {c}")
    print(f"\nManifests saved to {PROCESSED_DIR}/[train|val|test].csv")
