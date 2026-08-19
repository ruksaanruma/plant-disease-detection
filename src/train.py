"""
Two-phase transfer-learning training for plant disease classification.

Phase 1: train the classifier head with the backbone frozen.
Phase 2: unfreeze the top backbone layers and fine-tune at a low learning rate.

Examples
--------
Full training on all classes:
    python -m src.train --epochs-head 12 --epochs-finetune 12

Fast smoke test on a small subset:
    python -m src.train --max-per-class 80 --epochs-head 2 --epochs-finetune 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from src.data_loader import (
    gather_samples, make_splits, build_dataset, build_augmenter,
    compute_weights, IMG_SIZE, BATCH_SIZE,
)
from src.model import build_model, enable_fine_tuning

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def subsample(df: pd.DataFrame, max_per_class: int | None, seed: int = 42) -> pd.DataFrame:
    """Cap the number of samples per class (for quick smoke tests)."""
    if not max_per_class:
        return df
    parts = [
        g.sample(min(len(g), max_per_class), random_state=seed)
        for _, g in df.groupby("label")
    ]
    return pd.concat(parts).reset_index(drop=True)


def make_callbacks(tag: str) -> list[tf.keras.callbacks.Callback]:
    """Fresh callbacks per phase (avoids state leaking across fit calls)."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / "best_model.keras"),
            monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.CSVLogger(str(OUTPUTS_DIR / f"training_log_{tag}.csv")),
    ]


def plot_history(h1, h2, path: Path) -> None:
    """Plot accuracy and loss across both phases, marking the phase boundary."""
    def merge(key):
        return h1.history.get(key, []) + h2.history.get(key, [])

    acc, val_acc = merge("accuracy"), merge("val_accuracy")
    loss, val_loss = merge("loss"), merge("val_loss")
    boundary = len(h1.history.get("accuracy", []))
    epochs = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, acc, label="train")
    ax1.plot(epochs, val_acc, label="val")
    ax1.axvline(boundary + 0.5, ls="--", c="gray", label="fine-tune start")
    ax1.set_title("Accuracy"); ax1.set_xlabel("epoch"); ax1.legend()
    ax2.plot(epochs, loss, label="train")
    ax2.plot(epochs, val_loss, label="val")
    ax2.axvline(boundary + 0.5, ls="--", c="gray")
    ax2.set_title("Loss"); ax2.set_xlabel("epoch"); ax2.legend()
    plt.tight_layout(); plt.savefig(path, dpi=90, bbox_inches="tight"); plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="mobilenetv2")
    ap.add_argument("--epochs-head", type=int, default=12)
    ap.add_argument("--epochs-finetune", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--unfreeze-fraction", type=float, default=0.3)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-finetune", type=float, default=1e-5)
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="Cap samples per class (for quick smoke tests).")
    args = ap.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # --- Data ---
    df, class_names = gather_samples()
    splits = make_splits(df)
    train_df = subsample(splits["train"], args.max_per_class)
    val_df = subsample(splits["val"], args.max_per_class)

    augmenter = build_augmenter()
    train_ds = build_dataset(train_df, training=True,
                             batch_size=args.batch_size, augmenter=augmenter)
    val_ds = build_dataset(val_df, training=False, batch_size=args.batch_size)
    class_weights = compute_weights(train_df, len(class_names))

    print(f"Classes: {len(class_names)} | train={len(train_df)} val={len(val_df)} "
          f"| backbone={args.backbone}")

    # --- Model ---
    model = build_model(len(class_names), IMG_SIZE, args.backbone)

    # --- Phase 1: frozen backbone, train head ---
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr_head),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print("\n=== Phase 1: training classifier head (backbone frozen) ===")
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_head,
                   class_weight=class_weights, callbacks=make_callbacks("head"),
                   shuffle=False)  # dataset is already shuffled in the pipeline

    # --- Phase 2: unfreeze top, fine-tune ---
    enable_fine_tuning(model, args.unfreeze_fraction)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr_finetune),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print("\n=== Phase 2: fine-tuning top layers (low LR) ===")
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_finetune,
                   class_weight=class_weights, callbacks=make_callbacks("finetune"),
                   shuffle=False)

    # --- Save artifacts ---
    model.save(MODELS_DIR / "plant_disease_model.keras")
    (MODELS_DIR / "class_names.json").write_text(json.dumps(class_names, indent=2))
    plot_history(h1, h2, OUTPUTS_DIR / "training_curves.png")

    best_val = max(h1.history["val_accuracy"] + h2.history["val_accuracy"])
    print(f"\nBest val accuracy: {best_val:.4f}")
    print(f"Saved model      -> {MODELS_DIR / 'plant_disease_model.keras'}")
    print(f"Saved classes    -> {MODELS_DIR / 'class_names.json'}")
    print(f"Saved curves     -> {OUTPUTS_DIR / 'training_curves.png'}")


if __name__ == "__main__":
    main()
