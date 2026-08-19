"""
Evaluate a trained plant-disease model on the held-out TEST split.

Produces:
  1. A per-class precision / recall / F1 report (printed + saved to txt).
  2. A confusion-matrix heatmap.
  3. Grad-CAM overlays showing which leaf regions drove each prediction.

The test split is read from data/processed/test.csv (written during Step 2),
so the model is scored on images it never saw during training.

Usage:
    python -m src.evaluate
    python -m src.evaluate --model models/best_model.keras --gradcam-samples 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

from src.data_loader import build_dataset, PROCESSED_DIR, IMG_SIZE, BATCH_SIZE
from src.model import BACKBONES, get_backbone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def short(name: str) -> str:
    return name.replace("___", " /\n").replace("_", " ")


def load_test_df() -> pd.DataFrame:
    p = PROCESSED_DIR / "test.csv"
    if not p.exists():
        raise FileNotFoundError(
            "data/processed/test.csv not found. Run training or "
            "`python -m src.data_loader` first to create the splits."
        )
    return pd.read_csv(p)


def find_last_conv(base: tf.keras.Model) -> str:
    """Name of the last layer with a 4D (spatial) output — the Grad-CAM target."""
    for layer in reversed(base.layers):
        if len(layer.output.shape) == 4:
            return layer.name
    raise RuntimeError("No 4D convolutional layer found in the backbone.")


def build_gradcam_models(model, base, last_conv, preprocess):
    """Split the model into: raw image -> conv features, and conv -> predictions."""
    # raw [0,255] image -> preprocess -> backbone -> last conv feature map
    inp = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = preprocess(inp)
    conv_out = tf.keras.Model(base.input, base.get_layer(last_conv).output)(x)
    conv_model = tf.keras.Model(inp, conv_out)

    # conv feature map -> classification head -> probabilities
    ci = tf.keras.Input(shape=base.get_layer(last_conv).output.shape[1:])
    h = model.get_layer("gap")(ci)
    h = model.get_layer("dropout")(h)
    h = model.get_layer("predictions")(h)
    classifier_model = tf.keras.Model(ci, h)
    return conv_model, classifier_model


def gradcam_heatmap(img_batch, conv_model, classifier_model, pred_index=None):
    with tf.GradientTape() as tape:
        conv_out = conv_model(img_batch)
        tape.watch(conv_out)
        preds = classifier_model(conv_out)
        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))          # importance per channel
    heatmap = tf.squeeze(conv_out[0] @ pooled[..., None])   # weight the feature maps
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), pred_index


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(MODELS_DIR / "plant_disease_model.keras"))
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--gradcam-samples", type=int, default=6)
    args = ap.parse_args()

    OUTPUTS_DIR.mkdir(exist_ok=True)
    model = tf.keras.models.load_model(args.model)
    class_names = json.loads((MODELS_DIR / "class_names.json").read_text())
    names = [short(c) for c in class_names]

    backbone = model.name.replace("plant_disease_", "")
    preprocess = BACKBONES[backbone][1]
    base = get_backbone(model)
    last_conv = find_last_conv(base)

    # --- Predictions on the held-out test set (build_dataset does not shuffle) ---
    test_df = load_test_df()
    test_ds = build_dataset(test_df, training=False, batch_size=args.batch_size)
    probs = model.predict(test_ds, verbose=0)
    y_pred = probs.argmax(axis=1)
    y_true = test_df["label"].values

    # --- 1. Classification report ---
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred,
                                   target_names=[c.replace("\n", " ") for c in names],
                                   digits=3)
    print(f"\nTest accuracy: {acc:.4f}\n")
    print(report)
    (OUTPUTS_DIR / "classification_report.txt").write_text(
        f"Test accuracy: {acc:.4f}\n\n{report}\n")

    # --- 2. Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(1.6 * len(class_names), 1.4 * len(class_names)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names, cbar=False)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"Confusion Matrix (test acc = {acc:.1%})")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=90, bbox_inches="tight")
    plt.close()

    # --- 3. Grad-CAM overlays ---
    conv_model, classifier_model = build_gradcam_models(model, base, last_conv, preprocess)
    # one representative image per class (up to the requested count)
    sample_rows = (test_df.groupby("label").first().reset_index()
                   .head(args.gradcam_samples))

    n = len(sample_rows)
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5))
    for i, (_, row) in enumerate(sample_rows.iterrows()):
        raw = tf.keras.utils.load_img(row["path"], target_size=IMG_SIZE)
        arr = tf.keras.utils.img_to_array(raw)                 # [0,255]
        batch = np.expand_dims(arr, 0)
        heatmap, pidx = gradcam_heatmap(batch, conv_model, classifier_model)
        hm = tf.image.resize(heatmap[..., None], IMG_SIZE).numpy().squeeze()

        true_lbl = short(class_names[row["label"]]).replace("\n", " ")
        pred_lbl = short(class_names[pidx]).replace("\n", " ")
        ok = row["label"] == pidx

        axes[0][i].imshow(arr.astype("uint8")); axes[0][i].axis("off")
        axes[0][i].set_title(f"true: {true_lbl}", fontsize=7)
        axes[1][i].imshow(arr.astype("uint8"))
        axes[1][i].imshow(hm, cmap="jet", alpha=0.45); axes[1][i].axis("off")
        axes[1][i].set_title(f"pred: {pred_lbl} {'✓' if ok else '✗'}",
                             fontsize=7, color="green" if ok else "red")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "gradcam.png", dpi=90, bbox_inches="tight")
    plt.close()

    print(f"Saved report -> {OUTPUTS_DIR / 'classification_report.txt'}")
    print(f"Saved matrix -> {OUTPUTS_DIR / 'confusion_matrix.png'}")
    print(f"Saved Grad-CAM -> {OUTPUTS_DIR / 'gradcam.png'}")


if __name__ == "__main__":
    main()
