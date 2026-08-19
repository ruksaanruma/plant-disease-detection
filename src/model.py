"""
Transfer-learning model for plant disease classification.

The model bakes the backbone-specific `preprocess_input` in as its first layer,
so it accepts the plain [0, 255] float images produced by the data pipeline and
handles normalization internally. This keeps training and inference consistent
(no risk of forgetting to preprocess at prediction time).

Two-phase training is supported:
  Phase 1 - backbone frozen, train only the new classifier head.
  Phase 2 - unfreeze the top layers (BatchNorm kept frozen) and fine-tune with
            a very low learning rate.
"""
from __future__ import annotations

import tensorflow as tf

# backbone name -> (constructor, preprocess_input). All expect [0,255] inputs.
BACKBONES = {
    "mobilenetv2": (
        tf.keras.applications.MobileNetV2,
        tf.keras.applications.mobilenet_v2.preprocess_input,
    ),
    "resnet50": (
        tf.keras.applications.ResNet50,
        tf.keras.applications.resnet50.preprocess_input,
    ),
    "efficientnetb0": (
        tf.keras.applications.EfficientNetB0,
        tf.keras.applications.efficientnet.preprocess_input,
    ),
}


def build_model(
    num_classes: int,
    img_size: tuple[int, int] = (224, 224),
    backbone: str = "mobilenetv2",
    dropout: float = 0.2,
) -> tf.keras.Model:
    """Build a transfer-learning classifier with a frozen backbone."""
    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone '{backbone}'. Options: {list(BACKBONES)}")
    app_fn, preprocess = BACKBONES[backbone]

    base = app_fn(include_top=False, weights="imagenet", input_shape=img_size + (3,))
    base.trainable = False  # Phase 1: frozen

    inputs = tf.keras.Input(shape=img_size + (3,), name="image")
    x = preprocess(inputs)
    # training=False keeps the backbone's BatchNorm in inference mode, using the
    # pretrained running statistics rather than our small batch statistics.
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return tf.keras.Model(inputs, outputs, name=f"plant_disease_{backbone}")


def get_backbone(model: tf.keras.Model) -> tf.keras.Model:
    """Return the nested pretrained backbone inside the model."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise RuntimeError("No nested backbone Model found.")


def enable_fine_tuning(model: tf.keras.Model, unfreeze_fraction: float = 0.3) -> tf.keras.Model:
    """
    Unfreeze the top `unfreeze_fraction` of backbone layers for fine-tuning,
    while keeping all BatchNormalization layers frozen (recommended when
    fine-tuning on a small dataset).
    """
    base = get_backbone(model)
    base.trainable = True
    n = len(base.layers)
    cutoff = int(n * (1.0 - unfreeze_fraction))
    for i, layer in enumerate(base.layers):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = i >= cutoff

    trainable = sum(1 for l in base.layers if l.trainable)
    print(f"Fine-tuning: {trainable}/{n} backbone layers unfrozen (BatchNorm kept frozen)")
    return base
