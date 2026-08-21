"""
Streamlit demo for the plant disease classifier.

Run locally:
    pip install -r requirements.txt
    python -m src.train          # trains models/plant_disease_model.keras first
    streamlit run app.py

For deployment (Streamlit Community Cloud), commit a trained
models/plant_disease_model.keras (it's ~24 MB, under GitHub's 100 MB limit) so
the app can load it — the CNN is too heavy to train on app startup.
"""
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌱", layout="centered")

MODELS_DIR = Path(__file__).parent / "models"
IMG_SIZE = (224, 224)


@st.cache_resource
def load_model():
    model_path = MODELS_DIR / "plant_disease_model.keras"
    names_path = MODELS_DIR / "class_names.json"
    if not model_path.exists() or not names_path.exists():
        return None, None
    model = tf.keras.models.load_model(model_path)
    class_names = json.loads(names_path.read_text())
    return model, class_names


def find_last_conv(base):
    for layer in reversed(base.layers):
        if len(layer.output.shape) == 4:
            return layer.name
    return None


def gradcam(model, arr, class_names):
    """Grad-CAM heatmap for the predicted class (handles the nested backbone)."""
    base = next(l for l in model.layers if isinstance(l, tf.keras.Model))
    backbone_name = model.name.replace("plant_disease_", "")
    preprocess = {
        "mobilenetv2": tf.keras.applications.mobilenet_v2.preprocess_input,
        "resnet50": tf.keras.applications.resnet50.preprocess_input,
        "efficientnetb0": tf.keras.applications.efficientnet.preprocess_input,
    }.get(backbone_name, lambda x: x)
    last_conv = find_last_conv(base)

    inp = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = preprocess(inp)
    conv_out = tf.keras.Model(base.input, base.get_layer(last_conv).output)(x)
    conv_model = tf.keras.Model(inp, conv_out)
    ci = tf.keras.Input(shape=base.get_layer(last_conv).output.shape[1:])
    h = model.get_layer("gap")(ci)
    h = model.get_layer("dropout")(h)
    h = model.get_layer("predictions")(h)
    classifier = tf.keras.Model(ci, h)

    batch = np.expand_dims(arr, 0)
    with tf.GradientTape() as tape:
        conv = conv_model(batch)
        tape.watch(conv)
        preds = classifier(conv)
        idx = int(tf.argmax(preds[0]))
        channel = preds[:, idx]
    grads = tape.gradient(channel, conv)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat = tf.squeeze(conv[0] @ pooled[..., None])
    heat = tf.maximum(heat, 0) / (tf.reduce_max(heat) + 1e-8)
    heat = tf.image.resize(heat[..., None], IMG_SIZE).numpy().squeeze()
    return heat, idx, preds.numpy()[0]


st.title("🌱 Plant Disease Detector")
st.caption("Upload a crop-leaf photo to classify its disease. CNN with transfer "
           "learning (MobileNetV2), with Grad-CAM showing where the model looks.")

model, class_names = load_model()
if model is None:
    st.warning("No trained model found at `models/plant_disease_model.keras`.\n\n"
               "Train it first with `python -m src.train`, then reload this app.")
    st.stop()

uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
show_cam = st.checkbox("Show Grad-CAM (where the model is looking)", value=True)

if uploaded:
    img = tf.keras.utils.load_img(uploaded, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)

    heat, idx, probs = gradcam(model, arr, class_names)
    label = class_names[idx].replace("___", " / ").replace("_", " ")

    c1, c2 = st.columns(2)
    with c1:
        st.image(img, caption="Uploaded leaf", use_container_width=True)
    with c2:
        if show_cam:
            import matplotlib.cm as cm
            overlay = (0.55 * arr / 255.0
                       + 0.45 * cm.jet(heat)[..., :3])
            st.image(np.clip(overlay, 0, 1), caption="Grad-CAM", use_container_width=True)

    st.divider()
    st.metric("Prediction", label, f"{probs[idx]:.0%} confidence")
    topk = np.argsort(probs)[::-1][:3]
    st.write("**Top predictions:**")
    for i in topk:
        st.write(f"- {class_names[i].replace('___', ' / ').replace('_', ' ')}: {probs[i]:.1%}")
