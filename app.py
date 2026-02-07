import streamlit as st
import numpy as np
import tensorflow as tf
import requests
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
# PAGE CONFIG
st.set_page_config(
    page_title="TumorSense-AI | Advanced MRI Diagnostics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# ADVANCED UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;700&display=swap');

    /* Global Background */
    [data-testid="stAppViewContainer"] {
        background-color: #05070a;
        background-image: 
            radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.05) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(129, 140, 248, 0.05) 0px, transparent 50%);
    }

    /* Main Container */
    .main-header {
        text-align: center;
        padding: 20px;
        margin-bottom: 30px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }

    .title-text {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(180deg, #ffffff 0%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 48px;
        font-weight: 700;
        letter-spacing: 4px;
        margin-bottom: 0px;
    }

    /* Command Center Cards */
    .system-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 0 20px rgba(0,0,0,0.4);
        transition: all 0.3s ease;
    }
    
    .system-card:hover {
        border: 1px solid rgba(56, 189, 248, 0.5);
        box-shadow: 0 0 30px rgba(56, 189, 248, 0.1);
    }

    /* Result Styling */
    .status-active {
        color: #22c55e;
        font-family: 'Orbitron', sans-serif;
        font-size: 12px;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }

    .prediction-output {
        font-family: 'Orbitron', sans-serif;
        font-size: 42px;
        font-weight: 700;
        color: #ffffff;
        text-shadow: 0 0 15px rgba(255,255,255,0.3);
        margin: 10px 0;
    }
    /* Progress Bars */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #38bdf8 , #818cf8);
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background: transparent;
        border: 1px solid #38bdf8;
        color: #38bdf8;
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 1px;
        padding: 15px;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background: #38bdf8;
        color: #000;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
    }

    /* Disclaimer Section */
    .footer-warning {
        font-size: 11px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-top: 1px solid rgba(255,255,255,0.05);
        padding-top: 20px;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)
# MODEL + CONSTANTS
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

MODEL_URL = "https://huggingface.co/itz-sudha/brain-tumor-ai/resolve/main/brain_tumor_final.keras"
MODEL_PATH = "brain_tumor_final.keras"

@st.cache_resource
def load_trained_model():
    # Download model only if not present
    if not os.path.exists(MODEL_PATH):
            response = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

    # Load model
    model = load_model(MODEL_PATH)
    return model

model = load_trained_model()
#  PREPROCESSING 
def preprocess_image(img):
    img = img.resize((224, 224))
    arr = image.img_to_array(img)
    arr = arr / 255.0            # SAME AS TRAINING
    arr = np.expand_dims(arr, axis=0)
    return arr

# STABLE GRAD-CAM (MEDICAL-GRADE)
def make_gradcam_heatmap(img_array, model, layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array) 

        # unwrap list output safely
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        class_idx = tf.argmax(preds[0])
        loss = preds[0, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        raise RuntimeError("Grad-CAM failed: gradients are None")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()
def overlay_heatmap(heatmap, original_img, alpha=0.45):
    """
    Overlay Grad-CAM heatmap on original image
    """

    # Convert heatmap to uint8
    heatmap = np.uint8(255 * heatmap)

    # Apply color map
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Resize heatmap to match original image
    heatmap = cv2.resize(
        heatmap,
        (original_img.shape[1], original_img.shape[0])
    )

    # Ensure original image is uint8
    if original_img.max() <= 1:
        original_img = np.uint8(255 * original_img)

    # Overlay
    overlayed = cv2.addWeighted(
        original_img, 1 - alpha,
        heatmap, alpha,
        0
    )

    return overlayed

# UI
st.markdown("""
<div class='main-header'>
    <div class='title-text'>TumorSense ֎AI - 𖣂</div>
    <p style='color: #38bdf8; letter-spacing: 5px; font-size: 12px;'>ADVANCED MRI DIAGNOSTICS</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1.5])

with left:
    st.markdown("<div class='system-card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload MRI Scan 🧠", type=["jpg","png","jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, use_container_width=True)
    analyze = st.button("RUN DEEP ANALYSIS 🔬")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='system-card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Report Analysis")
    if uploaded_file and analyze:
     with st.spinner("Preparing-report(🧬)..."):
        processed = preprocess_image(img)
        preds = model.predict(processed)
        idx = np.argmax(preds[0])
        confidence = preds[0][idx] * 100
        st.markdown("<div class='status-active'>● ANALYSIS COMPLETE ⌛</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='prediction-output'>{class_names[idx].upper()}</div>", unsafe_allow_html=True)
        st.progress(int(confidence))

        cols = st.columns(4)
        for i, cls in enumerate(class_names):
            cols[i].metric(cls.capitalize(), f"{preds[0][i]*100:.2f}%")

        heatmap = make_gradcam_heatmap(processed, model, "conv5_block3_out")
        original = np.array(img.resize((224,224)))
        gradcam = overlay_heatmap(heatmap, original)

        c1, c2 = st.columns(2)
        c1.image(original, caption="Original MRI", use_container_width=True)
        c2.image(gradcam, caption="Grad-CAM Focus", use_container_width=True)

    else:
        st.info("System awaiting input. Please upload a brain MRI scan to initiate the analysis sequence.")
    st.markdown("""
    <div class='footer-warning'>
        [!] ACADEMIC RESEARCH PROTOCOL: This system is for research visualization only. 
        Final clinical decisions must be performed by certified radiologists.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)