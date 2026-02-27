import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.model_builder import model as build_model

st.set_page_config(
    page_title="Kidney Fibrosis Grader",
    page_icon="🔬",
    layout="centered"
)

N_CLASS    = 4
MODE       = 1
IMG_SIZE   = 508
DEVICE     = "cpu"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "global_only.pth")

CLASS_NAMES  = [
    "Minimal  — < 10% fibrosis",
    "Mild     — 10–25% fibrosis",
    "Moderate — 25–50% fibrosis",
    "Severe   — > 50% fibrosis",
]
CLASS_COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
CLASS_ICONS  = ["🟢", "🟡", "🟠", "🔴"]

@st.cache_resource
def load_model():
    net, _ = build_model(N_CLASS, mode=MODE, evaluation=True, path_g=MODEL_PATH)
    net.eval()
    return net

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    
])

def predict(img):
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    net = load_model()
    # mode 1 forward: (image_global, patches, top_lefts, ratio)
    # patches/top_lefts/ratio are unused in mode 1 but required as arguments
    dummy_patches   = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    dummy_top_lefts = [(0, 0)]
    dummy_ratio     = (1.0, 1.0)
    with torch.no_grad():
        output, _ = net.module.forward(tensor, dummy_patches, dummy_top_lefts, dummy_ratio, mode=1)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
    print(f"DEBUG probs: {probs}")
    return probs

st.title("🔬 Kidney Fibrosis Grader")
st.markdown("Upload a **trichrome-stained kidney biopsy image** to get an automated fibrosis grade.")

with st.expander("ℹ️ About this tool"):
    st.markdown("""
    Deep learning model (ResNet-FPN) trained on trichrome-stained kidney biopsy images.

    | Grade | Fibrosis Level |
    |-------|---------------|
    | 🟢 Minimal | < 10% |
    | 🟡 Mild | 10–25% |
    | 🟠 Moderate | 25–50% |
    | 🔴 Severe | > 50% |

    **Test accuracy: 95%** on held-out biopsy images.
    > ⚠️ Research use only. Not validated for clinical diagnosis.
    """)

st.divider()

uploaded = st.file_uploader("Upload biopsy image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)

    with col2:
        with st.spinner("Analyzing..."):
            try:
                probs = predict(img)
                pred  = int(np.argmax(probs))

                st.markdown("### Result")
                st.markdown(
                    f"<div style='background-color:{CLASS_COLORS[pred]};padding:16px;"
                    f"border-radius:10px;text-align:center;font-size:20px;"
                    f"font-weight:bold;color:white'>"
                    f"{CLASS_ICONS[pred]} {CLASS_NAMES[pred]}</div>",
                    unsafe_allow_html=True
                )
                st.markdown(f"**Confidence: {probs[pred]*100:.1f}%**")
                st.markdown("#### All probabilities")
                for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
                    st.markdown(f"{CLASS_ICONS[i]} {name}")
                    st.progress(float(prob), text=f"{prob*100:.1f}%")

            except Exception as e:
                st.error(f"Error: {str(e)}")

st.divider()
st.caption("Research use only · Not for clinical diagnosis · ResNet-FPN · 95% test accuracy")
