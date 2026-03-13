import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import sys, os
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.model_builder import model as build_model
import gdown

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'global_only.pth')
if not os.path.exists(MODEL_PATH):
    print('Downloading model weights...')
    gdown.download('https://drive.google.com/uc?id=1KvJQ0YKL-I96UJ5zUGLR_Qpd4R0ach5t', MODEL_PATH, quiet=False)
    print('Done.')

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
CLASS_SHORT  = ["Minimal (<10%)", "Mild (10–25%)", "Moderate (25–50%)", "Severe (>50%)"]

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

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
    dummy_patches   = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    dummy_top_lefts = [(0, 0)]
    dummy_ratio     = (1.0, 1.0)
    with torch.no_grad():
        output, _ = net.module.forward(tensor, dummy_patches, dummy_top_lefts, dummy_ratio, mode=1)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
    print(f"DEBUG probs: {probs}")
    return probs


def get_ai_interpretation(grade_label: str, confidence: float, all_probs: list) -> str:
    """Call Llama-3.3-70B via Groq free API to interpret the fibrosis grade."""

    # Load token from environment or Streamlit secrets
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        try:
            groq_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass

    if not groq_key:
        raise ValueError("GROQ_API_KEY not found. Add it to your Streamlit secrets.")

    prob_breakdown = "\n".join(
        f"  - {CLASS_SHORT[i]}: {all_probs[i]*100:.1f}%"
        for i in range(4)
    )

    prompt = f"""You are a nephropathology AI assistant. A deep learning model has analyzed a 
trichrome-stained kidney biopsy image and produced the following fibrosis grading result:

Predicted Grade: {grade_label}
Confidence: {confidence:.1f}%

Full probability breakdown:
{prob_breakdown}

Based on this fibrosis grade, provide a concise clinical interpretation covering exactly these 4 sections:

1. **End-Stage Kidney Disease (ESKD) Assessment**
   Is this grade associated with end-stage kidney disease, or is ESKD a risk? Be specific about what this grade means in that context.

2. **Progression Risk**
   How likely is this to worsen over time? What factors typically drive progression at this fibrosis level?

3. **Clinical Recommendations**
   What are the typical next steps a nephrologist would consider at this fibrosis stage? (e.g. monitoring frequency, interventions, referrals)

4. **Plain-Language Summary**
   Explain the result in simple, clear language suitable for a patient with no medical background.

Keep each section focused and concise (3–5 sentences). End with a one-line disclaimer that this is AI-generated and not a substitute for clinical judgment."""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {groq_key}",
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.3,
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ── UI ──────────────────────────────────────────────────────────────────────

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
                st.stop()

    # ── AI Interpretation ────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🤖 AI Clinical Interpretation")
    st.caption("Powered by Llama 3.3-70B via Groq")

    with st.spinner("Generating clinical interpretation..."):
        try:
            interpretation = get_ai_interpretation(
                grade_label=CLASS_NAMES[pred],
                confidence=probs[pred] * 100,
                all_probs=probs.tolist()
            )
            st.markdown(interpretation)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                st.error("Groq API key missing or invalid. Add GROQ_API_KEY to your Streamlit secrets.")
            elif e.response.status_code == 429:
                st.warning("Groq rate limit reached. Please wait a moment and try again.")
            else:
                st.error(f"AI interpretation unavailable: {str(e)}")
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"AI interpretation unavailable: {str(e)}")

st.divider()
st.caption("Research use only · Not for clinical diagnosis · ResNet-FPN · 95% test accuracy")
