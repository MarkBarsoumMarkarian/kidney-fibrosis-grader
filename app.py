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
    page_title="KidneyGrade · Fibrosis Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0e14 !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] > .main {
    background-color: #0a0e14 !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Hero Header ── */
.hero {
    background: linear-gradient(135deg, #0d1520 0%, #0a1628 50%, #060d16 100%);
    border-bottom: 1px solid rgba(56, 139, 253, 0.15);
    padding: 48px 64px 40px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -120px; right: -120px;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(56, 139, 253, 0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 30%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0, 210, 190, 0.05) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #388bfd;
    margin-bottom: 12px;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 42px;
    line-height: 1.1;
    color: #f0f4ff;
    margin: 0 0 12px;
    font-weight: 400;
}
.hero-title em {
    font-style: italic;
    color: #388bfd;
}
.hero-sub {
    font-size: 15px;
    color: #7d8590;
    font-weight: 300;
    max-width: 540px;
    line-height: 1.6;
}
.hero-badges {
    display: flex;
    gap: 10px;
    margin-top: 24px;
    flex-wrap: wrap;
}
.badge {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    padding: 5px 12px;
    border-radius: 4px;
    letter-spacing: 0.05em;
}
.badge-blue  { background: rgba(56,139,253,0.12); color: #388bfd; border: 1px solid rgba(56,139,253,0.25); }
.badge-teal  { background: rgba(0,210,190,0.10);  color: #00d2be; border: 1px solid rgba(0,210,190,0.20); }
.badge-amber { background: rgba(255,176,0,0.10);  color: #ffb000; border: 1px solid rgba(255,176,0,0.20); }

/* ── Main Layout ── */
.main-content {
    padding: 40px 64px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 32px;
    max-width: 1400px;
    margin: 0 auto;
}

/* ── Cards ── */
.card {
    background: #0d1520;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 28px;
    position: relative;
}
.card-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #7d8590;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.card-label::before {
    content: '';
    display: inline-block;
    width: 20px; height: 1px;
    background: #388bfd;
}

/* ── Upload Zone ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
}
[data-testid="stFileUploadDropzone"] {
    background: rgba(56,139,253,0.04) !important;
    border: 1.5px dashed rgba(56,139,253,0.25) !important;
    border-radius: 10px !important;
    padding: 40px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    background: rgba(56,139,253,0.08) !important;
    border-color: rgba(56,139,253,0.5) !important;
}
[data-testid="stFileUploadDropzone"] p {
    color: #7d8590 !important;
    font-size: 14px !important;
}
[data-testid="stFileUploadDropzone"] small {
    color: #4a5260 !important;
}

/* ── Result Grade Box ── */
.grade-box {
    border-radius: 10px;
    padding: 20px 24px;
    margin: 16px 0;
    display: flex;
    align-items: center;
    gap: 16px;
    position: relative;
    overflow: hidden;
}
.grade-box::before {
    content: '';
    position: absolute;
    inset: 0;
    opacity: 0.06;
    background: currentColor;
}
.grade-icon { font-size: 28px; line-height: 1; }
.grade-info { flex: 1; }
.grade-name {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    font-weight: 400;
    line-height: 1.2;
}
.grade-pct {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    opacity: 0.7;
    margin-top: 2px;
}
.grade-conf {
    font-family: 'DM Mono', monospace;
    font-size: 24px;
    font-weight: 500;
    text-align: right;
}
.grade-conf small {
    font-size: 11px;
    opacity: 0.6;
    display: block;
    text-align: right;
}

/* ── Probability Bars ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 10px 0;
}
.prob-label {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #7d8590;
    width: 90px;
    flex-shrink: 0;
}
.prob-track {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 3px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.6s cubic-bezier(0.16,1,0.3,1);
}
.prob-value {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #7d8590;
    width: 40px;
    text-align: right;
    flex-shrink: 0;
}

/* ── AI Interpretation ── */
.ai-section {
    padding: 40px 64px;
    max-width: 1400px;
    margin: 0 auto;
}
.ai-card {
    background: #0d1520;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 36px 40px;
    position: relative;
    overflow: hidden;
}
.ai-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #388bfd, #00d2be, transparent);
}
.ai-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 28px;
}
.ai-title {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    font-weight: 400;
    color: #f0f4ff;
}
.ai-model-tag {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.12em;
    color: #388bfd;
    background: rgba(56,139,253,0.1);
    border: 1px solid rgba(56,139,253,0.2);
    padding: 4px 10px;
    border-radius: 4px;
}
.ai-content {
    color: #c9d1d9;
    font-size: 14.5px;
    line-height: 1.8;
    font-weight: 300;
}
.ai-content h3, .ai-content strong, .ai-content b {
    color: #e8eaf0 !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    letter-spacing: 0.03em !important;
}
.ai-content p { margin: 0 0 16px; }
.ai-content ol, .ai-content ul {
    padding-left: 20px;
    margin: 0 0 16px;
}

/* ── Divider ── */
.section-divider {
    height: 1px;
    background: rgba(255,255,255,0.05);
    margin: 0 64px;
}

/* ── Footer ── */
.footer {
    padding: 24px 64px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin-top: 40px;
}
.footer-left {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #4a5260;
    letter-spacing: 0.05em;
}
.footer-warning {
    font-size: 11px;
    color: #4a5260;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── Streamlit overrides ── */
[data-testid="stImage"] img {
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
}
[data-testid="stSpinner"] > div { color: #388bfd !important; }
.stSpinner > div > div { border-top-color: #388bfd !important; }

/* Hide default Streamlit elements */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    color: #7d8590 !important;
    font-size: 13px !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e14; }
::-webkit-scrollbar-thumb { background: #1e2a3a; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
N_CLASS    = 4
MODE       = 1
IMG_SIZE   = 508
DEVICE     = "cpu"

CLASS_NAMES  = ["Minimal", "Mild", "Moderate", "Severe"]
CLASS_RANGE  = ["< 10% fibrosis", "10–25% fibrosis", "25–50% fibrosis", "> 50% fibrosis"]
CLASS_COLORS = ["#00d2be", "#ffb000", "#ff7b29", "#ff4757"]
CLASS_ICONS  = ["●", "●", "●", "●"]
CLASS_SHORT  = ["Minimal (<10%)", "Mild (10–25%)", "Moderate (25–50%)", "Severe (>50%)"]
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ── Model ────────────────────────────────────────────────────────────────────
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
    return probs

def get_ai_interpretation(grade_label, confidence, all_probs):
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        try:
            groq_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
    if not groq_key:
        raise ValueError("GROQ_API_KEY not configured.")

    prob_breakdown = "\n".join(
        f"  - {CLASS_SHORT[i]}: {all_probs[i]*100:.1f}%"
        for i in range(4)
    )
    prompt = f"""You are a nephropathology AI assistant. A deep learning model analyzed a trichrome-stained kidney biopsy image:

Predicted Grade: {grade_label}
Confidence: {confidence:.1f}%
Probability breakdown:
{prob_breakdown}

Provide a concise clinical interpretation with exactly these 4 sections:

1. **End-Stage Kidney Disease (ESKD) Assessment**
   Is this grade associated with ESKD, or is it a risk? Be specific.

2. **Progression Risk**
   How likely is this to worsen? What factors drive progression at this level?

3. **Clinical Recommendations**
   What next steps would a nephrologist consider? (monitoring, interventions, referrals)

4. **Plain-Language Summary**
   Explain in simple language suitable for a patient.

Keep each section to 3–5 sentences. End with a one-line disclaimer that this is AI-generated and not a substitute for clinical judgment."""

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

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Nephropathology · AI-Assisted Analysis</div>
    <div class="hero-title">Kidney Fibrosis <em>Grader</em></div>
    <div class="hero-sub">
        Upload a trichrome-stained biopsy image for automated fibrosis grading
        and AI-powered clinical interpretation.
    </div>
    <div class="hero-badges">
        <span class="badge badge-blue">ResNet-FPN Architecture</span>
        <span class="badge badge-teal">95% Test Accuracy</span>
        <span class="badge badge-amber">Research Use Only</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Main Columns ─────────────────────────────────────────────────────────────
st.markdown('<div style="padding: 40px 64px 0;">', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="card-label">Upload Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop a trichrome-stained kidney biopsy image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_column_width=True)

    with st.expander("Grade Reference Guide"):
        st.markdown("""
| Grade | Range | Interpretation |
|-------|-------|---------------|
| 🟢 Minimal | < 10% | Early / no significant fibrosis |
| 🟡 Mild | 10–25% | Mild interstitial fibrosis |
| 🟠 Moderate | 25–50% | Significant fibrosis present |
| 🔴 Severe | > 50% | Extensive fibrosis / high ESKD risk |
        """)

with col2:
    if uploaded:
        st.markdown('<div class="card-label">Analysis Result</div>', unsafe_allow_html=True)
        with st.spinner("Running inference..."):
            try:
                probs = predict(img)
                pred  = int(np.argmax(probs))
                color = CLASS_COLORS[pred]

                # Grade result box
                st.markdown(f"""
<div class="grade-box" style="color:{color}; border: 1px solid {color}33; background: {color}0d;">
    <div class="grade-info">
        <div class="grade-name">{CLASS_NAMES[pred]}</div>
        <div class="grade-pct">{CLASS_RANGE[pred]}</div>
    </div>
    <div>
        <div class="grade-conf">{probs[pred]*100:.1f}<small>% confidence</small></div>
    </div>
</div>
""", unsafe_allow_html=True)

                # Probability breakdown
                st.markdown('<div class="card-label" style="margin-top:24px;">Probability Distribution</div>', unsafe_allow_html=True)
                for i in range(4):
                    pct = probs[i] * 100
                    st.markdown(f"""
<div class="prob-row">
    <div class="prob-label">{CLASS_NAMES[i]}</div>
    <div class="prob-track">
        <div class="prob-fill" style="width:{pct:.1f}%; background:{CLASS_COLORS[i]};"></div>
    </div>
    <div class="prob-value">{pct:.1f}%</div>
</div>
""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Inference error: {str(e)}")
                st.stop()
    else:
        st.markdown("""
<div style="height:320px; display:flex; flex-direction:column; align-items:center;
     justify-content:center; color:#4a5260; text-align:center; gap:12px;">
    <div style="font-size:48px; opacity:0.3;">🔬</div>
    <div style="font-size:13px; font-family:'DM Mono',monospace; letter-spacing:0.05em;">
        AWAITING IMAGE UPLOAD
    </div>
    <div style="font-size:12px; max-width:240px; line-height:1.6; opacity:0.7;">
        Upload a trichrome-stained biopsy image on the left to begin analysis
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── AI Interpretation ─────────────────────────────────────────────────────────
if uploaded and 'probs' in dir() and 'pred' in dir():
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="ai-section">', unsafe_allow_html=True)
    st.markdown("""
<div class="ai-card">
    <div class="ai-header">
        <div class="ai-title">Clinical Interpretation</div>
        <div class="ai-model-tag">LLAMA 3.3 · 70B · GROQ</div>
    </div>
""", unsafe_allow_html=True)

    with st.spinner("Generating clinical interpretation..."):
        try:
            interpretation = get_ai_interpretation(
                grade_label=f"{CLASS_NAMES[pred]} — {CLASS_RANGE[pred]}",
                confidence=probs[pred] * 100,
                all_probs=probs.tolist()
            )
            st.markdown(f'<div class="ai-content">', unsafe_allow_html=True)
            st.markdown(interpretation)
            st.markdown('</div>', unsafe_allow_html=True)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                st.error("Groq API key missing or invalid. Add GROQ_API_KEY to your Streamlit secrets.")
            elif e.response.status_code == 429:
                st.warning("Rate limit reached. Please wait a moment and try again.")
            else:
                st.error(f"AI interpretation unavailable: {str(e)}")
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"AI interpretation unavailable: {str(e)}")

    st.markdown('</div></div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-left">KIDNEYGRADE · v2.0 · ResNet-FPN</div>
    <div class="footer-warning">
        ⚠ Research use only — not validated for clinical diagnosis
    </div>
</div>
""", unsafe_allow_html=True)
