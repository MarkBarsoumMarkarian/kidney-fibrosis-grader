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
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&family=Playfair+Display:wght@600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

/* ── Base background: warm dark slate ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
section.main,
[data-testid="stMain"] {
    background: #1c2333 !important;
    color: #d0d6e0 !important;
    font-family: 'Inter', sans-serif !important;
}

.block-container {
    padding: 0 32px 32px 32px !important;
    max-width: 100% !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
#MainMenu, footer, header {
    display: none !important;
    visibility: hidden !important;
}

/* ── TOP BAR ── */
.topbar {
    background: #141b2d;
    border-bottom: 1px solid #2a3349;
    padding: 0 32px;
    height: 54px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 0 -32px 24px -32px;
}
.topbar-brand { display: flex; align-items: center; gap: 12px; }
.topbar-logo {
    width: 30px; height: 30px;
    background: #2563eb;
    border-radius: 7px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    color: #fff;
    display: flex; align-items: center; justify-content: center;
    letter-spacing: -0.5px;
}
.topbar-name {
    font-family: 'Playfair Display', serif;
    font-size: 16px;
    font-weight: 700;
    color: #e8edf5;
}
.topbar-desc {
    font-size: 10px;
    color: #5a6480;
    margin-top: 1px;
    letter-spacing: 0.02em;
}
.topbar-pills { display: flex; gap: 8px; }
.tpill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    padding: 4px 10px;
    border-radius: 4px;
    letter-spacing: 0.04em;
    border: 1px solid;
}
.tpill-blue  { color: #7eb3ff; border-color: #2a4a80; background: #1a2d4a; }
.tpill-green { color: #6ee7b7; border-color: #1a4a35; background: #102a20; }
.tpill-amber { color: #fbbf24; border-color: #4a3510; background: #2a1e08; }

/* ── SECTION LABEL ── */
.sec-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4a5470;
    margin-bottom: 10px;
    margin-top: 2px;
}

/* ── CARD ── */
.card {
    background: #202b3d;
    border: 1px solid #2a3549;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 16px;
}

/* ── UPLOAD ZONE ── */
[data-testid="stFileUploader"] { background: transparent !important; }
[data-testid="stFileUploadDropzone"] {
    background: #1a2435 !important;
    border: 2px dashed #2e3f5c !important;
    border-radius: 10px !important;
    padding: 28px !important;
    transition: all 0.2s !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #2563eb !important;
    background: #1a2845 !important;
}
[data-testid="stFileUploadDropzone"] p {
    color: #5a6480 !important;
    font-size: 13px !important;
}
[data-testid="stFileUploadDropzone"] small { color: #3a4460 !important; }
[data-testid="stFileUploadDropzone"] svg { fill: #2e3f5c !important; }

/* ── IMAGE ── */
[data-testid="stImage"] img {
    border-radius: 8px !important;
    border: 1px solid #2a3549 !important;
    width: 100% !important;
}

/* ── GRADE CARD ── */
.grade-card {
    border-radius: 10px;
    padding: 18px 20px;
    border: 1px solid;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.grade-name {
    font-family: 'Playfair Display', serif;
    font-size: 24px;
    font-weight: 700;
    line-height: 1.1;
}
.grade-range {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #7a8490;
    margin-top: 4px;
}
.grade-pct {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 30px;
    font-weight: 500;
    text-align: right;
    line-height: 1;
}
.grade-conf-lbl {
    font-size: 10px;
    color: #5a6480;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    text-align: right;
    margin-top: 3px;
}

/* ── PROB BARS ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 11px;
}
.prob-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #5a6480;
    width: 62px;
    flex-shrink: 0;
    font-weight: 500;
}
.prob-track {
    flex: 1;
    height: 6px;
    background: #1a2435;
    border-radius: 3px;
    overflow: hidden;
}
.prob-fill { height: 100%; border-radius: 3px; }
.prob-pct {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #5a6480;
    width: 38px;
    text-align: right;
    flex-shrink: 0;
}

/* ── GRADE REF ── */
.ref-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 0;
    border-bottom: 1px solid #242f42;
    font-size: 12px;
    color: #8a9ab0;
}
.ref-row:last-child { border-bottom: none; }
.ref-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.ref-grade { font-weight: 600; color: #b0bac8; width: 64px; flex-shrink: 0; }
.ref-range {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #4a5470;
    width: 60px;
    flex-shrink: 0;
}
.ref-desc { color: #5a6880; font-size: 11px; }

/* ── AI SECTION ── */
.ai-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 18px;
    padding-bottom: 14px;
    border-bottom: 1px solid #2a3549;
}
.ai-title {
    font-family: 'Playfair Display', serif;
    font-size: 18px;
    font-weight: 700;
    color: #e0e6f0;
}
.ai-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    font-weight: 500;
    letter-spacing: 0.08em;
    background: #1a2845;
    color: #7eb3ff;
    border: 1px solid #2a4a80;
    padding: 4px 10px;
    border-radius: 4px;
}
.ai-body {
    font-size: 13.5px;
    line-height: 1.8;
    color: #9aa8bc;
}
.ai-body strong, .ai-body b {
    color: #c8d4e4 !important;
    font-weight: 600 !important;
}
.ai-body p { margin-bottom: 14px; }
.ai-body ol, .ai-body ul { padding-left: 18px; margin-bottom: 14px; }
.ai-body li { margin-bottom: 4px; }

/* ── AWAIT STATE ── */
.await-wrap {
    background: #202b3d;
    border: 1px solid #2a3549;
    border-radius: 10px;
    padding: 40px 20px;
    text-align: center;
}
.await-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #3a4460;
    margin-bottom: 8px;
}
.await-sub {
    font-size: 12px;
    color: #3a4460;
    line-height: 1.6;
}

/* ── FOOTER ── */
.footer {
    margin: 24px -32px -32px -32px;
    background: #141b2d;
    border-top: 1px solid #2a3349;
    padding: 10px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #3a4460;
    letter-spacing: 0.04em;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p {
    color: #7eb3ff !important;
    font-size: 13px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
N_CLASS   = 4
MODE      = 1
IMG_SIZE  = 508
DEVICE    = "cpu"

CLASS_NAMES  = ["Minimal", "Mild", "Moderate", "Severe"]
CLASS_RANGE  = ["< 10% fibrosis", "10–25% fibrosis", "25–50% fibrosis", "> 50% fibrosis"]
CLASS_COLORS = ["#16a34a", "#d97706", "#ea580c", "#dc2626"]
CLASS_BG     = ["#0f2318", "#231a08", "#231208", "#230e0e"]
CLASS_BORDER = ["#1a4a2a", "#4a3510", "#4a2010", "#4a1010"]
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

Keep each section to 3-5 sentences. End with a brief disclaimer that this is AI-generated and not a substitute for clinical judgment."""

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

# ─────────────────────────────────────────────────────────────────────────────
# TOP BAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-brand">
        <div class="topbar-logo">KF</div>
        <div>
            <div class="topbar-name">Kidney Fibrosis Grader</div>
            <div class="topbar-desc">Automated Interstitial Fibrosis Analysis &nbsp;·&nbsp; ResNet-FPN</div>
        </div>
    </div>
    <div class="topbar-pills">
        <span class="tpill tpill-blue">ResNet-FPN</span>
        <span class="tpill tpill-green">95% Accuracy</span>
        <span class="tpill tpill-amber">Research Only</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────
probs = None
pred  = None
img   = None

# ─────────────────────────────────────────────────────────────────────────────
# ROW 1: Upload | Result | AI Interpretation
# ─────────────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([2.0, 1.9, 3.9], gap="large")

with col1:
    st.markdown('<div class="sec-label">Biopsy Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload biopsy image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_column_width=True)

    # Grade reference
    st.markdown('<div class="sec-label" style="margin-top:20px;">Grade Reference</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="card" style="padding:14px 16px;">
    <div class="ref-row">
        <div class="ref-dot" style="background:#16a34a;"></div>
        <div class="ref-grade">Minimal</div>
        <div class="ref-range">&lt; 10%</div>
        <div class="ref-desc">No significant fibrosis</div>
    </div>
    <div class="ref-row">
        <div class="ref-dot" style="background:#d97706;"></div>
        <div class="ref-grade">Mild</div>
        <div class="ref-range">10–25%</div>
        <div class="ref-desc">Early interstitial fibrosis</div>
    </div>
    <div class="ref-row">
        <div class="ref-dot" style="background:#ea580c;"></div>
        <div class="ref-grade">Moderate</div>
        <div class="ref-range">25–50%</div>
        <div class="ref-desc">Significant fibrosis present</div>
    </div>
    <div class="ref-row">
        <div class="ref-dot" style="background:#dc2626;"></div>
        <div class="ref-grade">Severe</div>
        <div class="ref-range">&gt; 50%</div>
        <div class="ref-desc">High ESKD risk</div>
    </div>
</div>
""", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="sec-label">Analysis Result</div>', unsafe_allow_html=True)

    if uploaded and img is not None:
        with st.spinner("Analyzing..."):
            try:
                probs = predict(img)
                pred  = int(np.argmax(probs))
                c   = CLASS_COLORS[pred]
                bg  = CLASS_BG[pred]
                bo  = CLASS_BORDER[pred]

                st.markdown(f"""
<div class="grade-card" style="background:{bg}; border-color:{bo};">
    <div>
        <div class="grade-name" style="color:{c};">{CLASS_NAMES[pred]}</div>
        <div class="grade-range">{CLASS_RANGE[pred]}</div>
    </div>
    <div>
        <div class="grade-pct" style="color:{c};">{probs[pred]*100:.1f}%</div>
        <div class="grade-conf-lbl">Confidence</div>
    </div>
</div>
""", unsafe_allow_html=True)

                st.markdown('<div class="sec-label">Probability Distribution</div>', unsafe_allow_html=True)
                for i in range(4):
                    pct = probs[i] * 100
                    st.markdown(f"""
<div class="prob-row">
    <div class="prob-name">{CLASS_NAMES[i]}</div>
    <div class="prob-track">
        <div class="prob-fill" style="width:{pct:.1f}%; background:{CLASS_COLORS[i]};"></div>
    </div>
    <div class="prob-pct">{pct:.1f}%</div>
</div>
""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Inference error: {str(e)}")
    else:
        st.markdown("""
<div class="await-wrap">
    <div class="await-label">No Result Yet</div>
    <div class="await-sub">Upload a biopsy image to run the analysis</div>
</div>
""", unsafe_allow_html=True)

with col3:
    st.markdown("""
<div class="ai-header">
    <div class="ai-title">Clinical Interpretation</div>
    <div class="ai-badge">LLAMA 3.3-70B &nbsp;·&nbsp; GROQ</div>
</div>
""", unsafe_allow_html=True)

    if uploaded and probs is not None and pred is not None:
        with st.spinner("Generating clinical interpretation..."):
            try:
                interpretation = get_ai_interpretation(
                    grade_label=f"{CLASS_NAMES[pred]} — {CLASS_RANGE[pred]}",
                    confidence=probs[pred] * 100,
                    all_probs=probs.tolist()
                )
                st.markdown(f'<div class="ai-body">', unsafe_allow_html=True)
                st.markdown(interpretation)
                st.markdown('</div>', unsafe_allow_html=True)
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:
                    st.error("Groq API key missing. Add GROQ_API_KEY to Streamlit secrets.")
                elif e.response.status_code == 429:
                    st.warning("Rate limit reached. Please wait and retry.")
                else:
                    st.error(f"AI unavailable: {str(e)}")
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"AI unavailable: {str(e)}")
    else:
        st.markdown("""
<div class="await-wrap">
    <div class="await-label">Awaiting Analysis</div>
    <div class="await-sub">AI clinical interpretation will appear here automatically after the image is graded</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <span>KIDNEY FIBROSIS GRADER &nbsp;·&nbsp; ResNet-FPN &nbsp;·&nbsp; 95% Test Accuracy</span>
    <span>Research use only — not validated for clinical diagnosis</span>
</div>
""", unsafe_allow_html=True)
