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
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&family=Playfair+Display:wght@600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background: #f5f6f8 !important;
    font-family: 'Inter', sans-serif !important;
    color: #1a1d23 !important;
}

.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], #MainMenu, footer { display: none !important; }

/* ── Top Bar ── */
.topbar {
    background: #ffffff;
    border-bottom: 1px solid #e2e5ea;
    padding: 12px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.topbar-left {
    display: flex;
    align-items: center;
    gap: 14px;
}
.topbar-icon {
    width: 32px; height: 32px;
    background: #1a4fba;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
}
.topbar-title {
    font-family: 'Playfair Display', serif;
    font-size: 17px;
    font-weight: 600;
    color: #1a1d23;
    letter-spacing: -0.01em;
}
.topbar-sub {
    font-size: 11px;
    color: #8a909c;
    margin-top: 1px;
    font-weight: 400;
    letter-spacing: 0.01em;
}
.topbar-right {
    display: flex;
    align-items: center;
    gap: 8px;
}
.pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.04em;
    font-weight: 500;
}
.pill-blue  { background: #e8f0fd; color: #1a4fba; }
.pill-green { background: #e6f6ee; color: #1a7a45; }
.pill-red   { background: #fdecea; color: #b91c1c; border: 1px solid #fca5a5; }

/* ── Three-column grid ── */
.app-grid {
    display: grid;
    grid-template-columns: 320px 280px 1fr;
    gap: 0;
    height: calc(100vh - 57px);
    overflow: hidden;
}

/* ── Panel base ── */
.panel {
    background: #ffffff;
    border-right: 1px solid #e2e5ea;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}
.panel-last { border-right: none; background: #f5f6f8; }

.panel-header {
    padding: 16px 20px 12px;
    border-bottom: 1px solid #e2e5ea;
    flex-shrink: 0;
}
.panel-title {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8a909c;
}
.panel-body {
    padding: 16px 20px;
    flex: 1;
    overflow-y: auto;
}
.panel-body::-webkit-scrollbar { width: 4px; }
.panel-body::-webkit-scrollbar-thumb { background: #dde0e6; border-radius: 2px; }

/* ── Upload zone ── */
[data-testid="stFileUploadDropzone"] {
    background: #f8f9fb !important;
    border: 1.5px dashed #c5cad4 !important;
    border-radius: 10px !important;
    transition: all 0.2s !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #1a4fba !important;
    background: #f0f4fd !important;
}
[data-testid="stFileUploadDropzone"] p { color: #8a909c !important; font-size: 13px !important; }
[data-testid="stFileUploader"] { background: transparent !important; }

/* ── Biopsy image ── */
[data-testid="stImage"] img {
    border-radius: 8px !important;
    border: 1px solid #e2e5ea !important;
    width: 100% !important;
}

/* ── Grade result ── */
.grade-card {
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 16px;
    border-left: 4px solid;
}
.grade-top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
}
.grade-name {
    font-family: 'Playfair Display', serif;
    font-size: 20px;
    font-weight: 600;
    line-height: 1.2;
}
.grade-range {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #8a909c;
    margin-top: 3px;
}
.grade-conf-big {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 26px;
    font-weight: 500;
    line-height: 1;
    text-align: right;
}
.grade-conf-label {
    font-size: 10px;
    color: #8a909c;
    text-align: right;
    margin-top: 2px;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Prob bars ── */
.prob-section-title {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8a909c;
    margin-bottom: 12px;
}
.prob-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 9px;
}
.prob-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #6b7280;
    width: 68px;
    flex-shrink: 0;
}
.prob-track {
    flex: 1;
    height: 5px;
    background: #eef0f3;
    border-radius: 3px;
    overflow: hidden;
}
.prob-fill { height: 100%; border-radius: 3px; }
.prob-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #6b7280;
    width: 36px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Grade reference table ── */
.ref-table { width: 100%; border-collapse: collapse; margin-top: 4px; }
.ref-table td {
    padding: 7px 8px;
    font-size: 11px;
    border-bottom: 1px solid #f0f1f3;
    vertical-align: middle;
}
.ref-table tr:last-child td { border-bottom: none; }
.ref-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    flex-shrink: 0;
}

/* ── AI panel ── */
.ai-panel-inner {
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #e2e5ea;
    height: 100%;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}
.ai-panel-head {
    padding: 16px 22px 14px;
    border-bottom: 1px solid #e2e5ea;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
    background: #fff;
}
.ai-panel-title {
    font-family: 'Playfair Display', serif;
    font-size: 15px;
    font-weight: 600;
    color: #1a1d23;
}
.ai-model-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.08em;
    background: #f0f4fd;
    color: #1a4fba;
    border: 1px solid #c5d4f5;
    padding: 3px 9px;
    border-radius: 4px;
    font-weight: 500;
}
.ai-panel-body {
    padding: 20px 22px;
    flex: 1;
    overflow-y: auto;
    font-size: 13px;
    line-height: 1.75;
    color: #374151;
}
.ai-panel-body::-webkit-scrollbar { width: 4px; }
.ai-panel-body::-webkit-scrollbar-thumb { background: #dde0e6; border-radius: 2px; }
.ai-panel-body strong, .ai-panel-body b {
    color: #111827 !important;
    font-weight: 600 !important;
}
.ai-panel-body p { margin-bottom: 12px; }
.ai-panel-body ol, .ai-panel-body ul { padding-left: 18px; margin-bottom: 12px; }

/* ── Awaiting state ── */
.await-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    gap: 10px;
    color: #c5cad4;
    text-align: center;
    padding: 32px;
}
.await-icon { font-size: 36px; opacity: 0.4; }
.await-text {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #b0b7c3;
}
.await-sub { font-size: 12px; color: #b0b7c3; line-height: 1.5; max-width: 200px; }

/* ── Divider ── */
.inner-divider {
    height: 1px;
    background: #f0f1f3;
    margin: 14px 0;
}

/* ── Footer strip ── */
.footer-strip {
    background: #fff;
    border-top: 1px solid #e2e5ea;
    padding: 7px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 10px;
    color: #b0b7c3;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.04em;
    flex-shrink: 0;
}

/* Streamlit spinner color */
[data-testid="stSpinner"] p { color: #1a4fba !important; font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
N_CLASS    = 4
MODE       = 1
IMG_SIZE   = 508
DEVICE     = "cpu"

CLASS_NAMES  = ["Minimal", "Mild", "Moderate", "Severe"]
CLASS_RANGE  = ["< 10% fibrosis", "10–25% fibrosis", "25–50% fibrosis", "> 50% fibrosis"]
CLASS_COLORS = ["#1a7a45", "#d97706", "#dc6b1a", "#b91c1c"]
CLASS_BG     = ["#e6f6ee", "#fef3c7", "#fff0e6", "#fdecea"]
CLASS_BORDER = ["#1a7a45", "#d97706", "#dc6b1a", "#b91c1c"]
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

# ── Top Bar ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-left">
        <div class="topbar-icon">🔬</div>
        <div>
            <div class="topbar-title">Kidney Fibrosis Grader</div>
            <div class="topbar-sub">Automated Interstitial Fibrosis Analysis · ResNet-FPN</div>
        </div>
    </div>
    <div class="topbar-right">
        <span class="pill pill-blue">ResNet-FPN</span>
        <span class="pill pill-green">95% Accuracy</span>
        <span class="pill pill-red">Research Only</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Three columns via Streamlit ───────────────────────────────────────────────
col1, col2, col3 = st.columns([2.2, 1.9, 3.9], gap="small")

probs = None
pred  = None
img   = None

# ── Column 1: Upload + Image ──────────────────────────────────────────────────
with col1:
    st.markdown("""
    <div class="panel" style="min-height:calc(100vh - 100px);">
        <div class="panel-header"><div class="panel-title">Biopsy Image</div></div>
        <div class="panel-body">
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_column_width=True)

    st.markdown("""
        <div class="inner-divider"></div>
        <div class="prob-section-title">Grade Reference</div>
        <table class="ref-table">
            <tr>
                <td><span class="ref-dot" style="background:#1a7a45;"></span><b>Minimal</b></td>
                <td style="color:#6b7280;">&lt; 10%</td>
                <td style="color:#6b7280;">No significant fibrosis</td>
            </tr>
            <tr>
                <td><span class="ref-dot" style="background:#d97706;"></span><b>Mild</b></td>
                <td style="color:#6b7280;">10–25%</td>
                <td style="color:#6b7280;">Early interstitial fibrosis</td>
            </tr>
            <tr>
                <td><span class="ref-dot" style="background:#dc6b1a;"></span><b>Moderate</b></td>
                <td style="color:#6b7280;">25–50%</td>
                <td style="color:#6b7280;">Significant fibrosis</td>
            </tr>
            <tr>
                <td><span class="ref-dot" style="background:#b91c1c;"></span><b>Severe</b></td>
                <td style="color:#6b7280;">&gt; 50%</td>
                <td style="color:#6b7280;">High ESKD risk</td>
            </tr>
        </table>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Column 2: Grade Result + Probability Bars ─────────────────────────────────
with col2:
    st.markdown("""
    <div class="panel" style="min-height:calc(100vh - 100px);">
        <div class="panel-header"><div class="panel-title">Analysis Result</div></div>
        <div class="panel-body">
    """, unsafe_allow_html=True)

    if uploaded and img is not None:
        with st.spinner("Analyzing..."):
            try:
                probs = predict(img)
                pred  = int(np.argmax(probs))
                c     = CLASS_COLORS[pred]
                bg    = CLASS_BG[pred]

                st.markdown(f"""
<div class="grade-card" style="background:{bg}; border-left-color:{c};">
    <div class="grade-top">
        <div>
            <div class="grade-name" style="color:{c};">{CLASS_NAMES[pred]}</div>
            <div class="grade-range">{CLASS_RANGE[pred]}</div>
        </div>
        <div>
            <div class="grade-conf-big" style="color:{c};">{probs[pred]*100:.1f}%</div>
            <div class="grade-conf-label">Confidence</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

                st.markdown('<div class="prob-section-title">Probability Distribution</div>', unsafe_allow_html=True)
                for i in range(4):
                    pct = probs[i] * 100
                    st.markdown(f"""
<div class="prob-row">
    <div class="prob-label">{CLASS_NAMES[i]}</div>
    <div class="prob-track">
        <div class="prob-fill" style="width:{pct:.1f}%; background:{CLASS_COLORS[i]};"></div>
    </div>
    <div class="prob-val">{pct:.1f}%</div>
</div>
""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.markdown("""
<div class="await-state">
    <div class="await-icon">📊</div>
    <div class="await-text">No Analysis Yet</div>
    <div class="await-sub">Upload a biopsy image to see grading results here</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

# ── Column 3: AI Clinical Interpretation ─────────────────────────────────────
with col3:
    st.markdown("""
    <div style="padding: 0 4px; min-height:calc(100vh - 100px);">
    <div class="ai-panel-inner">
        <div class="ai-panel-head">
            <div class="ai-panel-title">Clinical Interpretation</div>
            <div class="ai-model-badge">LLAMA 3.3-70B · GROQ</div>
        </div>
        <div class="ai-panel-body">
    """, unsafe_allow_html=True)

    if uploaded and probs is not None and pred is not None:
        with st.spinner("Generating clinical interpretation..."):
            try:
                interpretation = get_ai_interpretation(
                    grade_label=f"{CLASS_NAMES[pred]} — {CLASS_RANGE[pred]}",
                    confidence=probs[pred] * 100,
                    all_probs=probs.tolist()
                )
                st.markdown(interpretation)
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:
                    st.error("Groq API key missing. Add GROQ_API_KEY to Streamlit secrets.")
                elif e.response.status_code == 429:
                    st.warning("Rate limit reached. Please wait and try again.")
                else:
                    st.error(f"AI interpretation unavailable: {str(e)}")
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"AI interpretation unavailable: {str(e)}")
    else:
        st.markdown("""
<div class="await-state">
    <div class="await-icon">🤖</div>
    <div class="await-text">Awaiting Analysis</div>
    <div class="await-sub">
        AI clinical interpretation will appear here automatically after image upload and grading
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("</div></div></div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-strip">
    <span>KIDNEY FIBROSIS GRADER · ResNet-FPN · 95% TEST ACCURACY</span>
    <span>⚠ RESEARCH USE ONLY — NOT VALIDATED FOR CLINICAL DIAGNOSIS</span>
</div>
""", unsafe_allow_html=True)
