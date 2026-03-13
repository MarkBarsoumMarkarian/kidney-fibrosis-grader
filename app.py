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

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
[data-testid="stMain"],
section.main {
    background: #eef0f3 !important;
    font-family: 'Inter', sans-serif !important;
    color: #1a1d23 !important;
}

.block-container {
    padding: 0 !important;
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

/* Force column content to fill height */
[data-testid="stHorizontalBlock"] {
    gap: 0 !important;
    align-items: stretch !important;
}
[data-testid="stColumn"] {
    padding: 0 !important;
}
[data-testid="stColumn"] > div {
    height: 100%;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    height: 100%;
}

/* ── TOP BAR ── */
.topbar {
    background: #1a2d5a;
    padding: 0 28px;
    height: 52px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #14244a;
}
.topbar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}
.topbar-logo {
    width: 28px; height: 28px;
    background: rgba(255,255,255,0.15);
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px;
    color: white;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: -1px;
}
.topbar-name {
    font-family: 'Playfair Display', serif;
    font-size: 16px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.01em;
}
.topbar-desc {
    font-size: 10px;
    color: rgba(255,255,255,0.45);
    font-weight: 400;
    margin-top: 1px;
    letter-spacing: 0.03em;
}
.topbar-pills {
    display: flex;
    gap: 8px;
    align-items: center;
}
.tpill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    padding: 4px 10px;
    border-radius: 4px;
    letter-spacing: 0.04em;
    border: 1px solid;
}
.tpill-blue  { color: #93b8ff; border-color: rgba(147,184,255,0.3); background: rgba(147,184,255,0.1); }
.tpill-green { color: #6ee7b7; border-color: rgba(110,231,183,0.3); background: rgba(110,231,183,0.1); }
.tpill-red   { color: #fca5a5; border-color: rgba(252,165,165,0.3); background: rgba(252,165,165,0.1); }

/* ── PANEL WRAPPERS ── */
.panel-wrap {
    background: #ffffff;
    border-right: 1px solid #dde1e8;
    height: calc(100vh - 88px);
    display: flex;
    flex-direction: column;
    overflow: hidden;
}
.panel-wrap-right {
    background: #f8f9fb;
    height: calc(100vh - 88px);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    border-left: 1px solid #dde1e8;
}

.panel-head {
    padding: 14px 20px 12px;
    border-bottom: 1px solid #edf0f4;
    flex-shrink: 0;
    background: #ffffff;
}
.panel-head-right {
    padding: 14px 22px 12px;
    border-bottom: 1px solid #edf0f4;
    flex-shrink: 0;
    background: #ffffff;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.panel-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #9aa0ac;
}
.ai-head-title {
    font-family: 'Playfair Display', serif;
    font-size: 15px;
    font-weight: 700;
    color: #1a1d23;
}
.ai-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    font-weight: 500;
    letter-spacing: 0.08em;
    background: #eef3fd;
    color: #1a4fba;
    border: 1px solid #c5d4f5;
    padding: 3px 9px;
    border-radius: 4px;
}
.panel-body {
    flex: 1;
    overflow-y: auto;
    padding: 18px 20px;
}
.panel-body::-webkit-scrollbar { width: 3px; }
.panel-body::-webkit-scrollbar-thumb { background: #dde1e8; border-radius: 2px; }

.panel-body-right {
    flex: 1;
    overflow-y: auto;
    padding: 20px 24px;
    font-size: 13.5px;
    line-height: 1.8;
    color: #374151;
}
.panel-body-right::-webkit-scrollbar { width: 3px; }
.panel-body-right::-webkit-scrollbar-thumb { background: #dde1e8; border-radius: 2px; }
.panel-body-right strong, .panel-body-right b {
    color: #111827 !important;
    font-weight: 600 !important;
}
.panel-body-right p { margin-bottom: 14px; }
.panel-body-right ol, .panel-body-right ul {
    padding-left: 18px;
    margin-bottom: 14px;
}
.panel-body-right li { margin-bottom: 4px; }

/* ── UPLOAD ZONE ── */
[data-testid="stFileUploader"] { background: transparent !important; }
[data-testid="stFileUploadDropzone"] {
    background: #f4f6fa !important;
    border: 2px dashed #c0c8d8 !important;
    border-radius: 10px !important;
    padding: 32px 20px !important;
    text-align: center !important;
    transition: all 0.2s !important;
    min-height: 120px !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #1a4fba !important;
    background: #eef3fd !important;
}
[data-testid="stFileUploadDropzone"] p {
    color: #6b7280 !important;
    font-size: 13px !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stFileUploadDropzone"] small { color: #9aa0ac !important; font-size: 11px !important; }
[data-testid="stFileUploadDropzone"] svg { fill: #c0c8d8 !important; }

/* ── BIOPSY IMAGE ── */
[data-testid="stImage"] img {
    border-radius: 8px !important;
    border: 1px solid #dde1e8 !important;
    width: 100% !important;
    margin-top: 14px !important;
}

/* ── GRADE CARD ── */
.grade-card {
    border-radius: 10px;
    padding: 18px 20px 16px;
    margin-bottom: 20px;
    border: 1px solid;
    position: relative;
}
.grade-row {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
}
.grade-left {}
.grade-name {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 700;
    line-height: 1.15;
}
.grade-range {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #6b7280;
    margin-top: 4px;
    font-weight: 400;
}
.grade-right { text-align: right; }
.grade-pct {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 28px;
    font-weight: 500;
    line-height: 1;
}
.grade-conf-lbl {
    font-size: 10px;
    color: #9aa0ac;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 3px;
}

/* ── PROB BARS ── */
.section-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #9aa0ac;
    margin-bottom: 13px;
    margin-top: 4px;
}
.prob-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}
.prob-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #6b7280;
    width: 64px;
    flex-shrink: 0;
    font-weight: 500;
}
.prob-track {
    flex: 1;
    height: 6px;
    background: #eef0f3;
    border-radius: 3px;
    overflow: hidden;
}
.prob-fill { height: 100%; border-radius: 3px; }
.prob-pct {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #6b7280;
    width: 38px;
    text-align: right;
    flex-shrink: 0;
}

/* ── GRADE REF TABLE ── */
.divider { height: 1px; background: #eef0f3; margin: 18px 0; }
.ref-table { width: 100%; border-collapse: collapse; }
.ref-table td {
    padding: 7px 6px;
    font-size: 11px;
    border-bottom: 1px solid #f3f4f6;
    color: #374151;
    vertical-align: middle;
}
.ref-table tr:last-child td { border-bottom: none; }
.ref-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 7px;
    vertical-align: middle;
    flex-shrink: 0;
}
.ref-grade { font-weight: 600; color: #1a1d23; font-size: 11px; }
.ref-range { font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #9aa0ac; }
.ref-desc  { font-size: 11px; color: #6b7280; }

/* ── AWAIT STATE ── */
.await-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 240px;
    gap: 10px;
    text-align: center;
}
.await-icon {
    width: 44px; height: 44px;
    background: #f0f2f5;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 4px;
}
.await-icon svg { opacity: 0.35; }
.await-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #b0b7c3;
}
.await-sub { font-size: 12px; color: #b0b7c3; line-height: 1.6; max-width: 200px; }

/* ── FOOTER ── */
.footer {
    height: 36px;
    background: #ffffff;
    border-top: 1px solid #dde1e8;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 28px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #b0b7c3;
    letter-spacing: 0.04em;
}

/* Spinner */
[data-testid="stSpinner"] p {
    color: #1a4fba !important;
    font-size: 13px !important;
    font-family: 'Inter', sans-serif !important;
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
CLASS_BG     = ["#f0fdf4", "#fffbeb", "#fff7ed", "#fef2f2"]
CLASS_BORDER = ["#bbf7d0", "#fde68a", "#fed7aa", "#fecaca"]
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
        <span class="tpill tpill-red">Research Only</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# THREE COLUMNS
# ─────────────────────────────────────────────────────────────────────────────
probs  = None
pred   = None
img    = None

col1, col2, col3 = st.columns([2.2, 2.0, 3.8], gap="small")

# ── COL 1: Upload + image ─────────────────────────────────────────────────────
with col1:
    st.markdown("""
    <div class="panel-wrap">
        <div class="panel-head"><div class="panel-label">Biopsy Image</div></div>
        <div class="panel-body">
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload biopsy image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_column_width=True)

    st.markdown("""
        <div class="divider"></div>
        <div class="section-label">Grade Reference</div>
        <table class="ref-table">
            <tr>
                <td><span class="ref-dot" style="background:#16a34a;"></span><span class="ref-grade">Minimal</span></td>
                <td class="ref-range">&lt; 10%</td>
                <td class="ref-desc">No significant fibrosis</td>
            </tr>
            <tr>
                <td><span class="ref-dot" style="background:#d97706;"></span><span class="ref-grade">Mild</span></td>
                <td class="ref-range">10–25%</td>
                <td class="ref-desc">Early interstitial fibrosis</td>
            </tr>
            <tr>
                <td><span class="ref-dot" style="background:#ea580c;"></span><span class="ref-grade">Moderate</span></td>
                <td class="ref-range">25–50%</td>
                <td class="ref-desc">Significant fibrosis</td>
            </tr>
            <tr>
                <td><span class="ref-dot" style="background:#dc2626;"></span><span class="ref-grade">Severe</span></td>
                <td class="ref-range">&gt; 50%</td>
                <td class="ref-desc">High ESKD risk</td>
            </tr>
        </table>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── COL 2: Grade result + prob bars ──────────────────────────────────────────
with col2:
    st.markdown("""
    <div class="panel-wrap">
        <div class="panel-head"><div class="panel-label">Analysis Result</div></div>
        <div class="panel-body">
    """, unsafe_allow_html=True)

    if uploaded and img is not None:
        with st.spinner("Analyzing image..."):
            try:
                probs = predict(img)
                pred  = int(np.argmax(probs))
                c     = CLASS_COLORS[pred]
                bg    = CLASS_BG[pred]
                bo    = CLASS_BORDER[pred]

                st.markdown(f"""
<div class="grade-card" style="background:{bg}; border-color:{bo};">
    <div class="grade-row">
        <div class="grade-left">
            <div class="grade-name" style="color:{c};">{CLASS_NAMES[pred]}</div>
            <div class="grade-range">{CLASS_RANGE[pred]}</div>
        </div>
        <div class="grade-right">
            <div class="grade-pct" style="color:{c};">{probs[pred]*100:.1f}%</div>
            <div class="grade-conf-lbl">Confidence</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

                st.markdown('<div class="section-label">Probability Distribution</div>', unsafe_allow_html=True)
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
    <div class="await-icon">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#6b7280" stroke-width="1.5">
            <path d="M9 17H7A5 5 0 0 1 7 7h2"/>
            <path d="M15 7h2a5 5 0 1 1 0 10h-2"/>
            <line x1="8" y1="12" x2="16" y2="12"/>
        </svg>
    </div>
    <div class="await-label">No Result Yet</div>
    <div class="await-sub">Upload a biopsy image to run the analysis</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

# ── COL 3: AI Clinical Interpretation ────────────────────────────────────────
with col3:
    st.markdown("""
    <div class="panel-wrap-right">
        <div class="panel-head-right">
            <div class="ai-head-title">Clinical Interpretation</div>
            <div class="ai-badge">LLAMA 3.3-70B &nbsp;·&nbsp; GROQ</div>
        </div>
        <div class="panel-body-right">
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
    <div class="await-icon">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#6b7280" stroke-width="1.5">
            <circle cx="12" cy="12" r="10"/>
            <path d="M12 16v-4M12 8h.01"/>
        </svg>
    </div>
    <div class="await-label">Awaiting Analysis</div>
    <div class="await-sub">AI clinical interpretation will appear here after grading</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <span>KIDNEY FIBROSIS GRADER &nbsp;·&nbsp; ResNet-FPN &nbsp;·&nbsp; 95% Test Accuracy</span>
    <span>Research use only — not validated for clinical diagnosis</span>
</div>
""", unsafe_allow_html=True)
