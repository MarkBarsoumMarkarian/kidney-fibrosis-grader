import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import sys, os
import requests
import base64
import io

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

/* TOP BAR */
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
    font-size: 11px; font-weight: 700; color: #fff;
    display: flex; align-items: center; justify-content: center;
    letter-spacing: -0.5px;
}
.topbar-name {
    font-family: 'Playfair Display', serif;
    font-size: 16px; font-weight: 700; color: #e8edf5;
}
.topbar-desc { font-size: 10px; color: #5a6480; margin-top: 1px; letter-spacing: 0.02em; }
.topbar-pills { display: flex; gap: 8px; }
.tpill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; font-weight: 500;
    padding: 4px 10px; border-radius: 4px;
    letter-spacing: 0.04em; border: 1px solid;
}
.tpill-blue  { color: #7eb3ff; border-color: #2a4a80; background: #1a2d4a; }
.tpill-green { color: #6ee7b7; border-color: #1a4a35; background: #102a20; }
.tpill-amber { color: #fbbf24; border-color: #4a3510; background: #2a1e08; }

/* SECTION LABEL */
.sec-label {
    font-size: 10px; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: #4a5470; margin-bottom: 10px; margin-top: 2px;
}

/* CARD */
.card {
    background: #202b3d; border: 1px solid #2a3549;
    border-radius: 10px; padding: 20px; margin-bottom: 16px;
}

/* UPLOAD ZONE */
[data-testid="stFileUploader"] { background: transparent !important; }
[data-testid="stFileUploadDropzone"] {
    background: #1a2435 !important;
    border: 2px dashed #2e3f5c !important;
    border-radius: 10px !important;
    padding: 28px !important; transition: all 0.2s !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #2563eb !important; background: #1a2845 !important;
}
[data-testid="stFileUploadDropzone"] p { color: #5a6480 !important; font-size: 13px !important; }
[data-testid="stFileUploadDropzone"] small { color: #3a4460 !important; }
[data-testid="stFileUploadDropzone"] svg { fill: #2e3f5c !important; }

/* IMAGE */
[data-testid="stImage"] img {
    border-radius: 8px !important; border: 1px solid #2a3549 !important; width: 100% !important;
}

/* GRADE CARD */
.grade-card {
    border-radius: 10px; padding: 18px 20px; border: 1px solid; margin-bottom: 20px;
}
.grade-name {
    font-family: 'Playfair Display', serif;
    font-size: 26px; font-weight: 700; line-height: 1.1;
}
.grade-sublabel {
    font-size: 10px; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #4a5470; margin-bottom: 4px;
}
.grade-range {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; color: #7a8490; margin-top: 3px;
}
.grade-divider { height: 1px; background: rgba(255,255,255,0.06); margin: 14px 0; }
.grade-conf-row { display: flex; align-items: center; justify-content: space-between; }
.grade-conf-sublabel {
    font-size: 10px; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #4a5470;
}
.grade-conf-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 14px; font-weight: 500; color: #8a9ab0;
}

/* PROB BARS */
.prob-row { display: flex; align-items: center; gap: 10px; margin-bottom: 11px; }
.prob-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #5a6480; width: 62px; flex-shrink: 0; font-weight: 500;
}
.prob-track { flex: 1; height: 6px; background: #1a2435; border-radius: 3px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 3px; }
.prob-pct {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #5a6480; width: 38px; text-align: right; flex-shrink: 0;
}

/* GRADE REF */
.ref-row {
    display: flex; align-items: center; gap: 10px;
    padding: 7px 0; border-bottom: 1px solid #242f42;
    font-size: 12px; color: #8a9ab0;
}
.ref-row:last-child { border-bottom: none; }
.ref-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.ref-grade { font-weight: 600; color: #b0bac8; width: 64px; flex-shrink: 0; }
.ref-range {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #4a5470; width: 60px; flex-shrink: 0;
}
.ref-desc { color: #5a6880; font-size: 11px; }

/* AI SECTION */
.ai-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 18px; padding-bottom: 14px; border-bottom: 1px solid #2a3549;
}
.ai-title {
    font-family: 'Playfair Display', serif;
    font-size: 18px; font-weight: 700; color: #e0e6f0;
}
.ai-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px; font-weight: 500; letter-spacing: 0.08em;
    background: #1a2845; color: #7eb3ff;
    border: 1px solid #2a4a80; padding: 4px 10px; border-radius: 4px;
}
.ai-body { font-size: 13.5px; line-height: 1.8; color: #9aa8bc; }
.ai-body strong, .ai-body b { color: #c8d4e4 !important; font-weight: 600 !important; }
.ai-body p { margin-bottom: 14px; }
.ai-body ol, .ai-body ul { padding-left: 18px; margin-bottom: 14px; }
.ai-body li { margin-bottom: 4px; }

/* AWAIT STATE */
.await-wrap {
    background: #202b3d; border: 1px solid #2a3549;
    border-radius: 10px; padding: 40px 20px; text-align: center;
}
.await-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; font-weight: 500; letter-spacing: 0.1em;
    text-transform: uppercase; color: #3a4460; margin-bottom: 8px;
}
.await-sub { font-size: 12px; color: #3a4460; line-height: 1.6; }

/* VISUAL ANALYSIS SECTION */
.visual-section {
    margin-top: 28px; border-top: 1px solid #2a3349; padding-top: 24px;
}
.visual-header {
    display: flex; align-items: center;
    justify-content: space-between; margin-bottom: 18px;
}
.visual-title {
    font-family: 'Playfair Display', serif;
    font-size: 18px; font-weight: 700; color: #e0e6f0;
}
.visual-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px; font-weight: 500; letter-spacing: 0.08em;
    background: #1a2d1a; color: #6ee7b7;
    border: 1px solid #1a4a35; padding: 4px 10px; border-radius: 4px;
}
.novel-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px; font-weight: 700; letter-spacing: 0.1em;
    background: #2a1a4a; color: #c084fc;
    border: 1px solid #4a2a80; padding: 4px 10px;
    border-radius: 4px; margin-left: 8px;
}

/* FOOTER */
.footer {
    margin: 24px -32px -32px -32px;
    background: #141b2d; border-top: 1px solid #2a3349;
    padding: 10px 32px; display: flex;
    align-items: center; justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #3a4460; letter-spacing: 0.04em;
}

[data-testid="stSpinner"] p { color: #7eb3ff !important; font-size: 13px !important; }
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

**End-Stage Kidney Disease (ESKD) Assessment**
Is this grade associated with ESKD, or is it a risk? Be specific.

**Progression Risk**
How likely is this to worsen? What factors drive progression at this level?

**Clinical Recommendations**
What next steps would a nephrologist consider? (monitoring, interventions, referrals)

**Plain-Language Summary**
Explain in simple language suitable for a patient.

Keep each section to 3-5 sentences. Do not number the sections. End with a brief disclaimer that this is AI-generated and not a substitute for clinical judgment."""

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {groq_key}"}
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000, "temperature": 0.3,
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def get_visual_analysis(pil_image, grade_label, confidence):
    """Send biopsy image to Gemini 2.5 Flash for visual pathology analysis."""
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        try:
            gemini_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY not configured.")

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    prompt = f"""You are an expert nephropathologist analyzing a trichrome-stained kidney biopsy image.
An automated deep learning model has predicted: {grade_label} (model confidence: {confidence:.1f}%).

Carefully examine this biopsy image and provide a structured visual pathology report covering:

**Visual Observations**
Describe what you can see — collagen deposition patterns (blue/green staining), tubular atrophy, interstitial expansion, glomerular changes, and vascular features. Be specific about the distribution and extent of fibrotic areas visible.

**Agreement with Model Prediction**
Does what you visually observe agree with the model's predicted grade? Note areas of the image that particularly support or contradict the predicted grade.

**Histological Features of Note**
Highlight any specific histological features visible that are clinically significant beyond the fibrosis grade — such as tubular dropout, periglomerular fibrosis, arterial changes, or inflammatory infiltrates.

Be precise and use proper nephropathology terminology. Keep each section to 3-5 sentences. End with a note that this is AI-assisted visual analysis and should be reviewed by a qualified pathologist."""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_key}"
    payload = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
        ]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1000}
    }
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]


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
# SESSION STATE — persists values across column boundaries
# ─────────────────────────────────────────────────────────────────────────────
for key in ["probs", "pred", "img"]:
    if key not in st.session_state:
        st.session_state[key] = None

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
        st.session_state.img   = Image.open(uploaded).convert("RGB")
        st.session_state.probs = None  # reset so model re-runs on new image
        st.session_state.pred  = None

    if st.session_state.img is not None:
        st.image(st.session_state.img, use_column_width=True)

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
    if st.session_state.img is not None and st.session_state.probs is None:
        with st.spinner("Analyzing..."):
            try:
                st.session_state.probs = predict(st.session_state.img)
                st.session_state.pred  = int(np.argmax(st.session_state.probs))
            except Exception as e:
                st.error(f"Inference error: {str(e)}")

    # Display results if we have them (whether just computed or from session state)
    if st.session_state.probs is not None and st.session_state.pred is not None:
        p  = st.session_state.pred
        c  = CLASS_COLORS[p]
        bg = CLASS_BG[p]
        bo = CLASS_BORDER[p]
        st.markdown(f"""
<div class="grade-card" style="background:{bg}; border-color:{bo};">
    <div class="grade-sublabel">Fibrosis Grade</div>
    <div class="grade-name" style="color:{c};">{CLASS_NAMES[p]}</div>
    <div class="grade-range">{CLASS_RANGE[p]}</div>
    <div class="grade-divider"></div>
    <div class="grade-conf-row">
        <div class="grade-conf-sublabel">Model Confidence</div>
        <div class="grade-conf-value">{st.session_state.probs[p]*100:.1f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Probability Distribution</div>', unsafe_allow_html=True)
        for i in range(4):
            pct = st.session_state.probs[i] * 100
            st.markdown(f"""
<div class="prob-row">
    <div class="prob-name">{CLASS_NAMES[i]}</div>
    <div class="prob-track">
        <div class="prob-fill" style="width:{pct:.1f}%; background:{CLASS_COLORS[i]};"></div>
    </div>
    <div class="prob-pct">{pct:.1f}%</div>
</div>
""", unsafe_allow_html=True)
    elif st.session_state.img is None:
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

    if st.session_state.probs is not None and st.session_state.pred is not None:
        with st.spinner("Generating clinical interpretation..."):
            try:
                p = st.session_state.pred
                interpretation = get_ai_interpretation(
                    grade_label=f"{CLASS_NAMES[p]} — {CLASS_RANGE[p]}",
                    confidence=st.session_state.probs[p] * 100,
                    all_probs=st.session_state.probs.tolist()
                )
                st.markdown('<div class="ai-body">', unsafe_allow_html=True)
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
# VISUAL ANALYSIS (Gemini 2.5 Flash — image grounded)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.img is not None and st.session_state.probs is not None and st.session_state.pred is not None:
    p = st.session_state.pred

    st.markdown("""
<div class="visual-section">
    <div class="visual-header">
        <div style="display:flex; align-items:center; gap:10px;">
            <div class="visual-title">Visual Pathology Analysis</div>
            <span class="novel-tag">NOVEL</span>
        </div>
        <div class="visual-badge">GEMINI 2.5 FLASH &nbsp;·&nbsp; VISION</div>
    </div>
</div>
""", unsafe_allow_html=True)

    vcol1, vcol2 = st.columns([1, 2], gap="large")

    with vcol1:
        st.image(st.session_state.img, use_column_width=True, caption="Analyzed biopsy image")

    with vcol2:
        with st.spinner("Gemini is analyzing the biopsy image..."):
            try:
                visual_report = get_visual_analysis(
                    pil_image=st.session_state.img,
                    grade_label=f"{CLASS_NAMES[p]} — {CLASS_RANGE[p]}",
                    confidence=st.session_state.probs[p] * 100
                )
                st.markdown('<div class="ai-body">', unsafe_allow_html=True)
                st.markdown(visual_report)
                st.markdown('</div>', unsafe_allow_html=True)
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 400:
                    st.error("Gemini API error — check your API key.")
                elif e.response.status_code == 429:
                    st.warning("Gemini rate limit reached. Please wait and retry.")
                else:
                    st.error(f"Visual analysis unavailable: {str(e)}")
            except ValueError as e:
                st.info(str(e) + " — Get a free key at aistudio.google.com and add GEMINI_API_KEY to Streamlit secrets.")
            except Exception as e:
                st.error(f"Visual analysis unavailable: {str(e)}")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <span>KIDNEY FIBROSIS GRADER &nbsp;·&nbsp; ResNet-FPN &nbsp;·&nbsp; 95% Test Accuracy</span>
    <span>Research use only — not validated for clinical diagnosis</span>
</div>
""", unsafe_allow_html=True)import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import sys, os
import requests
import base64
import io

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

/* TOP BAR */
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
    font-size: 11px; font-weight: 700; color: #fff;
    display: flex; align-items: center; justify-content: center;
    letter-spacing: -0.5px;
}
.topbar-name {
    font-family: 'Playfair Display', serif;
    font-size: 16px; font-weight: 700; color: #e8edf5;
}
.topbar-desc { font-size: 10px; color: #5a6480; margin-top: 1px; letter-spacing: 0.02em; }
.topbar-pills { display: flex; gap: 8px; }
.tpill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; font-weight: 500;
    padding: 4px 10px; border-radius: 4px;
    letter-spacing: 0.04em; border: 1px solid;
}
.tpill-blue  { color: #7eb3ff; border-color: #2a4a80; background: #1a2d4a; }
.tpill-green { color: #6ee7b7; border-color: #1a4a35; background: #102a20; }
.tpill-amber { color: #fbbf24; border-color: #4a3510; background: #2a1e08; }

/* SECTION LABEL */
.sec-label {
    font-size: 10px; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: #4a5470; margin-bottom: 10px; margin-top: 2px;
}

/* CARD */
.card {
    background: #202b3d; border: 1px solid #2a3549;
    border-radius: 10px; padding: 20px; margin-bottom: 16px;
}

/* UPLOAD ZONE */
[data-testid="stFileUploader"] { background: transparent !important; }
[data-testid="stFileUploadDropzone"] {
    background: #1a2435 !important;
    border: 2px dashed #2e3f5c !important;
    border-radius: 10px !important;
    padding: 28px !important; transition: all 0.2s !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #2563eb !important; background: #1a2845 !important;
}
[data-testid="stFileUploadDropzone"] p { color: #5a6480 !important; font-size: 13px !important; }
[data-testid="stFileUploadDropzone"] small { color: #3a4460 !important; }
[data-testid="stFileUploadDropzone"] svg { fill: #2e3f5c !important; }

/* IMAGE */
[data-testid="stImage"] img {
    border-radius: 8px !important; border: 1px solid #2a3549 !important; width: 100% !important;
}

/* GRADE CARD */
.grade-card {
    border-radius: 10px; padding: 18px 20px; border: 1px solid; margin-bottom: 20px;
}
.grade-name {
    font-family: 'Playfair Display', serif;
    font-size: 26px; font-weight: 700; line-height: 1.1;
}
.grade-sublabel {
    font-size: 10px; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #4a5470; margin-bottom: 4px;
}
.grade-range {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; color: #7a8490; margin-top: 3px;
}
.grade-divider { height: 1px; background: rgba(255,255,255,0.06); margin: 14px 0; }
.grade-conf-row { display: flex; align-items: center; justify-content: space-between; }
.grade-conf-sublabel {
    font-size: 10px; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #4a5470;
}
.grade-conf-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 14px; font-weight: 500; color: #8a9ab0;
}

/* PROB BARS */
.prob-row { display: flex; align-items: center; gap: 10px; margin-bottom: 11px; }
.prob-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #5a6480; width: 62px; flex-shrink: 0; font-weight: 500;
}
.prob-track { flex: 1; height: 6px; background: #1a2435; border-radius: 3px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 3px; }
.prob-pct {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #5a6480; width: 38px; text-align: right; flex-shrink: 0;
}

/* GRADE REF */
.ref-row {
    display: flex; align-items: center; gap: 10px;
    padding: 7px 0; border-bottom: 1px solid #242f42;
    font-size: 12px; color: #8a9ab0;
}
.ref-row:last-child { border-bottom: none; }
.ref-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.ref-grade { font-weight: 600; color: #b0bac8; width: 64px; flex-shrink: 0; }
.ref-range {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #4a5470; width: 60px; flex-shrink: 0;
}
.ref-desc { color: #5a6880; font-size: 11px; }

/* AI SECTION */
.ai-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 18px; padding-bottom: 14px; border-bottom: 1px solid #2a3549;
}
.ai-title {
    font-family: 'Playfair Display', serif;
    font-size: 18px; font-weight: 700; color: #e0e6f0;
}
.ai-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px; font-weight: 500; letter-spacing: 0.08em;
    background: #1a2845; color: #7eb3ff;
    border: 1px solid #2a4a80; padding: 4px 10px; border-radius: 4px;
}
.ai-body { font-size: 13.5px; line-height: 1.8; color: #9aa8bc; }
.ai-body strong, .ai-body b { color: #c8d4e4 !important; font-weight: 600 !important; }
.ai-body p { margin-bottom: 14px; }
.ai-body ol, .ai-body ul { padding-left: 18px; margin-bottom: 14px; }
.ai-body li { margin-bottom: 4px; }

/* AWAIT STATE */
.await-wrap {
    background: #202b3d; border: 1px solid #2a3549;
    border-radius: 10px; padding: 40px 20px; text-align: center;
}
.await-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; font-weight: 500; letter-spacing: 0.1em;
    text-transform: uppercase; color: #3a4460; margin-bottom: 8px;
}
.await-sub { font-size: 12px; color: #3a4460; line-height: 1.6; }

/* VISUAL ANALYSIS SECTION */
.visual-section {
    margin-top: 28px; border-top: 1px solid #2a3349; padding-top: 24px;
}
.visual-header {
    display: flex; align-items: center;
    justify-content: space-between; margin-bottom: 18px;
}
.visual-title {
    font-family: 'Playfair Display', serif;
    font-size: 18px; font-weight: 700; color: #e0e6f0;
}
.visual-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px; font-weight: 500; letter-spacing: 0.08em;
    background: #1a2d1a; color: #6ee7b7;
    border: 1px solid #1a4a35; padding: 4px 10px; border-radius: 4px;
}
.novel-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px; font-weight: 700; letter-spacing: 0.1em;
    background: #2a1a4a; color: #c084fc;
    border: 1px solid #4a2a80; padding: 4px 10px;
    border-radius: 4px; margin-left: 8px;
}

/* FOOTER */
.footer {
    margin: 24px -32px -32px -32px;
    background: #141b2d; border-top: 1px solid #2a3349;
    padding: 10px 32px; display: flex;
    align-items: center; justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #3a4460; letter-spacing: 0.04em;
}

[data-testid="stSpinner"] p { color: #7eb3ff !important; font-size: 13px !important; }
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

**End-Stage Kidney Disease (ESKD) Assessment**
Is this grade associated with ESKD, or is it a risk? Be specific.

**Progression Risk**
How likely is this to worsen? What factors drive progression at this level?

**Clinical Recommendations**
What next steps would a nephrologist consider? (monitoring, interventions, referrals)

**Plain-Language Summary**
Explain in simple language suitable for a patient.

Keep each section to 3-5 sentences. Do not number the sections. End with a brief disclaimer that this is AI-generated and not a substitute for clinical judgment."""

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {groq_key}"}
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000, "temperature": 0.3,
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def get_visual_analysis(pil_image, grade_label, confidence):
    """Send biopsy image to Gemini 2.5 Flash for visual pathology analysis."""
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        try:
            gemini_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY not configured.")

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    prompt = f"""You are an expert nephropathologist analyzing a trichrome-stained kidney biopsy image.
An automated deep learning model has predicted: {grade_label} (model confidence: {confidence:.1f}%).

Carefully examine this biopsy image and provide a structured visual pathology report covering:

**Visual Observations**
Describe what you can see — collagen deposition patterns (blue/green staining), tubular atrophy, interstitial expansion, glomerular changes, and vascular features. Be specific about the distribution and extent of fibrotic areas visible.

**Agreement with Model Prediction**
Does what you visually observe agree with the model's predicted grade? Note areas of the image that particularly support or contradict the predicted grade.

**Histological Features of Note**
Highlight any specific histological features visible that are clinically significant beyond the fibrosis grade — such as tubular dropout, periglomerular fibrosis, arterial changes, or inflammatory infiltrates.

Be precise and use proper nephropathology terminology. Keep each section to 3-5 sentences. End with a note that this is AI-assisted visual analysis and should be reviewed by a qualified pathologist."""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-04-17:generateContent?key={gemini_key}"
    payload = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
        ]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1000}
    }
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]


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
# SESSION STATE — persists values across column boundaries
# ─────────────────────────────────────────────────────────────────────────────
for key in ["probs", "pred", "img"]:
    if key not in st.session_state:
        st.session_state[key] = None

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
        st.session_state.img   = Image.open(uploaded).convert("RGB")
        st.session_state.probs = None  # reset so model re-runs on new image
        st.session_state.pred  = None

    if st.session_state.img is not None:
        st.image(st.session_state.img, use_column_width=True)

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
    if st.session_state.img is not None and st.session_state.probs is None:
        with st.spinner("Analyzing..."):
            try:
                st.session_state.probs = predict(st.session_state.img)
                st.session_state.pred  = int(np.argmax(st.session_state.probs))
            except Exception as e:
                st.error(f"Inference error: {str(e)}")

    # Display results if we have them (whether just computed or from session state)
    if st.session_state.probs is not None and st.session_state.pred is not None:
        p  = st.session_state.pred
        c  = CLASS_COLORS[p]
        bg = CLASS_BG[p]
        bo = CLASS_BORDER[p]
        st.markdown(f"""
<div class="grade-card" style="background:{bg}; border-color:{bo};">
    <div class="grade-sublabel">Fibrosis Grade</div>
    <div class="grade-name" style="color:{c};">{CLASS_NAMES[p]}</div>
    <div class="grade-range">{CLASS_RANGE[p]}</div>
    <div class="grade-divider"></div>
    <div class="grade-conf-row">
        <div class="grade-conf-sublabel">Model Confidence</div>
        <div class="grade-conf-value">{st.session_state.probs[p]*100:.1f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Probability Distribution</div>', unsafe_allow_html=True)
        for i in range(4):
            pct = st.session_state.probs[i] * 100
            st.markdown(f"""
<div class="prob-row">
    <div class="prob-name">{CLASS_NAMES[i]}</div>
    <div class="prob-track">
        <div class="prob-fill" style="width:{pct:.1f}%; background:{CLASS_COLORS[i]};"></div>
    </div>
    <div class="prob-pct">{pct:.1f}%</div>
</div>
""", unsafe_allow_html=True)
    elif st.session_state.img is None:
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

    if st.session_state.probs is not None and st.session_state.pred is not None:
        with st.spinner("Generating clinical interpretation..."):
            try:
                p = st.session_state.pred
                interpretation = get_ai_interpretation(
                    grade_label=f"{CLASS_NAMES[p]} — {CLASS_RANGE[p]}",
                    confidence=st.session_state.probs[p] * 100,
                    all_probs=st.session_state.probs.tolist()
                )
                st.markdown('<div class="ai-body">', unsafe_allow_html=True)
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
# VISUAL ANALYSIS (Gemini 2.5 Flash — image grounded)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.img is not None and st.session_state.probs is not None and st.session_state.pred is not None:
    p = st.session_state.pred

    st.markdown("""
<div class="visual-section">
    <div class="visual-header">
        <div style="display:flex; align-items:center; gap:10px;">
            <div class="visual-title">Visual Pathology Analysis</div>
            <span class="novel-tag">NOVEL</span>
        </div>
        <div class="visual-badge">GEMINI 2.5 FLASH &nbsp;·&nbsp; VISION</div>
    </div>
</div>
""", unsafe_allow_html=True)

    vcol1, vcol2 = st.columns([1, 2], gap="large")

    with vcol1:
        st.image(st.session_state.img, use_column_width=True, caption="Analyzed biopsy image")

    with vcol2:
        with st.spinner("Gemini is analyzing the biopsy image..."):
            try:
                visual_report = get_visual_analysis(
                    pil_image=st.session_state.img,
                    grade_label=f"{CLASS_NAMES[p]} — {CLASS_RANGE[p]}",
                    confidence=st.session_state.probs[p] * 100
                )
                st.markdown('<div class="ai-body">', unsafe_allow_html=True)
                st.markdown(visual_report)
                st.markdown('</div>', unsafe_allow_html=True)
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 400:
                    st.error("Gemini API error — check your API key.")
                elif e.response.status_code == 429:
                    st.warning("Gemini rate limit reached. Please wait and retry.")
                else:
                    st.error(f"Visual analysis unavailable: {str(e)}")
            except ValueError as e:
                st.info(str(e) + " — Get a free key at aistudio.google.com and add GEMINI_API_KEY to Streamlit secrets.")
            except Exception as e:
                st.error(f"Visual analysis unavailable: {str(e)}")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <span>KIDNEY FIBROSIS GRADER &nbsp;·&nbsp; ResNet-FPN &nbsp;·&nbsp; 95% Test Accuracy</span>
    <span>Research use only — not validated for clinical diagnosis</span>
</div>
""", unsafe_allow_html=True)
