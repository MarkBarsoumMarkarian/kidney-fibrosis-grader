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
CLASS_RANGE  = ["&lt; 10% fibrosis", "10–25% fibrosis", "25–50% fibrosis", "&gt; 50% fibrosis"]
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

def get_unified_report(images, all_probs, all_preds, avg_probs, consensus_pred, consensus_conf):
    """Llama 4 Scout: sees all images + grades, returns one cohesive report."""
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        try:
            groq_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
    if not groq_key:
        raise ValueError("GROQ_API_KEY not configured.")

    n = len(images)
    per_image_summary = ""
    for i in range(n):
        p = all_preds[i]
        per_image_summary += f"  - Image {i+1}: {CLASS_NAMES[p]} ({CLASS_RANGE[p]}, confidence {all_probs[i][p]*100:.1f}%)\n"

    avg_breakdown = "\n".join(
        f"  - {CLASS_SHORT[i]}: {avg_probs[i]*100:.1f}%" for i in range(4)
    )

    multi_note = ""
    if n > 1:
        multi_note = f"""
This is a multi-image analysis ({n} biopsy images from the same patient). Each image was graded independently; the consensus grade is derived from averaged model probabilities across all images.

Per-image grades:
{per_image_summary}"""

    prompt = f"""You are an expert nephropathologist and clinical AI assistant analyzing trichrome-stained kidney biopsy image(s).

Automated Model Output:
- Consensus Grade: {CLASS_NAMES[consensus_pred]} ({CLASS_RANGE[consensus_pred]})
- Consensus Confidence: {consensus_conf:.1f}%
- Averaged probability breakdown:
{avg_breakdown}
{multi_note}
Carefully examine the biopsy image(s) and produce a single cohesive clinical report with exactly these 6 sections:

**Visual Observations**
Describe what you see — collagen deposition (blue/green staining), tubular atrophy, interstitial expansion, glomerular and vascular changes. If multiple images are provided, note consistency or variation across them.

**Agreement with Model Prediction**
Does your visual assessment agree with the consensus grade? Cite specific visual features that support or challenge the model output.

**ESKD Risk & Progression**
What is the risk of end-stage kidney disease at this grade? How likely is progression, and what histological findings drive that risk?

**Treatment Approach**
Based on the fibrosis grade and visual findings, is this patient a candidate for conservative management (blood pressure control, RAAS blockade, lifestyle) or does the severity warrant targeted/interventional therapy? Discuss whether a combined multimodal approach would be appropriate and what that would involve.

**Clinical Recommendations**
What next steps would a nephrologist consider — monitoring intervals, specific interventions, referrals, or additional workup?

**Plain-Language Summary**
Explain the findings and treatment direction in simple terms suitable for a patient.

Keep each section to 3-5 sentences. Do not number the sections. Do not add any disclaimer or closing statement at the end."""

    # Build message content with all images
    content_parts = [{"type": "text", "text": prompt}]
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {groq_key}"}
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [{"role": "user", "content": content_parts}],
        "max_tokens": 1800,
        "temperature": 0.3,
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=90)
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
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for key in ["imgs", "all_probs", "all_preds"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ─────────────────────────────────────────────────────────────────────────────
# ROW 1: Upload | Result
# ─────────────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([2.0, 2.2], gap="large")

with col1:
    st.markdown('<div class="sec-label">Biopsy Images (1–3)</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload 1–3 biopsy images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        imgs = [Image.open(f).convert("RGB") for f in uploaded_files[:3]]
        if imgs != st.session_state.imgs:
            st.session_state.imgs      = imgs
            st.session_state.all_probs = None
            st.session_state.all_preds = None

        if len(imgs) == 1:
            st.image(imgs[0], use_column_width=True)
        else:
            thumb_cols = st.columns(len(imgs))
            for i, (tc, im) in enumerate(zip(thumb_cols, imgs)):
                with tc:
                    st.image(im, use_column_width=True, caption=f"Image {i+1}")
    else:
        st.session_state.imgs      = None
        st.session_state.all_probs = None
        st.session_state.all_preds = None

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

    if st.session_state.imgs is not None and st.session_state.all_probs is None:
        with st.spinner("Analyzing..."):
            try:
                all_probs = [predict(im) for im in st.session_state.imgs]
                all_preds = [int(np.argmax(p)) for p in all_probs]
                st.session_state.all_probs = all_probs
                st.session_state.all_preds = all_preds
            except Exception as e:
                st.error(f"Inference error: {str(e)}")

    if st.session_state.all_probs is not None:
        all_probs = st.session_state.all_probs
        all_preds = st.session_state.all_preds
        n = len(all_probs)

        # Consensus: average probabilities across all images
        avg_probs = np.mean(all_probs, axis=0)
        consensus_pred = int(np.argmax(avg_probs))
        consensus_conf = avg_probs[consensus_pred] * 100

        c  = CLASS_COLORS[consensus_pred]
        bg = CLASS_BG[consensus_pred]
        bo = CLASS_BORDER[consensus_pred]

        # Per-image mini grades if multi-image
        if n > 1:
            per_image_html = ""
            for i in range(n):
                pi = all_preds[i]
                pc = CLASS_COLORS[pi]
                per_image_html += f"""
<div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
    <div style="font-family:'IBM Plex Mono',monospace; font-size:10px; color:#4a5470; width:56px; flex-shrink:0;">Image {i+1}</div>
    <div style="width:8px; height:8px; border-radius:50%; background:{pc}; flex-shrink:0;"></div>
    <div style="font-size:12px; color:{pc}; font-weight:600;">{CLASS_NAMES[pi]}</div>
    <div style="font-family:'IBM Plex Mono',monospace; font-size:10px; color:#4a5470; margin-left:4px;">{all_probs[i][pi]*100:.1f}%</div>
</div>"""
            consensus_label = "CONSENSUS GRADE"
            per_image_block = f"""
    <div style="margin-bottom:12px;">{per_image_html}</div>
    <div class="grade-divider"></div>"""
        else:
            consensus_label = "FIBROSIS GRADE"
            per_image_block = ""

        st.markdown(f"""
<div class="grade-card" style="background:{bg}; border-color:{bo};">
    {per_image_block}
    <div class="grade-sublabel">{consensus_label}</div>
    <div class="grade-name" style="color:{c};">{CLASS_NAMES[consensus_pred]}</div>
    <div class="grade-range">{CLASS_RANGE[consensus_pred]}</div>
    <div class="grade-divider"></div>
    <div class="grade-conf-row">
        <div class="grade-conf-sublabel">Model Confidence</div>
        <div class="grade-conf-value">{consensus_conf:.1f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="sec-label">Probability Distribution</div>', unsafe_allow_html=True)
        for i in range(4):
            pct = avg_probs[i] * 100
            st.markdown(f"""
<div class="prob-row">
    <div class="prob-name">{CLASS_NAMES[i]}</div>
    <div class="prob-track">
        <div class="prob-fill" style="width:{pct:.1f}%; background:{CLASS_COLORS[i]};"></div>
    </div>
    <div class="prob-pct">{pct:.1f}%</div>
</div>
""", unsafe_allow_html=True)

    elif st.session_state.imgs is None:
        st.markdown("""
<div class="await-wrap">
    <div class="await-label">No Result Yet</div>
    <div class="await-sub">Upload 1–3 biopsy images to run the analysis</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED REPORT
# ─────────────────────────────────────────────────────────────────────────────
if (st.session_state.imgs is not None and
        st.session_state.all_probs is not None and
        st.session_state.all_preds is not None):

    all_probs       = st.session_state.all_probs
    all_preds       = st.session_state.all_preds
    avg_probs       = np.mean(all_probs, axis=0)
    consensus_pred  = int(np.argmax(avg_probs))
    consensus_conf  = avg_probs[consensus_pred] * 100

    st.markdown("""
<div class="visual-section">
    <div class="visual-header">
        <div class="visual-title">Pathology Report</div>
        <div class="visual-badge">LLAMA 4 SCOUT &nbsp;·&nbsp; VISION</div>
    </div>
</div>
""", unsafe_allow_html=True)

    rcol1, rcol2 = st.columns([1, 2.4], gap="large")

    with rcol1:
        for i, im in enumerate(st.session_state.imgs):
            caption = f"Image {i+1} — {CLASS_NAMES[all_preds[i]]}" if len(st.session_state.imgs) > 1 else "Analyzed biopsy image"
            st.image(im, use_column_width=True, caption=caption)

    with rcol2:
        with st.spinner("Generating pathology report..."):
            try:
                report = get_unified_report(
                    images=st.session_state.imgs,
                    all_probs=all_probs,
                    all_preds=all_preds,
                    avg_probs=avg_probs,
                    consensus_pred=consensus_pred,
                    consensus_conf=consensus_conf,
                )
                st.markdown('<div class="ai-body">', unsafe_allow_html=True)
                st.markdown(report)
                st.markdown('</div>', unsafe_allow_html=True)
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:
                    st.error("Groq API key missing. Add GROQ_API_KEY to Streamlit secrets.")
                elif e.response.status_code == 429:
                    st.warning("Rate limit reached. Please wait a moment and retry.")
                else:
                    st.error(f"Report unavailable: {str(e)}")
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Report unavailable: {str(e)}")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <span>KIDNEY FIBROSIS GRADER &nbsp;·&nbsp; ResNet-FPN &nbsp;·&nbsp; 95% Test Accuracy</span>
    <span>Research use only — not validated for clinical diagnosis</span>
</div>
""", unsafe_allow_html=True)
