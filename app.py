import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import sys, os
import requests
import base64
import io
import cv2
import hashlib

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

.sec-label {
    font-size: 10px; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: #4a5470; margin-bottom: 10px; margin-top: 2px;
}

.card {
    background: #202b3d; border: 1px solid #2a3549;
    border-radius: 10px; padding: 20px; margin-bottom: 16px;
}

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

[data-testid="stImage"] img {
    border-radius: 8px !important; border: 1px solid #2a3549 !important; width: 100% !important;
    max-height: 320px !important; object-fit: cover !important;
}

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

.ai-body { font-size: 13.5px; line-height: 1.8; color: #9aa8bc; }
.ai-body strong, .ai-body b { color: #c8d4e4 !important; font-weight: 600 !important; }
.ai-body p { margin-bottom: 14px; }
.ai-body ol, .ai-body ul { padding-left: 18px; margin-bottom: 14px; }
.ai-body li { margin-bottom: 4px; }

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

.footer {
    margin: 24px -32px -32px -32px;
    background: #141b2d; border-top: 1px solid #2a3349;
    padding: 10px 32px; display: flex;
    align-items: center; justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #3a4460; letter-spacing: 0.04em;
}

[data-testid="stSpinner"] p { color: #7eb3ff !important; font-size: 13px !important; }

[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 2px solid #2a3349 !important;
    gap: 6px !important; margin-bottom: 20px !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important; font-weight: 600 !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important;
    color: #5a6890 !important; background: #1a2333 !important;
    border: 1px solid #2a3349 !important; padding: 10px 24px !important;
    border-radius: 6px 6px 0 0 !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #e8edf5 !important;
    border-color: #2563eb !important;
    border-bottom: 2px solid #1c2333 !important;
    background: #202b3d !important;
}
[data-testid="stTabs"] [role="tab"]:hover {
    color: #a0b8d8 !important; background: #1e2840 !important;
}

.info-box {
    background: #1a2435; border: 1px solid #2a3549; border-left: 3px solid #2563eb;
    border-radius: 6px; padding: 14px 16px; margin-bottom: 16px;
    font-size: 12.5px; color: #8a9ab0; line-height: 1.7;
}
.info-box strong { color: #c0cce0; }

.norm-panel-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
    color: #4a5470; text-align: center; margin-bottom: 6px;
}
.norm-metric-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 8px 0; border-bottom: 1px solid #242f42; font-size: 12px;
}
.norm-metric-row:last-child { border-bottom: none; }
.norm-metric-name { color: #7a8890; }
.norm-metric-val { font-family: 'IBM Plex Mono', monospace; font-size: 11px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
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


# ── Model ──────────────────────────────────────────────────────────────────────
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


# ── Grad-CAM ───────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Hooks into the last convolutional layer of the global encoder,
    computes class-discriminative activation maps via gradient weighting.
    Compatible with ResNet backbones as used in the vkola model.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output, _ = self.model.forward(
            input_tensor,
            torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE),
            [(0, 0)],
            (1.0, 1.0),
            mode=1
        )
        probs = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
        score = output[0, class_idx]
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Grad-CAM hooks did not fire — the target layer may not be "
                "part of the computation graph for this input."
            )

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze(0)
        cam = F.relu(cam)

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.cpu().numpy(), probs


def find_last_conv(module):
    """Recursively find the last Conv2d in a module."""
    last = None
    for _, m in module.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last


def compute_gradcam(img: Image.Image, target_class: int = None):
    """
    Grad-CAM on the global ResNet branch (layer4).
    Returns (heatmap_rgb, overlay_rgb, predicted_class, probs, cam_raw).
    """
    net = load_model()
    model_inner = net.module if hasattr(net, 'module') else net

    tensor = transform(img).unsqueeze(0).to(DEVICE)

    # First pass (no grad) to get class probabilities and resolve target_class.
    with torch.no_grad():
        out_ng, _ = model_inner.forward(
            tensor,
            torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE),
            [(0, 0)],
            (1.0, 1.0),
            mode=1
        )
    probs = torch.softmax(out_ng, dim=1)[0].cpu().numpy()
    if target_class is None:
        target_class = int(np.argmax(probs))

    # Scope the hook to resnet_global.layer4 so it fires during mode=1 inference.
    target_layer = None
    for name, m in model_inner.named_modules():
        if 'resnet_global' in name and name.endswith('layer4') and isinstance(m, torch.nn.Sequential):
            target_layer = m
            break
    if target_layer is None:
        global_branch = getattr(model_inner, 'resnet_global', model_inner)
        target_layer = find_last_conv(global_branch)
    if target_layer is None:
        raise RuntimeError("No suitable target layer found for Grad-CAM")

    gcam = GradCAM(model_inner, target_layer)
    try:
        cam_np, _ = gcam.generate(tensor, target_class)
    finally:
        gcam.remove()

    orig_w, orig_h = img.size
    cam_u8 = (cam_np * 255).clip(0, 255).astype(np.uint8)
    cam_resized = np.array(
        Image.fromarray(cam_u8).resize((orig_w, orig_h), Image.BICUBIC)
    ).astype(np.float32) / 255.0

    heatmap_u8    = (cam_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_rgb   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    orig_np = np.array(img.convert("RGB"))
    overlay = cv2.addWeighted(orig_np, 0.55, heatmap_rgb, 0.45, 0)

    return heatmap_rgb, overlay, target_class, probs, cam_resized


# ── Stain Normalisation ────────────────────────────────────────────────────────
def _get_stain_matrix(img_np: np.ndarray, lum_thr: float = 0.8):
    img = img_np.astype(np.float32) / 255.0 + 1e-6
    od  = -np.log(np.clip(img, 1e-6, 1.0))
    mask = (od * od).sum(axis=2) > (1 - lum_thr) ** 2
    od_m = od[mask].reshape(-1, 3)
    if len(od_m) < 10:
        return None
    _, _, Vt  = np.linalg.svd(od_m, full_matrices=False)
    plane     = Vt[:2]
    proj      = od_m @ plane.T
    angles    = np.arctan2(proj[:, 1], proj[:, 0])
    phi_min, phi_max = np.percentile(angles, 1), np.percentile(angles, 99)
    v1 = plane[0] * np.cos(phi_min) + plane[1] * np.sin(phi_min)
    v2 = plane[0] * np.cos(phi_max) + plane[1] * np.sin(phi_max)
    if v1[0] < v2[0]:
        v1, v2 = v2, v1
    return np.stack([v1 / (np.linalg.norm(v1) + 1e-8),
                     v2 / (np.linalg.norm(v2) + 1e-8)], axis=0)


def _get_concentrations(img_np: np.ndarray, S: np.ndarray):
    img  = img_np.astype(np.float32) / 255.0 + 1e-6
    od   = -np.log(np.clip(img, 1e-6, 1.0))
    h, w = od.shape[:2]
    return (od.reshape(-1, 3) @ np.linalg.pinv(S)).reshape(h, w, 2)


def macenko_normalise(source: Image.Image, reference: Image.Image) -> Image.Image:
    src_np, ref_np = np.array(source.convert("RGB")), np.array(reference.convert("RGB"))
    S_s = _get_stain_matrix(src_np)
    S_r = _get_stain_matrix(ref_np)
    if S_s is None or S_r is None:
        return source
    C_s    = _get_concentrations(src_np, S_s)
    C_r    = _get_concentrations(ref_np, S_r)
    max_s  = np.percentile(C_s.reshape(-1, 2), 99, axis=0) + 1e-8
    max_r  = np.percentile(C_r.reshape(-1, 2), 99, axis=0) + 1e-8
    C_norm = C_s * (max_r / max_s)
    h, w   = C_norm.shape[:2]
    od_norm  = (C_norm.reshape(-1, 2) @ S_r).reshape(h, w, 3)
    img_norm = np.clip((np.exp(-od_norm) - 1e-6) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img_norm)


def reinhard_normalise(source: Image.Image, reference: Image.Image) -> Image.Image:
    s  = np.array(source.convert("RGB"))    # uint8 — cvtColor needs [0,255] uint8 for correct LAB
    r  = np.array(reference.convert("RGB"))
    sl = cv2.cvtColor(s, cv2.COLOR_RGB2LAB).astype(np.float32)
    rl = cv2.cvtColor(r, cv2.COLOR_RGB2LAB).astype(np.float32)
    out = sl.copy()
    for ch in range(3):
        sm, ss = sl[:, :, ch].mean(), sl[:, :, ch].std() + 1e-6
        rm, rs = rl[:, :, ch].mean(), rl[:, :, ch].std() + 1e-6
        out[:, :, ch] = (sl[:, :, ch] - sm) / ss * rs + rm
    return Image.fromarray(cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB))


def vahadane_normalise(source: Image.Image, reference: Image.Image) -> Image.Image:
    """Full LAB-space histogram matching — transfers the complete colour distribution."""
    s  = np.array(source.convert("RGB"))    # uint8 — cvtColor needs [0,255] uint8 for correct LAB
    r  = np.array(reference.convert("RGB"))
    sl = cv2.cvtColor(s, cv2.COLOR_RGB2LAB).astype(np.float32)
    rl = cv2.cvtColor(r, cv2.COLOR_RGB2LAB).astype(np.float32)
    out = sl.copy()
    for ch in range(3):
        src_sorted = np.sort(sl[:, :, ch].flatten())
        ref_sorted = np.sort(rl[:, :, ch].flatten())
        # Map each source pixel to the matching quantile in the reference distribution
        out[:, :, ch] = np.interp(
            sl[:, :, ch].flatten(), src_sorted, ref_sorted
        ).reshape(sl[:, :, ch].shape)
    return Image.fromarray(cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB))


def compute_stain_metrics(img: Image.Image):
    np_img = np.array(img.convert("RGB"))   # uint8 — cvtColor needs [0,255] uint8 for correct LAB
    lab    = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB).astype(np.float32)
    return {
        "L* mean":  f"{lab[:,:,0].mean():.1f}",
        "A* mean":  f"{lab[:,:,1].mean():.1f}",
        "B* mean":  f"{lab[:,:,2].mean():.1f}",
        "Blue std": f"{np_img[:,:,2].std():.1f}",
        "Red mean": f"{np_img[:,:,0].mean():.1f}",
    }


# ── LLM Report ─────────────────────────────────────────────────────────────────
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
        per_image_summary += (
            f"  - Image {i+1}: {CLASS_NAMES[p]} ({CLASS_RANGE[p]}, "
            f"confidence {all_probs[i][p]*100:.1f}%)\n"
        )

    avg_breakdown = "\n".join(
        f"  - {CLASS_SHORT[i]}: {avg_probs[i]*100:.1f}%" for i in range(4)
    )

    multi_note = ""
    if n > 1:
        multi_note = (
            f"\nThis is a multi-image analysis ({n} biopsy images from the same patient). "
            f"Each image was graded independently; the consensus grade is derived from averaged "
            f"model probabilities across all images.\n\nPer-image grades:\n{per_image_summary}"
        )

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

    content_parts = [{"type": "text", "text": prompt}]
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {groq_key}"
    }
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [{"role": "user", "content": content_parts}],
        "max_tokens": 1800,
        "temperature": 0.3,
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ── Top Bar ────────────────────────────────────────────────────────────────────
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


# ── Session State ──────────────────────────────────────────────────────────────
for key in ["imgs", "all_probs", "all_preds"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Analysis",
    "Pathology Report",
    "Grad-CAM Explainability",
    "Stain Normalisation",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
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

            avg_probs      = np.mean(all_probs, axis=0)
            consensus_pred = int(np.argmax(avg_probs))
            consensus_conf = avg_probs[consensus_pred] * 100

            c  = CLASS_COLORS[consensus_pred]
            bg = CLASS_BG[consensus_pred]
            bo = CLASS_BORDER[consensus_pred]

            if n > 1:
                per_image_html = ""
                for i in range(n):
                    pi = all_preds[i]
                    pc = CLASS_COLORS[pi]
                    per_image_html += (
                        '<div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">'
                        + '<div style="font-family:\'IBM Plex Mono\',monospace; font-size:10px; color:#4a5470; width:56px; flex-shrink:0;">Image ' + str(i+1) + '</div>'
                        + '<div style="width:8px; height:8px; border-radius:50%; background:' + pc + '; flex-shrink:0;"></div>'
                        + '<div style="font-size:12px; color:' + pc + '; font-weight:600;">' + CLASS_NAMES[pi] + '</div>'
                        + '<div style="font-family:\'IBM Plex Mono\',monospace; font-size:10px; color:#4a5470; margin-left:4px;">' + f"{all_probs[i][pi]*100:.1f}" + '%</div>'
                        + '</div>'
                    )
                consensus_label = "CONSENSUS GRADE"
                per_image_block = '<div style="margin-bottom:12px;">' + per_image_html + '</div><div class="grade-divider"></div>'
            else:
                consensus_label = "FIBROSIS GRADE"
                per_image_block = ""

            grade_html = (
                '<div class="grade-card" style="background:' + bg + '; border-color:' + bo + ';">'
                + per_image_block
                + '<div class="grade-sublabel">' + consensus_label + '</div>'
                + '<div class="grade-name" style="color:' + c + ';">' + CLASS_NAMES[consensus_pred] + '</div>'
                + '<div class="grade-range">' + CLASS_RANGE[consensus_pred] + '</div>'
                + '<div class="grade-divider"></div>'
                + '<div class="grade-conf-row">'
                + '<div class="grade-conf-sublabel">Model Confidence</div>'
                + '<div class="grade-conf-value">' + f"{consensus_conf:.1f}" + '%</div>'
                + '</div></div>'
            )
            st.markdown(grade_html, unsafe_allow_html=True)

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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PATHOLOGY REPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if (st.session_state.imgs is not None
            and st.session_state.all_probs is not None
            and st.session_state.all_preds is not None):

        all_probs      = st.session_state.all_probs
        all_preds      = st.session_state.all_preds
        avg_probs      = np.mean(all_probs, axis=0)
        consensus_pred = int(np.argmax(avg_probs))
        consensus_conf = avg_probs[consensus_pred] * 100

        st.markdown("""
<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:18px;">
    <div style="font-family:'Playfair Display',serif; font-size:18px; font-weight:700; color:#e0e6f0;">Pathology Report</div>
    <div style="font-family:'IBM Plex Mono',monospace; font-size:9px; font-weight:500; letter-spacing:0.08em;
                background:#1a2d1a; color:#6ee7b7; border:1px solid #1a4a35; padding:4px 10px; border-radius:4px;">
        LLAMA 4 SCOUT &nbsp;·&nbsp; VISION</div>
</div>
""", unsafe_allow_html=True)

        rcol1, rcol2 = st.columns([1, 2.4], gap="large")

        with rcol1:
            for i, im in enumerate(st.session_state.imgs):
                caption = (
                    f"Image {i+1} — {CLASS_NAMES[all_preds[i]]}"
                    if len(st.session_state.imgs) > 1
                    else "Analyzed biopsy image"
                )
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
    else:
        st.markdown("""
<div class="await-wrap">
    <div class="await-label">No Report Yet</div>
    <div class="await-sub">Run an analysis in the Analysis tab first</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — GRAD-CAM EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
    <div style="font-family:'Playfair Display',serif; font-size:18px; font-weight:700; color:#e0e6f0;">
        Grad-CAM Explainability
    </div>
    <div style="font-family:'IBM Plex Mono',monospace; font-size:9px; font-weight:500; letter-spacing:0.08em;
                background:#2a1a4a; color:#c084fc; border:1px solid #4a2a80; padding:4px 10px; border-radius:4px;">
        GRADIENT-WEIGHTED CLASS ACTIVATION MAPS
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="info-box">
    <strong>How it works:</strong> Grad-CAM backpropagates the gradient of the predicted class score through
    the final convolutional layer of the global ResNet backbone (layer4), then globally average-pools those
    gradients to weight each feature map. The resulting spatial map highlights regions that most strongly
    drove the prediction — giving pathologists an interpretable explanation of the model's decision.<br><br>
    <strong>Colour scale:</strong>
    <span style="color:#4466ff;">Blue</span> = low attention &nbsp;|&nbsp;
    <span style="color:#00cc88;">Green</span> = moderate &nbsp;|&nbsp;
    <span style="color:#ffbb00;">Yellow</span> = high &nbsp;|&nbsp;
    <span style="color:#ff3333;">Red</span> = peak discriminative region.
</div>
""", unsafe_allow_html=True)

    if st.session_state.imgs is not None:
        gcam_imgs = st.session_state.imgs
        st.markdown('<div class="sec-label">Using images from Analysis tab</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sec-label">Upload Biopsy Image</div>', unsafe_allow_html=True)
        gcam_upload = st.file_uploader(
            "Upload a biopsy image for Grad-CAM",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            key="gcam_upload",
            label_visibility="collapsed",
        )
        gcam_imgs = [Image.open(gcam_upload).convert("RGB")] if gcam_upload else None

    if gcam_imgs is not None:
        img_idx = 0
        if len(gcam_imgs) > 1:
            img_idx = st.selectbox(
                "Select image", list(range(len(gcam_imgs))),
                format_func=lambda x: f"Image {x+1}", key="gcam_img_sel"
            )
        selected_img = gcam_imgs[img_idx]

        gcam_ctrl, gcam_alpha_col = st.columns([2, 1])
        with gcam_ctrl:
            target_class_name = st.selectbox(
                "Explain prediction for class", CLASS_NAMES,
                index=0, key="gcam_class_sel",
                help="Which fibrosis grade should the heatmap explain?"
            )
            target_class_idx = CLASS_NAMES.index(target_class_name)
        with gcam_alpha_col:
            overlay_alpha = st.slider("Overlay opacity", 0.1, 0.9, 0.45, 0.05, key="gcam_alpha")

        # Only recompute when the image or target class changes; the slider just reblends.
        # Use a small-thumbnail MD5 as a stable, content-based image fingerprint.
        _thumb = selected_img.resize((16, 16)).tobytes()
        gcam_img_key = hashlib.md5(_thumb).hexdigest()
        gcam_needs_recompute = (
            st.session_state.get("gcam_last_img_key") != gcam_img_key
            or st.session_state.get("gcam_last_class") != target_class_idx
        )

        if gcam_needs_recompute:
            with st.spinner("Computing Grad-CAM..."):
                try:
                    heatmap_rgb, _, pred_class, probs, cam_raw = compute_gradcam(
                        selected_img, target_class=target_class_idx
                    )
                    st.session_state["gcam_last_img_key"] = gcam_img_key
                    st.session_state["gcam_last_class"]   = target_class_idx
                    st.session_state["gcam_cache"] = (heatmap_rgb, pred_class, probs, cam_raw)
                except Exception as e:
                    st.error(f"Grad-CAM error: {str(e)}")
                    st.stop()

        if "gcam_cache" not in st.session_state:
            st.error("Grad-CAM computation failed to initialise. Please refresh the page.")
            st.stop()

        heatmap_rgb, pred_class, probs, cam_raw = st.session_state["gcam_cache"]
        orig_np    = np.array(selected_img)
        overlay_np = cv2.addWeighted(orig_np, 1 - overlay_alpha, heatmap_rgb, overlay_alpha, 0)

        g1, g2, g3 = st.columns(3, gap="medium")
        with g1:
            st.markdown('<div class="norm-panel-label">Original Image</div>', unsafe_allow_html=True)
            st.image(selected_img, use_column_width=True)
        with g2:
            st.markdown('<div class="norm-panel-label">Grad-CAM Heatmap</div>', unsafe_allow_html=True)
            st.image(heatmap_rgb, use_column_width=True)
        with g3:
            st.markdown('<div class="norm-panel-label">Overlay</div>', unsafe_allow_html=True)
            st.image(overlay_np, use_column_width=True)

        st.markdown('<div class="sec-label" style="margin-top:20px;">Activation Statistics</div>', unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)

        def stat_card(label, value, sub, color):
            return (
                '<div class="card" style="text-align:center; padding:14px;">'
                + '<div class="grade-sublabel">' + label + '</div>'
                + '<div style="font-family:Playfair Display,serif; font-size:20px; font-weight:700; color:' + color + ';">' + str(value) + '</div>'
                + '<div style="font-family:IBM Plex Mono,monospace; font-size:11px; color:#5a6480; margin-top:4px;">' + sub + '</div>'
                + '</div>'
            )

        with s1:
            st.markdown(stat_card("Predicted Class", CLASS_NAMES[pred_class],
                f"{probs[pred_class]*100:.1f}% confidence", CLASS_COLORS[pred_class]), unsafe_allow_html=True)
        with s2:
            st.markdown(stat_card("Explained Class", CLASS_NAMES[target_class_idx],
                f"{probs[target_class_idx]*100:.1f}% probability", CLASS_COLORS[target_class_idx]), unsafe_allow_html=True)
        with s3:
            st.markdown(stat_card("Peak Activation", f"{cam_raw.max()*100:.0f}%",
                "of normalised range", "#c084fc"), unsafe_allow_html=True)
        with s4:
            hot = float((cam_raw > 0.5).mean()) * 100
            st.markdown(stat_card("High-Attention Area", f"{hot:.1f}%",
                "pixels > 0.5 threshold", "#fbbf24"), unsafe_allow_html=True)

        st.markdown('<div class="sec-label" style="margin-top:4px;">All Class Probabilities</div>', unsafe_allow_html=True)
        for i in range(4):
            pct = probs[i] * 100
            st.markdown(f"""
<div class="prob-row">
    <div class="prob-name">{CLASS_NAMES[i]}</div>
    <div class="prob-track"><div class="prob-fill" style="width:{pct:.1f}%; background:{CLASS_COLORS[i]};"></div></div>
    <div class="prob-pct">{pct:.1f}%</div>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div class="await-wrap">
    <div class="await-label">No Image Loaded</div>
    <div class="await-sub">Upload images in the Analysis tab first, or use the uploader above.</div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — STAIN NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
    <div style="font-family:'Playfair Display',serif; font-size:18px; font-weight:700; color:#e0e6f0;">
        Stain Normalisation
    </div>
    <div style="font-family:'IBM Plex Mono',monospace; font-size:9px; font-weight:500; letter-spacing:0.08em;
                background:#102a20; color:#6ee7b7; border:1px solid #1a4a35; padding:4px 10px; border-radius:4px;">
        CROSS-SITE COLOUR CALIBRATION
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="info-box">
    <strong>Why normalise?</strong> Trichrome staining intensity varies across laboratories due to differences
    in reagent batches, protocols, and scanner calibration. A biopsy from a different site may have a shifted
    colour distribution at the same fibrosis grade, causing the model to mis-predict. Normalising the source
    image to a reference stain profile before inference reduces this domain shift and can improve cross-site accuracy.<br><br>
    <strong>Methods:</strong> <strong>Macenko</strong> (SVD stain separation — most accurate for trichrome) &nbsp;|&nbsp;
    <strong>Reinhard</strong> (LAB channel statistics transfer — fast) &nbsp;|&nbsp;
    <strong>Vahadane</strong> (LAB histogram matching — robust to outliers).
</div>
""", unsafe_allow_html=True)

    sn_left, sn_right = st.columns(2, gap="large")

    with sn_left:
        st.markdown('<div class="sec-label">Source Image (to normalise)</div>', unsafe_allow_html=True)
        if st.session_state.imgs is not None:
            src_opts = {f"Analysis Image {i+1}": im for i, im in enumerate(st.session_state.imgs)}
            src_opts["Upload new image..."] = None
        else:
            src_opts = {"Upload new image...": None}

        src_choice = st.selectbox("Source", list(src_opts.keys()), key="sn_src_choice")
        if src_opts[src_choice] is None:
            sn_src_upload = st.file_uploader(
                "Upload source image", type=["jpg", "jpeg", "png"],
                key="sn_src_upload", label_visibility="collapsed"
            )
            sn_source = Image.open(sn_src_upload).convert("RGB") if sn_src_upload else None
        else:
            sn_source = src_opts[src_choice]

        if sn_source:
            st.image(sn_source, use_column_width=True, caption="Source (original)")

    with sn_right:
        st.markdown('<div class="sec-label">Reference Image (target stain)</div>', unsafe_allow_html=True)
        sn_ref_upload = st.file_uploader(
            "Upload reference image", type=["jpg", "jpeg", "png"],
            key="sn_ref_upload", label_visibility="collapsed"
        )
        sn_reference = Image.open(sn_ref_upload).convert("RGB") if sn_ref_upload else None
        if sn_reference:
            st.image(sn_reference, use_column_width=True, caption="Reference (target lab)")

    if sn_source and sn_reference:
        st.markdown("---")
        m_col, r_col = st.columns([2, 1])
        with m_col:
            norm_method = st.selectbox(
                "Normalisation method",
                ["Macenko (SVD stain separation)", "Reinhard (LAB statistics)", "Vahadane (LAB histogram)"],
                key="sn_method"
            )
        with r_col:
            run_norm = st.button("▶  Run Normalisation", use_container_width=True, key="sn_run")

        if run_norm:
            with st.spinner("Normalising stain..."):
                try:
                    if "Macenko" in norm_method:
                        result_img = macenko_normalise(sn_source, sn_reference)
                    elif "Reinhard" in norm_method:
                        result_img = reinhard_normalise(sn_source, sn_reference)
                    else:
                        result_img = vahadane_normalise(sn_source, sn_reference)
                    st.session_state["sn_result"]      = result_img
                    st.session_state["sn_method_used"] = norm_method
                except Exception as e:
                    st.error(f"Normalisation error: {str(e)}")
                    st.stop()

        if "sn_result" in st.session_state:
            result_img = st.session_state["sn_result"]

            st.markdown('<div class="sec-label" style="margin-top:8px;">Visual Comparison</div>', unsafe_allow_html=True)
            r1, r2, r3 = st.columns(3, gap="medium")
            with r1:
                st.markdown('<div class="norm-panel-label">Source (Original)</div>', unsafe_allow_html=True)
                st.image(sn_source, use_column_width=True)
            with r2:
                st.markdown('<div class="norm-panel-label">Reference</div>', unsafe_allow_html=True)
                st.image(sn_reference, use_column_width=True)
            with r3:
                st.markdown('<div class="norm-panel-label">Normalised Output</div>', unsafe_allow_html=True)
                st.image(result_img, use_column_width=True)

            st.markdown('<div class="sec-label" style="margin-top:20px;">Stain Statistics</div>', unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3, gap="medium")
            src_m = compute_stain_metrics(sn_source)
            ref_m = compute_stain_metrics(sn_reference)
            res_m = compute_stain_metrics(result_img)

            def metric_card(title, metrics, accent):
                rows = "".join(
                    '<div class="norm-metric-row">'
                    + '<span class="norm-metric-name">' + k + '</span>'
                    + '<span class="norm-metric-val" style="color:' + accent + ';">' + v + '</span>'
                    + '</div>'
                    for k, v in metrics.items()
                )
                return (
                    '<div class="card" style="padding:14px 16px;">'
                    + '<div class="norm-panel-label" style="margin-bottom:10px;">' + title + '</div>'
                    + rows + '</div>'
                )

            with m1:
                st.markdown(metric_card("Source", src_m, "#7eb3ff"), unsafe_allow_html=True)
            with m2:
                st.markdown(metric_card("Reference", ref_m, "#6ee7b7"), unsafe_allow_html=True)
            with m3:
                st.markdown(metric_card("Normalised", res_m, "#c084fc"), unsafe_allow_html=True)

            st.markdown('<div class="sec-label" style="margin-top:4px;">Inference Comparison</div>', unsafe_allow_html=True)
            with st.spinner("Running inference on both images..."):
                try:
                    pb     = predict(sn_source)
                    pa     = predict(result_img)
                    pred_b = int(np.argmax(pb))
                    pred_a = int(np.argmax(pa))

                    ic1, ic2 = st.columns(2, gap="large")

                    def infer_card(label, probs, pred):
                        c, bg, bo = CLASS_COLORS[pred], CLASS_BG[pred], CLASS_BORDER[pred]
                        bars = "".join(
                            '<div class="prob-row">'
                            + '<div class="prob-name">' + CLASS_NAMES[i] + '</div>'
                            + '<div class="prob-track"><div class="prob-fill" style="width:' + f"{probs[i]*100:.1f}" + '%; background:' + CLASS_COLORS[i] + ';"></div></div>'
                            + '<div class="prob-pct">' + f"{probs[i]*100:.1f}" + '%</div>'
                            + '</div>'
                            for i in range(4)
                        )
                        return (
                            '<div style="font-family:IBM Plex Mono,monospace; font-size:10px; color:#4a5470; font-weight:700; letter-spacing:0.1em; margin-bottom:6px;">' + label + '</div>'
                            + '<div class="grade-card" style="background:' + bg + '; border-color:' + bo + '; margin-bottom:12px;">'
                            + '<div class="grade-sublabel">PREDICTION</div>'
                            + '<div class="grade-name" style="color:' + c + ';">' + CLASS_NAMES[pred] + '</div>'
                            + '<div class="grade-range">' + CLASS_RANGE[pred] + '</div>'
                            + '<div class="grade-divider"></div>'
                            + '<div class="grade-conf-row">'
                            + '<div class="grade-conf-sublabel">Confidence</div>'
                            + '<div class="grade-conf-value">' + f"{probs[pred]*100:.1f}" + '%</div>'
                            + '</div></div>'
                            + bars
                        )

                    with ic1:
                        st.markdown(infer_card("BEFORE NORMALISATION", pb, pred_b), unsafe_allow_html=True)
                    with ic2:
                        st.markdown(infer_card("AFTER NORMALISATION", pa, pred_a), unsafe_allow_html=True)

                    if pred_b == pred_a:
                        delta = (pa[pred_a] - pb[pred_b]) * 100
                        sign  = "+" if delta >= 0 else ""
                        col   = "#6ee7b7" if delta >= 0 else "#f87171"
                        st.markdown(
                            '<div class="info-box">Prediction unchanged: <strong>'
                            + CLASS_NAMES[pred_b] + '</strong>. Confidence shift: <strong style="color:'
                            + col + ';">' + sign + f"{delta:.1f}" + 'pp</strong>.</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div class="info-box" style="border-left-color:#f59e0b;">'
                            + '&#9888; Prediction <strong>changed</strong> from '
                            + '<strong style="color:' + CLASS_COLORS[pred_b] + ';">' + CLASS_NAMES[pred_b] + '</strong> to '
                            + '<strong style="color:' + CLASS_COLORS[pred_a] + ';">' + CLASS_NAMES[pred_a] + '</strong>'
                            + ' after normalisation. This indicates the original stain profile was influencing '
                            + 'the model — normalisation may yield a more accurate cross-site result.</div>',
                            unsafe_allow_html=True
                        )
                except Exception as e:
                    st.warning(f"Inference comparison unavailable: {str(e)}")

    elif sn_source and not sn_reference:
        st.markdown("""
<div class="await-wrap" style="margin-top:20px;">
    <div class="await-label">Reference Image Required</div>
    <div class="await-sub">Upload a reference biopsy from your target lab or scanner.</div>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div class="await-wrap" style="margin-top:20px;">
    <div class="await-label">Upload Images to Begin</div>
    <div class="await-sub">Provide a source biopsy to normalise and a reference image from the target lab.<br>
    The tool will transfer the reference stain profile and compare grade predictions before and after.</div>
</div>""", unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <span>KIDNEY FIBROSIS GRADER &nbsp;·&nbsp; ResNet-FPN &nbsp;·&nbsp; 95% Test Accuracy</span>
    <span>Research use only — not validated for clinical diagnosis</span>
</div>
""", unsafe_allow_html=True)
