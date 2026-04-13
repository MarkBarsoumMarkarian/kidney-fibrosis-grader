import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import sys, os
import re
import time
import requests
import base64
import io
import cv2
import hashlib
import tempfile
import json
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.model_builder import model as build_model
import gdown

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'global_only.pth')
if not os.path.exists(MODEL_PATH):
    print('Downloading model weights...')
    gdown.download('https://drive.google.com/uc?id=1KvJQ0YKL-I96UJ5zUGLR_Qpd4R0ach5t', MODEL_PATH, quiet=False)
    print('Done.')

IF_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'if_classifier_best.pth')
if not os.path.exists(IF_MODEL_PATH):
    print('Downloading IF classifier weights...')
    gdown.download('https://drive.google.com/uc?id=1JVGcgp8cxc5ZSnHGXTCX9Ez2fI3MEk9R', IF_MODEL_PATH, quiet=False)
    print('Done.')

try:
    from inference import IFClassifier, IF_CHANNELS, IF_CLASSES, CLASS_DISPLAY as IF_CLASS_DISPLAY
    _IF_MODULE_OK = True
except Exception as _if_import_err:
    print(f"WARNING: Could not import inference module: {_if_import_err}")
    IFClassifier    = None
    IF_CHANNELS     = ['IgG', 'IgA', 'IgM', 'C3', 'C1q', 'kappa', 'lambda', 'fibrinogen', 'albumin']
    IF_CLASSES      = []
    IF_CLASS_DISPLAY = {}
    _IF_MODULE_OK   = False

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

html, body, [data-testid="stAppViewContainer"] {
    scroll-behavior: auto !important;
    overflow-y: scroll !important;
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
section.main,
[data-testid="stMain"] {
    background: #111827 !important;
    color: #d1d9e6 !important;
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
    background: linear-gradient(90deg, #0d1525 0%, #111827 60%, #0d1525 100%);
    border-bottom: 1px solid #1e2d45;
    padding: 0 32px;
    height: 58px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 0 -32px 28px -32px;
    box-shadow: 0 1px 12px rgba(0,0,0,0.35);
}
.topbar-brand { display: flex; align-items: center; gap: 14px; }
.topbar-logo {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
    border-radius: 9px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; font-weight: 700; color: #fff;
    display: flex; align-items: center; justify-content: center;
    letter-spacing: -0.5px;
    box-shadow: 0 2px 8px rgba(99,102,241,0.4);
}
.topbar-name {
    font-family: 'Playfair Display', serif;
    font-size: 17px; font-weight: 700; color: #f0f4ff;
}
.topbar-desc { font-size: 10px; color: #4a5880; margin-top: 2px; letter-spacing: 0.03em; }
.topbar-pills { display: flex; gap: 8px; }
.tpill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; font-weight: 500;
    padding: 4px 11px; border-radius: 5px;
    letter-spacing: 0.04em; border: 1px solid;
}
.tpill-blue  { color: #93c5fd; border-color: #1e3a6e; background: #0f1f40; }
.tpill-green { color: #6ee7b7; border-color: #14472e; background: #0a2218; }
.tpill-amber { color: #fcd34d; border-color: #4a3010; background: #261808; }

.sec-label {
    font-size: 10px; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: #3d4f6e; margin-bottom: 10px; margin-top: 2px;
}

.card {
    background: #172033; border: 1px solid #1e2d45;
    border-radius: 12px; padding: 20px; margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}

[data-testid="stFileUploader"] { background: transparent !important; }
[data-testid="stFileUploadDropzone"] {
    background: #131e30 !important;
    border: 2px dashed #1e3050 !important;
    border-radius: 10px !important;
    padding: 28px !important; transition: all 0.2s !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #3b82f6 !important; background: #131f3a !important;
}
[data-testid="stFileUploadDropzone"] p { color: #4a5880 !important; font-size: 13px !important; }
[data-testid="stFileUploadDropzone"] small { color: #303d55 !important; }
[data-testid="stFileUploadDropzone"] svg { fill: #1e3050 !important; }

[data-testid="stImage"] img {
    border-radius: 10px !important; border: 1px solid #1e2d45 !important; width: 100% !important;
    max-height: 320px !important; object-fit: cover !important;
}

.grade-card {
    border-radius: 12px; padding: 20px 22px; border: 1px solid; margin-bottom: 20px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25);
}
.grade-name {
    font-family: 'Playfair Display', serif;
    font-size: 28px; font-weight: 700; line-height: 1.1;
}
.grade-sublabel {
    font-size: 10px; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #3d4f6e; margin-bottom: 4px;
}
.grade-range {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; color: #6a7888; margin-top: 3px;
}
.grade-divider { height: 1px; background: rgba(255,255,255,0.05); margin: 14px 0; }
.grade-conf-row { display: flex; align-items: center; justify-content: space-between; }
.grade-conf-sublabel {
    font-size: 10px; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #3d4f6e;
}
.grade-conf-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 14px; font-weight: 500; color: #7a8aaa;
}

.prob-row { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
.prob-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #4a5880; width: 62px; flex-shrink: 0; font-weight: 500;
}
.prob-track { flex: 1; height: 6px; background: #131e30; border-radius: 3px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 3px; }
.prob-pct {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #4a5880; width: 38px; text-align: right; flex-shrink: 0;
}

.ref-row {
    display: flex; align-items: center; gap: 10px;
    padding: 7px 0; border-bottom: 1px solid #1a2535;
    font-size: 12px; color: #7a8aaa;
}
.ref-row:last-child { border-bottom: none; }
.ref-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.ref-grade { font-weight: 600; color: #9aa8c0; width: 64px; flex-shrink: 0; }
.ref-range {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #3d4f6e; width: 60px; flex-shrink: 0;
}
.ref-desc { color: #485868; font-size: 11px; }

.ai-body { font-size: 13.5px; line-height: 1.85; color: #8a9ab8; }
.ai-body strong, .ai-body b { color: #c0cede !important; font-weight: 600 !important; }
.ai-body p { margin-bottom: 14px; }
.ai-body ol, .ai-body ul { padding-left: 18px; margin-bottom: 14px; }
.ai-body li { margin-bottom: 4px; }

.await-wrap {
    background: #172033; border: 1px solid #1e2d45;
    border-radius: 12px; padding: 44px 20px; text-align: center;
}
.await-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; font-weight: 500; letter-spacing: 0.1em;
    text-transform: uppercase; color: #2a3a56; margin-bottom: 8px;
}
.await-sub { font-size: 12px; color: #2a3a56; line-height: 1.6; }

.footer {
    margin: 24px -32px -32px -32px;
    background: linear-gradient(90deg, #0d1525 0%, #111827 60%, #0d1525 100%);
    border-top: 1px solid #1e2d45;
    padding: 11px 32px; display: flex;
    align-items: center; justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #2e3e58; letter-spacing: 0.04em;
}

[data-testid="stSpinner"] p { color: #93c5fd !important; font-size: 13px !important; }

[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 2px solid #1a2840 !important;
    gap: 6px !important; margin-bottom: 20px !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important; font-weight: 600 !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important;
    color: #3d5070 !important; background: #131e30 !important;
    border: 1px solid #1a2840 !important; padding: 10px 24px !important;
    border-radius: 6px 6px 0 0 !important; transition: all 0.15s !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #e8eeff !important;
    border-color: #3b82f6 !important;
    border-bottom: 2px solid #111827 !important;
    background: #172033 !important;
}
[data-testid="stTabs"] [role="tab"]:hover {
    color: #8ab4d8 !important; background: #162038 !important;
}

.info-box {
    background: #131e30; border: 1px solid #1e2d45; border-left: 3px solid #3b82f6;
    border-radius: 8px; padding: 14px 16px; margin-bottom: 16px;
    font-size: 12.5px; color: #7a8aaa; line-height: 1.75;
}
.info-box strong { color: #a8bcda; }

.norm-panel-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
    color: #3d4f6e; text-align: center; margin-bottom: 6px;
}
.norm-metric-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 8px 0; border-bottom: 1px solid #192538; font-size: 12px;
}
.norm-metric-row:last-child { border-bottom: none; }
.norm-metric-name { color: #6a7888; }
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
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Ordered fallback list for the full clinicopathological report (multimodal)
REPORT_MODELS = [
    "meta-llama/llama-4-maverick:free",
    "meta-llama/llama-4-scout:free",
    "google/gemma-4-31b-it:free",
]

# Ordered fallback list for the IF panel safety review
IF_REVIEW_MODELS = [
    "google/gemma-4-31b-it:free",
    "meta-llama/llama-4-scout:free",
    "meta-llama/llama-4-maverick:free",
]
# Thumbnail size used to generate stable MD5 fingerprints for IF channel images
_IF_HASH_THUMB_SIZE = (4, 4)


# ── Model ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    net, _ = build_model(N_CLASS, mode=MODE, evaluation=True, path_g=MODEL_PATH)
    net.eval()
    return net


@st.cache_resource
def load_if_model():
    if not _IF_MODULE_OK or IFClassifier is None:
        return None
    if not os.path.exists(IF_MODEL_PATH):
        return None
    return IFClassifier(IF_MODEL_PATH)


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



# ── IF Grad-CAM ────────────────────────────────────────────────────────────────

def compute_if_gradcam(
    channel_img: Image.Image,
    channel_name: str,
    predicted_class_idx: int,
) -> "Image.Image | None":
    """
    Compute a Grad-CAM activation overlay for a single IF channel image using the
    IF ResNet-50 classifier (IFClassifier).

    The channel is placed at its correct 9-channel position; all other channels are
    zero-filled — exactly as ``IFClassifier._build_stack`` does for inference.
    Targets ``model.layer4`` (last residual block) for gradient attribution.

    Args:
        channel_img:         PIL image for the single IF channel.
        channel_name:        Name of the IF channel (e.g. ``"IgA"``).
        predicted_class_idx: Class index returned by the classifier.

    Returns:
        PIL overlay image (same size as *channel_img*), or ``None`` on failure.
    """
    clf = load_if_model()
    if clf is None:
        return None
    if channel_name not in IF_CHANNELS:
        return None

    device = clf.device
    size = 224

    # Build 9-channel tensor — only the target channel is populated.
    arr = np.array(
        channel_img.convert("L").resize((size, size), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    nz = arr[arr > 0.05]
    if len(nz) > 50:
        arr = (arr - nz.mean()) / (nz.std() + 1e-6)

    # Quality gate: if less than 5% of pixels have meaningful signal,
    # the channel is too dark/uniform for reliable Grad-CAM
    signal_fraction = np.sum(arr > 0.05) / arr.size
    if signal_fraction < 0.05:
        return None  # skip Grad-CAM for this channel — signal too weak

    planes = [
        arr if ch == channel_name else np.zeros((size, size), dtype=np.float32)
        for ch in IF_CHANNELS
    ]
    tensor = torch.from_numpy(np.stack(planes, axis=0)).unsqueeze(0).to(device)

    # Locate layer4 in the ResNet-50 backbone.
    target_layer = None
    for name, m in clf.model.named_modules():
        if name == "layer4" and isinstance(m, torch.nn.Sequential):
            target_layer = m
            break
    if target_layer is None:
        return None

    activations_ref: list = [None]
    gradients_ref: list = [None]

    def _save_act(module, inp, out):
        activations_ref[0] = out.detach()

    def _save_grad(module, grad_in, grad_out):
        gradients_ref[0] = grad_out[0].detach()

    fwd_h = target_layer.register_forward_hook(_save_act)
    bwd_h = target_layer.register_full_backward_hook(_save_grad)
    try:
        clf.model.eval()
        clf.model.zero_grad()
        with torch.enable_grad():
            output = clf.model(tensor)
            score = output[0, predicted_class_idx]
            score.backward()
    finally:
        fwd_h.remove()
        bwd_h.remove()

    if activations_ref[0] is None or gradients_ref[0] is None:
        return None

    weights = gradients_ref[0].mean(dim=[2, 3], keepdim=True)
    cam = (weights * activations_ref[0]).sum(dim=1).squeeze(0)
    cam = F.relu(cam)
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = torch.zeros_like(cam)

    cam_np = cam.cpu().numpy()

    # Check if activation is inside tissue vs outside
    # Tissue mask: original array pixels with signal > 0.05
    tissue_mask = cv2.resize(
        (arr > 0.05).astype(np.uint8),
        (cam_np.shape[1], cam_np.shape[0]),
    )
    cam_in_tissue = cam_np[tissue_mask == 1].mean() if tissue_mask.sum() > 0 else 0
    cam_outside_tissue = cam_np[tissue_mask == 0].mean() if (tissue_mask == 0).sum() > 0 else 0

    # If activation outside tissue is stronger than inside, skip this channel
    if cam_outside_tissue > cam_in_tissue:
        return None  # peripheral/artifact activation — not clinically meaningful

    orig_w, orig_h = channel_img.size
    cam_u8 = (cam_np * 255).clip(0, 255).astype(np.uint8)
    cam_resized = (
        np.array(
            Image.fromarray(cam_u8).resize((orig_w, orig_h), Image.BICUBIC)
        ).astype(np.float32)
        / 255.0
    )

    heatmap_u8 = (cam_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    orig_np = np.array(channel_img.convert("RGB"))
    overlay = cv2.addWeighted(orig_np, 0.55, heatmap_rgb, 0.45, 0)
    return Image.fromarray(overlay)

_MOSAIC_CELL = 224   # px per cell
_MOSAIC_COLS = 3
_MOSAIC_ROWS = 3


def pil_to_base64(img: Image.Image, quality: int = 85) -> str:
    """Convert a PIL Image to a base64-encoded JPEG string."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _gray_normalized_from_pil(pil_img: Image.Image, size: int = _MOSAIC_CELL) -> np.ndarray:
    """Apply the same grayscale normalisation as load_gray_normalized but from a PIL Image."""
    arr = np.array(pil_img.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    nz = arr[arr > 0.05]
    if len(nz) > 50:
        arr = (arr - nz.mean()) / (nz.std() + 1e-6)
    return arr


def build_if_mosaic(channel_imgs: dict) -> Image.Image:
    """
    Build a 3×3 RGB grid of the 9 IF channels in the order defined by IF_CHANNELS.
    Each cell is _MOSAIC_CELL × _MOSAIC_CELL pixels.

    channel_imgs: dict[str, list[PIL.Image]] — one or more PIL images per channel.
    Missing channels are filled with a black square labelled "MISSING".
    Channels with multiple images are pixel-wise averaged (grayscale-normalised) and
    the count is appended to the label, e.g. "IgG (×3)".
    Channel names are drawn in white in the top-left corner of each cell.
    """
    cell = _MOSAIC_CELL
    canvas_w = cell * _MOSAIC_COLS
    canvas_h = cell * _MOSAIC_ROWS
    mosaic = Image.new("RGB", (canvas_w, canvas_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(mosaic)

    # Attempt to load a small built-in font; fall back to default if unavailable
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for idx, ch in enumerate(IF_CHANNELS):
        row = idx // _MOSAIC_COLS
        col = idx % _MOSAIC_COLS
        x0, y0 = col * cell, row * cell

        imgs = channel_imgs.get(ch)
        if imgs:
            n = len(imgs)
            if n == 1:
                cell_img = imgs[0].convert("RGB").resize((cell, cell), Image.BILINEAR)
                label = ch
            else:
                # Average all images as grayscale normalised float32 arrays
                arrays = [_gray_normalized_from_pil(pil_img, size=cell) for pil_img in imgs]
                avg = np.mean(arrays, axis=0)
                # Rescale to [0, 255] for display
                lo, hi = avg.min(), avg.max()
                if hi > lo:
                    avg = (avg - lo) / (hi - lo) * 255.0
                else:
                    avg = np.zeros_like(avg)
                cell_img = Image.fromarray(avg.astype(np.uint8), mode="L").convert("RGB")
                label = f"{ch} (\u00d7{n})"
            mosaic.paste(cell_img, (x0, y0))
        else:
            # Black square already present; just add label
            label = f"{ch}\nMISSING"

        # White label with a thin dark shadow for readability
        draw.text((x0 + 5, y0 + 4), label, fill=(50, 50, 50), font=font)
        draw.text((x0 + 4, y0 + 3), label, fill=(255, 255, 255), font=font)

    return mosaic


def get_api_keys() -> list:
    keys = []
    for i in range(1, 20):
        key = os.environ.get(f"OPENROUTER_API_KEY_{i}", "")
        if not key:
            try:
                key = st.secrets.get(f"OPENROUTER_API_KEY_{i}", "")
            except Exception:
                pass
        if key and key.strip().startswith("sk-"):
            keys.append(key.strip())

    print(f"DEBUG: found {len(keys)} numbered keys")  # ADD THIS

    if not keys:
        single = os.environ.get("OPENROUTER_API_KEY", "")
        if not single:
            try:
                single = st.secrets.get("OPENROUTER_API_KEY", "")
            except Exception:
                pass
        if single:
            keys.append(single.strip())

    print(f"DEBUG: total keys returned: {len(keys)}")  # ADD THIS

    random.shuffle(keys)
    return keys


def get_groq_keys() -> list:
    keys = []
    for i in range(1, 20):
        key = os.environ.get(f"GROQ_API_KEY_{i}", "")
        if not key:
            try:
                key = st.secrets.get(f"GROQ_API_KEY_{i}", "")
            except Exception:
                pass
        if key and key.strip().startswith("gsk_"):
            keys.append(key.strip())
    if not keys:
        single = os.environ.get("GROQ_API_KEY", "")
        if not single:
            try:
                single = st.secrets.get("GROQ_API_KEY", "")
            except Exception:
                pass
        if single:
            keys.append(single.strip())
    random.shuffle(keys)
    return keys


def llm_review_if_panel(mosaic_b64: str, top_predictions: list, channels_used: list,
                        multi_channel_notes: str = "") -> str:
    """
    Send the IF mosaic to OpenRouter (Gemma 4 31B free) for a concise
    nephropathologist-style safety review.

    Returns the response text, or an error string on failure.
    """
    api_keys = get_api_keys()
    if not api_keys:
        return "⚠️ OPENROUTER_API_KEY not configured — LLM review skipped."

    top = top_predictions[0] if top_predictions else {}
    diag_label  = top.get("display", "Unknown")
    diag_pct    = top.get("pct", 0.0)
    ch_str      = ", ".join(channels_used) if channels_used else "none"

    prompt_text = (
        "You are an expert nephropathologist reviewing an immunofluorescence (IF) panel.\n"
        f"The image is a 3×3 mosaic of 9 IF channels in this order: "
        f"IgG, IgA, IgM (row 1), C3, C1q, kappa (row 2), lambda, fibrinogen, albumin (row 3). "
        f"Channels shown in black labelled MISSING were not uploaded.\n\n"
        f"The AI model predicted: {diag_label} ({diag_pct:.1f}% confidence).\n"
        f"Channels used: {ch_str}.\n"
        + (f"{multi_channel_notes}\n" if multi_channel_notes else "")
        + "\nReview the mosaic carefully. In 3-5 concise sentences:\n"
        "1. Does the staining pattern agree with the model's prediction?\n"
        "2. Flag anything that contradicts or is clinically concerning.\n"
        "3. Note any missing channels that would be critical for this diagnosis.\n"
        "Be concise, clinically focused, and use standard nephropathology terminology."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{mosaic_b64}"},
                },
                {
                    "type": "text",
                    "text": prompt_text,
                },
            ],
        }
    ]
    headers = {
        "Content-Type": "application/json",
    }

    last_err = None
    for api_key in api_keys:
        headers["Authorization"] = f"Bearer {api_key}"
        for model_id in IF_REVIEW_MODELS:
            payload = {
                "model": model_id,
                "messages": messages,
                "max_tokens": 512,
                "temperature": 0.3,
            }
            for attempt in range(3):
                try:
                    resp = requests.post(
                        OPENROUTER_API_URL,
                        headers=headers,
                        json=payload,
                        timeout=60,
                    )
                    if resp.status_code == 401:
                        last_err = f"key ...{api_key[-4:]} invalid or missing"
                        break
                    if resp.status_code == 429:
                        wait = 2 ** attempt
                        time.sleep(wait)
                        continue
                    if resp.status_code in (404, 529):
                        last_err = f"{model_id} unavailable ({resp.status_code})"
                        break
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"].strip()
                except requests.exceptions.Timeout:
                    last_err = f"{model_id} timed out"
                    break
                except Exception as exc:
                    last_err = str(exc)
                    break
            else:
                last_err = f"{model_id} rate limited on key ...{api_key[-4:]}"
                break  # try next key
    # OpenRouter exhausted — try Groq as fallback
    groq_keys = get_groq_keys()
    if groq_keys:
        for groq_key in groq_keys:
            groq_headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {groq_key}",
            }
            groq_payload = {
                "model": GROQ_MODEL,
                "messages": messages,
                "max_tokens": 512,
                "temperature": 0.3,
            }
            for attempt in range(2):
                try:
                    response = requests.post(
                        GROQ_API_URL,
                        headers=groq_headers,
                        json=groq_payload,
                        timeout=60
                    )
                    if response.status_code == 429:
                        time.sleep([5, 15][attempt])
                        continue
                    response.raise_for_status()
                    return response.json()["choices"][0]["message"]["content"].strip()
                except requests.exceptions.Timeout:
                    last_err = "Groq timed out"
                    break
                except Exception as e:
                    last_err = str(e)
                    break
    return f"⚠️ LLM review failed: all keys and models exhausted. Last error: {last_err}"
def _get_od(img_np: np.ndarray):
    """Convert uint8 RGB image to optical density (OD) space."""
    img = np.maximum(img_np.astype(np.float32) / 255.0, 1e-6)
    return -np.log(img)


def _get_stain_matrix(img_np: np.ndarray, lum_thr: float = 0.8):
    od   = _get_od(img_np)
    mask = (od * od).sum(axis=2) > (1 - lum_thr) ** 2
    od_m = od[mask].reshape(-1, 3)
    if len(od_m) < 10:
        return None, None
    _, _, Vt  = np.linalg.svd(od_m, full_matrices=False)
    plane     = Vt[:2]
    proj      = od_m @ plane.T
    angles    = np.arctan2(proj[:, 1], proj[:, 0])
    phi_min, phi_max = np.percentile(angles, 1), np.percentile(angles, 99)
    v1 = plane[0] * np.cos(phi_min) + plane[1] * np.sin(phi_min)
    v2 = plane[0] * np.cos(phi_max) + plane[1] * np.sin(phi_max)
    if v1[0] < v2[0]:
        v1, v2 = v2, v1
    S = np.stack([v1 / (np.linalg.norm(v1) + 1e-8),
                  v2 / (np.linalg.norm(v2) + 1e-8)], axis=0)
    return S, mask


def _get_concentrations(img_np: np.ndarray, S: np.ndarray):
    od   = _get_od(img_np)
    h, w = od.shape[:2]
    return np.maximum((od.reshape(-1, 3) @ np.linalg.pinv(S)).reshape(h, w, 2), 0)


def macenko_normalise(source: Image.Image, reference: Image.Image) -> Image.Image:
    src_np, ref_np = np.array(source.convert("RGB")), np.array(reference.convert("RGB"))
    S_s, mask_s = _get_stain_matrix(src_np)
    S_r, mask_r = _get_stain_matrix(ref_np)
    if S_s is None or S_r is None:
        return source
    C_s    = _get_concentrations(src_np, S_s)
    C_r    = _get_concentrations(ref_np, S_r)
    # Compute max concentrations only over tissue (non-background) pixels so
    # the scaling ratio is not skewed by mostly-white source or reference images.
    max_s  = np.percentile(C_s[mask_s].reshape(-1, 2), 99, axis=0) + 1e-8
    max_r  = np.percentile(C_r[mask_r].reshape(-1, 2), 99, axis=0) + 1e-8
    C_norm = C_s * (max_r / max_s)
    h, w   = C_norm.shape[:2]
    od_norm  = (C_norm.reshape(-1, 2) @ S_r).reshape(h, w, 3)
    img_norm = np.clip(np.exp(-od_norm) * 255, 0, 255).astype(np.uint8)
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
    quantiles = np.linspace(0, 1, 1024)
    for ch in range(3):
        src_vals = sl[:, :, ch].flatten()
        ref_vals = rl[:, :, ch].flatten()
        # Build a quantile lookup table of equal length for both distributions
        src_quantiles = np.quantile(src_vals, quantiles)
        ref_quantiles = np.quantile(ref_vals, quantiles)
        # Map each source pixel to the matching quantile in the reference distribution
        out[:, :, ch] = np.interp(src_vals, src_quantiles, ref_quantiles).reshape(sl[:, :, ch].shape)
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
def get_unified_report(images, all_probs, all_preds, avg_probs, consensus_pred, consensus_conf,
                       overlay_images=None, if_result=None, if_channel_imgs=None):
    """
    Llama 4 Scout via OpenRouter: sees trichrome images + Grad-CAM overlays + IF channel images,
    then returns one cohesive multimodal nephropathological report.
    """
    api_keys = get_api_keys()
    if not api_keys:
        raise ValueError(
            "OPENROUTER_API_KEY not configured. "
            "Add your OpenRouter API key to Streamlit secrets (OPENROUTER_API_KEY) "
            "or set it as an environment variable. "
            "Get a free key at https://openrouter.ai"
        )

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

    # Build the IF section of the prompt
    if if_result and if_result.get("top_predictions"):
        top_preds = if_result["top_predictions"]
        if_summary_lines = "\n".join(
            f"  #{i+1}: {p['display']} ({p['pct']:.1f}%)"
            for i, p in enumerate(top_preds)
        )
        channels_used = ", ".join(if_result.get("channels_used", []))
        channels_missing = ", ".join(if_result.get("channels_missing", []))
        if_data_block = (
            f"\nIF (IMMUNOFLUORESCENCE) CLASSIFIER OUTPUT:\n"
            f"{if_summary_lines}\n"
            f"  Channels analyzed: {channels_used}\n"
            + (f"  Zero-filled (missing): {channels_missing}\n" if channels_missing else "")
        )
        has_if = True
    else:
        if_data_block = "\nIF (IMMUNOFLUORESCENCE) DATA: Not provided for this case.\n"
        has_if = False

    if_channel_note = ""
    if has_if and if_channel_imgs:
        uploaded_chs = [ch for ch in IF_CHANNELS if ch in if_channel_imgs]
        if_channel_note = (
            f"\n3. IF channel images ({', '.join(uploaded_chs)}) — "
            f"each image represents a single fluorescence channel.\n"
        )

    # Resolve fallback for the generic diagnosis label in the critical instruction
    _critical_diagnosis_label = (
        top_preds[0]['display']
        if has_if and if_result and if_result.get('top_predictions')
        else 'nephropathy'
    )

    prompt = f"""You are an expert nephropathologist reviewing a kidney biopsy with both \
Masson's trichrome staining (for interstitial fibrosis grading) and immunofluorescence (IF) staining.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRICHROME FIBROSIS GRADING (ResNet-FPN model):
- Consensus Grade: {CLASS_NAMES[consensus_pred]} ({CLASS_RANGE[consensus_pred]})
- Confidence: {consensus_conf:.1f}%
- Probability breakdown:
{avg_breakdown}
{multi_note}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{if_data_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You are provided with:
1. The original trichrome biopsy image(s)
2. Trichrome Grad-CAM overlay(s) — red/yellow = regions most discriminative for the fibrosis grade
{if_channel_note}
CRITICAL INSTRUCTION: Every sentence must be anchored to a specific visual feature you actually \
observe in the images provided. Do not write generic statements that would be true of any \
{CLASS_NAMES[consensus_pred]} biopsy or any {_critical_diagnosis_label} case.

**Trichrome — Visual Observations**
Describe the spatial distribution of collagen deposition — is it periglomerular, peritubular, \
or diffuse interstitial? What proportion of the cortex appears affected? Are tubules atrophied \
uniformly or focally? What do the glomeruli look like — sclerotic, globally collapsed, segmentally \
scarred, or relatively preserved? Where exactly does the Grad-CAM heatmap focus — periglomerular \
zones, interstitium, vascular walls? Does that focus correlate with what you see there?

{"**IF — Pattern Analysis**" if has_if else ""}
{"Examine each uploaded IF channel image. Which markers show mesangial, capillary wall, or linear " if has_if else ""}
{"tubular basement membrane positivity? Is the staining granular, linear, or diffuse? " if has_if else ""}
{"Does the IF pattern support or conflict with the top classifier prediction?" if has_if else ""}

{"**Integrated Diagnosis**" if has_if else "**Diagnosis**"}
{"Synthesise trichrome fibrosis grade with IF staining pattern. What single diagnosis best explains " if has_if else ""}
{"all findings — fibrosis distribution, glomerular morphology, and IF positivity profile? " if has_if else ""}
{"Note any discordance between classifier outputs and the visual evidence." if has_if else ""}
{"Based on the trichrome pattern (not general fibrosis severity), what is the likely etiology?" if not has_if else ""}

**Model Agreement**
Does the trichrome fibrosis grade match what you see visually? If yes, which specific feature is \
the strongest supporting evidence? {"Does the IF classifier diagnosis align with the IF staining pattern?" if has_if else ""} \
If the Grad-CAM focuses on an unexpected region, say so and explain why it might or might not be valid.

**Treatment & Recommendations**
One paragraph. Be specific to these findings, not generic. If glomeruli appear preserved, note that. \
If vascular changes are prominent, address that. {"If IF positivity suggests an immune-mediated process, " if has_if else ""}
{"tailor the recommendation accordingly (e.g. immunosuppression, plasmapheresis, etc.)." if has_if else ""}

**Plain-Language Summary**
2-3 sentences for the patient. No jargon."""

    content_parts = [{"type": "text", "text": prompt}]

    # Trichrome images + Grad-CAM overlays
    for i, img in enumerate(images):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
        if overlay_images and i < len(overlay_images) and overlay_images[i] is not None:
            buf_cam = io.BytesIO()
            overlay_images[i].save(buf_cam, format="JPEG", quality=85)
            b64_cam = base64.b64encode(buf_cam.getvalue()).decode("utf-8")
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_cam}"}
            })

    # IF panel — send as single mosaic instead of individual channels
    if has_if and if_channel_imgs:
        # Build multi-list dict expected by build_if_mosaic
        mosaic_dict = {ch: [img] for ch, img in if_channel_imgs.items()}
        mosaic_img = build_if_mosaic(mosaic_dict)
        mosaic_img_resized = mosaic_img.resize((672, 672), Image.LANCZOS)
        buf_mosaic = io.BytesIO()
        mosaic_img_resized.save(buf_mosaic, format="JPEG", quality=85)
        b64_mosaic = base64.b64encode(buf_mosaic.getvalue()).decode("utf-8")
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_mosaic}"}
        })
        del mosaic_img, mosaic_img_resized, buf_mosaic

    headers = {
        "Content-Type": "application/json",
    }
    base_payload = {
        "messages": [{"role": "user", "content": content_parts}],
        "max_tokens": 1800,
        "temperature": 0.3,
    }

    last_err = None
    for api_key in api_keys:
        headers["Authorization"] = f"Bearer {api_key}"
        for model_id in REPORT_MODELS:
            payload = {**base_payload, "model": model_id}
            for attempt in range(3):
                try:
                    response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=90)
                    if response.status_code == 401:
                        last_err = f"key ...{api_key[-4:]} invalid or missing"
                        break
                    if response.status_code == 429:
                        wait = 2 ** attempt
                        time.sleep(wait)
                        continue
                    if response.status_code in (404, 529):
                        last_err = (
                            f"Model not found or unavailable on OpenRouter: {model_id}. "
                            f"Status: {response.status_code}. Body: {response.text[:300]}"
                        )
                        break
                    response.raise_for_status()
                    raw = response.json()["choices"][0]["message"]["content"]
                    return _clean_report_text(raw)
                except requests.exceptions.Timeout:
                    last_err = f"{model_id} timed out"
                    break
                except Exception as exc:
                    last_err = str(exc)
                    break
            else:
                last_err = f"{model_id} rate limited on key ...{api_key[-4:]}"
                break  # try next key
    # OpenRouter exhausted — try Groq as fallback
    groq_keys = get_groq_keys()
    if groq_keys:
        for groq_key in groq_keys:
            groq_headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {groq_key}",
            }
            groq_payload = {
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": content_parts}],
                "max_tokens": 1800,
                "temperature": 0.3,
            }
            for attempt in range(2):
                try:
                    response = requests.post(
                        GROQ_API_URL,
                        headers=groq_headers,
                        json=groq_payload,
                        timeout=90
                    )
                    if response.status_code == 429:
                        time.sleep([5, 15][attempt])
                        continue
                    response.raise_for_status()
                    raw = response.json()["choices"][0]["message"]["content"]
                    return _clean_report_text(raw)
                except requests.exceptions.Timeout:
                    last_err = f"Groq timed out"
                    break
                except Exception as e:
                    last_err = str(e)
                    break
    raise ValueError(f"All keys and models exhausted. Last error: {last_err}")


def _clean_report_text(text: str) -> str:
    """Remove em dashes, en dashes, and markdown # headers from LLM report text.

    Em dashes (—) and en dashes (–) are replaced with a plain hyphen-minus.
    Lines beginning with one or more '#' characters (Markdown headings) have
    the leading '#' symbols stripped so the text is rendered as plain prose
    or as the **bold** heading style used elsewhere in the report.
    """
    # Replace em dash and en dash with a plain hyphen
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    # Replace triple/double dashes that LLMs sometimes emit
    text = re.sub(r"---+", "-", text)

    # Strip leading '#' heading markers from each line, keeping the heading text.
    # e.g. "## Findings" → "**Findings**" so it still renders as a bold header.
    def _strip_hashes(m):
        return f"**{m.group(1).strip()}**"

    text = re.sub(r"^#{1,6}\s+(.*)", _strip_hashes, text, flags=re.MULTILINE)

    return text


# ── PDF Report Generation ──────────────────────────────────────────────────────
def _pil_to_jpeg_bytes(img: Image.Image, max_px: int = 900) -> bytes:
    """Return JPEG bytes for a PIL image, downsampled if larger than max_px on the long side."""
    if max(img.width, img.height) > max_px:
        ratio = max_px / max(img.width, img.height)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return buf.getvalue()


def _latin1(text: str) -> str:
    """Decode HTML entities and coerce to Latin-1 for fpdf2 core fonts."""
    import html as _html
    text = _html.unescape(text)
    for src, dst in [("\u2013", "-"), ("\u2014", "--"), ("\u2018", "'"), ("\u2019", "'"),
                     ("\u201c", '"'), ("\u201d", '"'), ("\u2022", "-"), ("\u2026", "...")]:
        text = text.replace(src, dst)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def generate_pdf_report(images, overlay_images, all_probs, all_preds,
                        avg_probs, consensus_pred, consensus_conf, report_text,
                        if_result=None, if_channel_imgs=None,
                        if_gradcam_overlays=None):
    from fpdf import FPDF
    from datetime import date as _date
    import html as _html

    C_BLACK   = (30,  30,  30)
    C_DARK    = (55,  65,  81)
    C_MID     = (107, 114, 128)
    C_LIGHT   = (156, 163, 175)
    C_BORDER  = (209, 213, 219)
    C_ACCENT  = (37,  99,  235)
    C_BG_BOX  = (239, 246, 255)
    GRADE_COLOURS = [
        (22, 163, 74),
        (202, 138, 4),
        (234, 88,  12),
        (220, 38,  38),
    ]

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=False)
    pdf.set_margins(left=14, top=12, right=14)
    W = pdf.w - pdf.l_margin - pdf.r_margin  # ~182 mm

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1
    # ══════════════════════════════════════════════════════════════════════════
    pdf.add_page()

    # ── Header bar ────────────────────────────────────────────────────────────
    pdf.set_fill_color(*C_ACCENT)
    pdf.rect(pdf.l_margin, pdf.t_margin, W, 11, style="F")
    pdf.set_font("Helvetica", style="B", size=11)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(pdf.l_margin + 4, pdf.t_margin + 1.5)
    pdf.cell(W * 0.55, 5, "Kidney Fibrosis Grader", ln=0)
    pdf.set_font("Helvetica", size=7)
    pdf.set_text_color(199, 218, 255)
    pdf.set_xy(pdf.l_margin + 4, pdf.t_margin + 6)
    pdf.cell(W - 8, 4,
             _latin1(f"Clinicopathological Observation  |  {_date.today().strftime('%B %d, %Y')}  |  ResNet-FPN  |  Llama 4 Scout Vision"),
             ln=0)
    y = pdf.t_margin + 14

    # ── Grade box (left) + probability bars (right) ───────────────────────────
    col_l = W * 0.38
    col_r = W - col_l - 5
    box_h = 28
    gc = GRADE_COLOURS[consensus_pred]

    # Left grade box
    pdf.set_fill_color(*C_BG_BOX)
    pdf.set_draw_color(*C_BORDER)
    pdf.set_line_width(0.25)
    pdf.rect(pdf.l_margin, y, col_l, box_h, style="FD")
    pdf.set_fill_color(*gc)
    pdf.rect(pdf.l_margin, y, 3, box_h, style="F")

    pdf.set_font("Helvetica", size=6.5)
    pdf.set_text_color(*C_MID)
    pdf.set_xy(pdf.l_margin + 5, y + 2.5)
    pdf.cell(col_l, 3.5, "CONSENSUS GRADE", ln=0)

    pdf.set_font("Helvetica", style="B", size=14)
    pdf.set_text_color(*gc)
    pdf.set_xy(pdf.l_margin + 5, y + 6)
    pdf.cell(col_l, 8, _latin1(CLASS_NAMES[consensus_pred]), ln=0)

    pdf.set_font("Helvetica", size=8)
    pdf.set_text_color(*C_DARK)
    pdf.set_xy(pdf.l_margin + 5, y + 14.5)
    pdf.cell(col_l, 4.5, _latin1(_html.unescape(CLASS_RANGE[consensus_pred])), ln=0)

    pdf.set_font("Helvetica", size=7)
    pdf.set_text_color(*C_MID)
    pdf.set_xy(pdf.l_margin + 5, y + 20)
    pdf.cell(col_l, 4, f"Confidence: {consensus_conf:.1f}%", ln=0)

    # Right probability bars
    rx = pdf.l_margin + col_l + 5
    ry = y
    pdf.set_font("Helvetica", style="B", size=6.5)
    pdf.set_text_color(*C_MID)
    pdf.set_xy(rx, ry)
    pdf.cell(col_r, 4, "PROBABILITY BREAKDOWN", ln=0)
    ry += 4.5
    bar_h = 5
    for i in range(4):
        pct   = avg_probs[i] * 100
        bar_w = col_r * pct / 100
        pdf.set_fill_color(229, 231, 235)
        pdf.rect(rx, ry, col_r, bar_h, style="F")
        pdf.set_fill_color(*GRADE_COLOURS[i])
        if bar_w > 0.5:
            pdf.rect(rx, ry, bar_w, bar_h, style="F")
        pdf.set_font("Helvetica", size=6.5)
        pdf.set_text_color(*C_BLACK)
        pdf.set_xy(rx + 1, ry + 0.5)
        pdf.cell(col_r - 2, bar_h - 0.5,
                 _latin1(f"{CLASS_SHORT[i]}   {pct:.1f}%"), ln=0, align="L")
        ry += bar_h + 1

    y += box_h + 4

    # ── IF Diagnosis box (if available) ───────────────────────────────────────
    if if_result and if_result.get("top_predictions"):
        if_top = if_result["top_predictions"]
        if_h   = 6 + len(if_top) * 4.5 + 3
        pdf.set_fill_color(240, 240, 255)
        pdf.set_draw_color(200, 200, 240)
        pdf.set_line_width(0.25)
        pdf.rect(pdf.l_margin, y, W, if_h, style="FD")
        pdf.set_fill_color(80, 70, 200)
        pdf.rect(pdf.l_margin, y, 3, if_h, style="F")

        pdf.set_font("Helvetica", style="B", size=6.5)
        pdf.set_text_color(80, 70, 200)
        pdf.set_xy(pdf.l_margin + 5, y + 2)
        pdf.cell(W - 10, 3.5, "IF (IMMUNOFLUORESCENCE) DIAGNOSIS", ln=0)

        pdf.set_font("Helvetica", size=7)
        pdf.set_text_color(*C_DARK)
        iy = y + 6
        for rank_i, pred in enumerate(if_top):
            pdf.set_xy(pdf.l_margin + 5, iy)
            label = _latin1(f"#{rank_i+1}  {pred['display']}  —  {pred['pct']:.1f}%")
            pdf.cell(W - 10, 4, label, ln=0)
            iy += 4.5

        if if_result.get("channels_used"):
            pdf.set_font("Helvetica", size=6)
            pdf.set_text_color(*C_MID)
            pdf.set_xy(pdf.l_margin + 5, iy - 1)
            pdf.cell(W - 10, 3.5, _latin1("Channels: " + ", ".join(if_result["channels_used"])), ln=0)

        y += if_h + 3

    # ── Divider ───────────────────────────────────────────────────────────────
    pdf.set_draw_color(*C_BORDER)
    pdf.set_line_width(0.25)
    pdf.line(pdf.l_margin, y, pdf.l_margin + W, y)
    y += 3

    # ── Images section label ──────────────────────────────────────────────────
    pdf.set_font("Helvetica", style="B", size=8)
    pdf.set_text_color(*C_DARK)
    pdf.set_xy(pdf.l_margin, y)
    pdf.cell(W, 5, "Biopsy Images & Grad-CAM Overlays", ln=0)
    y += 6

    # ── Image grid ────────────────────────────────────────────────────────────
    n = len(images)
    has_overlay = bool(overlay_images and any(o is not None for o in overlay_images))

    # Available height for images: page height - current y - bottom margin (footer ~10mm)
    avail_h = pdf.h - y - 14
    avail_w = W

    if has_overlay:
        # Each image takes (W / n) wide, split into original|cam halves
        pair_w  = (avail_w - (n - 1) * 3) / n
        img_w   = (pair_w - 2) / 2
        img_h   = min(img_w * 1.1, avail_h - 8)

        for i, orig in enumerate(images):
            x_pair = pdf.l_margin + i * (pair_w + 3)
            overlay = overlay_images[i] if i < len(overlay_images) else None

            # Caption
            pdf.set_font("Helvetica", size=6)
            pdf.set_text_color(*C_MID)
            pdf.set_xy(x_pair, y)
            lbl = _latin1(f"Img {i+1} -- {CLASS_NAMES[all_preds[i]]} ({all_probs[i][all_preds[i]]*100:.0f}%)")
            pdf.cell(pair_w, 4, lbl, ln=0, align="C")

            orig_bytes = _pil_to_jpeg_bytes(orig)
            pdf.image(io.BytesIO(orig_bytes), x=x_pair, y=y + 4, w=img_w, h=img_h)

            if overlay is not None:
                cam_bytes = _pil_to_jpeg_bytes(overlay)
                pdf.image(io.BytesIO(cam_bytes), x=x_pair + img_w + 2, y=y + 4, w=img_w, h=img_h)

            # Sub-captions
            pdf.set_font("Helvetica", size=5.5)
            pdf.set_text_color(*C_LIGHT)
            pdf.set_xy(x_pair, y + 4 + img_h + 0.5)
            pdf.cell(img_w, 3, "Original", ln=0, align="C")
            if overlay is not None:
                pdf.set_xy(x_pair + img_w + 2, y + 4 + img_h + 0.5)
                pdf.cell(img_w, 3, "Grad-CAM", ln=0, align="C")
    else:
        img_w = (avail_w - (n - 1) * 3) / max(n, 1)
        img_h = min(img_w * 1.1, avail_h - 8)
        for i, orig in enumerate(images):
            x = pdf.l_margin + i * (img_w + 3)
            pdf.set_font("Helvetica", size=6)
            pdf.set_text_color(*C_MID)
            pdf.set_xy(x, y)
            pdf.cell(img_w, 4, _latin1(f"Img {i+1}"), ln=0, align="C")
            orig_bytes = _pil_to_jpeg_bytes(orig)
            pdf.image(io.BytesIO(orig_bytes), x=x, y=y + 4, w=img_w, h=img_h)

    # ── Page 1 footer ─────────────────────────────────────────────────────────
    pdf.set_draw_color(*C_BORDER)
    pdf.set_line_width(0.2)
    pdf.line(pdf.l_margin, pdf.h - 9, pdf.l_margin + W, pdf.h - 9)
    pdf.set_font("Helvetica", size=5.5)
    pdf.set_text_color(*C_LIGHT)
    pdf.set_xy(pdf.l_margin, pdf.h - 8)
    pdf.cell(W, 4,
             "For research use only. Not validated for clinical diagnosis. Always consult a qualified pathologist.",
             ln=0, align="C")

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2 — IF Panel Mosaic & Grad-CAM (if available) — before AI report
    # ══════════════════════════════════════════════════════════════════════════
    if if_channel_imgs:
        uploaded_chs = [ch for ch in IF_CHANNELS if ch in if_channel_imgs]
        if uploaded_chs:
            has_if_gcam = bool(if_gradcam_overlays)

            pdf.add_page()

            # Header bar
            page2_title = (
                "IF Panel Mosaic & Grad-CAM Activation Maps"
                if has_if_gcam else "IF Panel Mosaic"
            )
            pdf.set_fill_color(*C_ACCENT)
            pdf.rect(pdf.l_margin, pdf.t_margin, W, 9, style="F")
            pdf.set_font("Helvetica", style="B", size=10)
            pdf.set_text_color(255, 255, 255)
            pdf.set_xy(pdf.l_margin + 4, pdf.t_margin + 1.5)
            pdf.cell(W, 6, page2_title, ln=0)

            y_if = pdf.t_margin + 13

            # ── Mosaic image (replaces individual channel grid) ───────────────
            mosaic_dict_pdf = {ch: [if_channel_imgs[ch]] for ch in uploaded_chs}
            mosaic_pdf = build_if_mosaic(mosaic_dict_pdf)
            # Scale mosaic to fit page width (max ~130 mm to leave breathing room)
            mosaic_max_w = min(W, 130.0)
            mosaic_aspect = mosaic_pdf.height / mosaic_pdf.width
            mosaic_w = mosaic_max_w
            mosaic_h = mosaic_w * mosaic_aspect
            # If mosaic is too tall, constrain by height instead
            available_h = pdf.h - y_if - 14  # leave footer room
            if mosaic_h > available_h:
                mosaic_h = available_h
                mosaic_w = mosaic_h / mosaic_aspect

            mosaic_x = pdf.l_margin + (W - mosaic_w) / 2  # centred
            mosaic_bytes = _pil_to_jpeg_bytes(mosaic_pdf, max_px=1200)
            pdf.image(io.BytesIO(mosaic_bytes), x=mosaic_x, y=y_if, w=mosaic_w, h=mosaic_h)

            # Sub-caption
            pdf.set_font("Helvetica", size=6)
            pdf.set_text_color(*C_LIGHT)
            pdf.set_xy(pdf.l_margin, y_if + mosaic_h + 1)
            pdf.cell(W, 3,
                     _latin1(f"Channels: {', '.join(uploaded_chs)}  |  "
                             "Black cells = missing channels (zero-filled)"),
                     ln=1, align="C")

            y_if = pdf.get_y() + 4

            # ── IF Grad-CAM section (top 1-2 channels) ───────────────────────
            if has_if_gcam:
                gcam_channels = [ch for ch in IF_CHANNELS if ch in if_gradcam_overlays]
                if gcam_channels:
                    # Section sub-header
                    pdf.set_font("Helvetica", style="B", size=8)
                    pdf.set_text_color(*C_ACCENT)
                    pdf.set_xy(pdf.l_margin, y_if)
                    pdf.cell(W, 5, "IF Grad-CAM -- Top Channel(s)", ln=1)
                    pdf.set_draw_color(*C_BORDER)
                    pdf.set_line_width(0.2)
                    pdf.line(pdf.l_margin, pdf.get_y(),
                             pdf.l_margin + W * 0.4, pdf.get_y())
                    pdf.ln(2)

                    pdf.set_font("Helvetica", size=6)
                    pdf.set_text_color(199, 218, 255)
                    pdf.set_xy(pdf.l_margin, pdf.get_y())
                    pdf.cell(W, 3,
                             "Dark-red = peak attention  |  Yellow/green = secondary  |  Blue = ignored",
                             ln=1, align="L")
                    pdf.ln(2)
                    y_gcam = pdf.get_y()

                    n_gcam = len(gcam_channels)
                    gcam_gap = 4
                    gcam_pair_w = (W - (n_gcam - 1) * gcam_gap) / max(n_gcam, 1)
                    gcam_orig_w = (gcam_pair_w - 2) / 2
                    gcam_h = gcam_orig_w * 0.9

                    for ci, gcam_ch in enumerate(gcam_channels):
                        x_gcam = pdf.l_margin + ci * (gcam_pair_w + gcam_gap)
                        pdf.set_font("Helvetica", size=6.5)
                        pdf.set_text_color(*C_MID)
                        pdf.set_xy(x_gcam, y_gcam)
                        pdf.cell(gcam_pair_w, 4, _latin1(gcam_ch), ln=0, align="C")
                        # Original channel image
                        if gcam_ch in if_channel_imgs:
                            orig_bytes = _pil_to_jpeg_bytes(if_channel_imgs[gcam_ch])
                            pdf.image(io.BytesIO(orig_bytes),
                                      x=x_gcam, y=y_gcam + 4, w=gcam_orig_w, h=gcam_h)
                        # Grad-CAM overlay
                        overlay_val = if_gradcam_overlays.get(gcam_ch)
                        if overlay_val is None or isinstance(overlay_val, str):
                            continue
                        gcam_bytes_img = _pil_to_jpeg_bytes(overlay_val)
                        pdf.image(io.BytesIO(gcam_bytes_img),
                                  x=x_gcam + gcam_orig_w + 2, y=y_gcam + 4,
                                  w=gcam_orig_w, h=gcam_h)
                        # Sub-captions
                        pdf.set_font("Helvetica", size=5.5)
                        pdf.set_text_color(*C_LIGHT)
                        pdf.set_xy(x_gcam, y_gcam + 4 + gcam_h + 0.5)
                        pdf.cell(gcam_orig_w, 3, "Original", ln=0, align="C")
                        pdf.set_xy(x_gcam + gcam_orig_w + 2, y_gcam + 4 + gcam_h + 0.5)
                        pdf.cell(gcam_orig_w, 3, "Grad-CAM", ln=0, align="C")

            # Footer on IF page
            pdf.set_draw_color(*C_BORDER)
            pdf.set_line_width(0.2)
            pdf.line(pdf.l_margin, pdf.h - 9, pdf.l_margin + W, pdf.h - 9)
            pdf.set_font("Helvetica", size=5.5)
            pdf.set_text_color(*C_LIGHT)
            pdf.set_xy(pdf.l_margin, pdf.h - 8)
            pdf.cell(W, 4,
                     "For research use only. Not validated for clinical diagnosis. Always consult a qualified pathologist.",
                     ln=0, align="C")

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 3 — AI Report text (after IF images)
    # ══════════════════════════════════════════════════════════════════════════
    pdf.add_page()

    # Header bar
    pdf.set_fill_color(*C_ACCENT)
    pdf.rect(pdf.l_margin, pdf.t_margin, W, 9, style="F")
    pdf.set_font("Helvetica", style="B", size=10)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(pdf.l_margin + 4, pdf.t_margin + 1.5)
    pdf.cell(W, 6, "AI Clinicopathological Observation", ln=0)
    pdf.set_xy(pdf.l_margin, pdf.t_margin + 12)

    # Body text — stop before footer
    text_bottom_limit = pdf.h - 12

    for line in report_text.split("\n"):
        if pdf.get_y() > text_bottom_limit:
            break
        stripped = line.strip()
        if not stripped:
            pdf.ln(2)
            continue
        if stripped.startswith("**") and stripped.endswith("**") and len(stripped) > 4:
            header = stripped[2:-2].strip()
            pdf.set_font("Helvetica", style="B", size=9)
            pdf.set_text_color(*C_ACCENT)
            pdf.set_x(pdf.l_margin)
            pdf.cell(W, 5.5, _latin1(header), ln=1)
            pdf.set_draw_color(*C_BORDER)
            pdf.set_line_width(0.2)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + W * 0.3, pdf.get_y())
            pdf.ln(1.5)
        else:
            body = re.sub(r'\*\*(.*?)\*\*', r'\1', stripped)
            body = re.sub(r'\*(.*?)\*',   r'\1', body)
            pdf.set_font("Helvetica", size=8)
            pdf.set_text_color(*C_DARK)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(W, 4.5, _latin1(body), align="L")
            pdf.ln(0.5)

    # Footer on AI report page
    pdf.set_draw_color(*C_BORDER)
    pdf.set_line_width(0.2)
    pdf.line(pdf.l_margin, pdf.h - 9, pdf.l_margin + W, pdf.h - 9)
    pdf.set_font("Helvetica", size=5.5)
    pdf.set_text_color(*C_LIGHT)
    pdf.set_xy(pdf.l_margin, pdf.h - 8)
    pdf.cell(W, 4,
             "For research use only. Not validated for clinical diagnosis. Always consult a qualified pathologist.",
             ln=0, align="C")

    return bytes(pdf.output())


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
for key in ["imgs", "all_probs", "all_preds", "if_channel_imgs", "if_result"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Analysis & IF Diagnosis",
    "Clinicopathological Observation",
    "Grad-CAM Explainability",
    "Stain Normalisation",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYSIS & IF DIAGNOSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # ── Trichrome Fibrosis Grading ─────────────────────────────────────────────
    st.markdown("""
<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
    <div style="font-family:'Playfair Display',serif; font-size:18px; font-weight:700; color:#e0e6f0;">
        Trichrome Fibrosis Grading
    </div>
    <div style="font-family:'IBM Plex Mono',monospace; font-size:9px; font-weight:500; letter-spacing:0.08em;
                background:#0f1f40; color:#93c5fd; border:1px solid #1e3a6e; padding:4px 10px; border-radius:4px;">
        ResNet-FPN &nbsp;·&nbsp; 4-CLASS GRADING
    </div>
</div>
""", unsafe_allow_html=True)

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
                st.image(imgs[0], use_container_width=True)
            else:
                thumb_cols = st.columns(len(imgs))
                for i, (tc, im) in enumerate(zip(thumb_cols, imgs)):
                    with tc:
                        st.image(im, use_container_width=True, caption=f"Image {i+1}")
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

    # ── IF Diagnosis section ───────────────────────────────────────────────────
    st.markdown("""
<div style="height:1px; background:linear-gradient(90deg,transparent,#1e2d45,transparent);
            margin:32px 0;"></div>
<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
    <div style="font-family:'Playfair Display',serif; font-size:18px; font-weight:700; color:#e0e6f0;">
        Immunofluorescence Diagnosis
    </div>
    <div style="font-family:'IBM Plex Mono',monospace; font-size:9px; font-weight:500; letter-spacing:0.08em;
                background:#1a1a2e; color:#818cf8; border:1px solid #3730a3; padding:4px 10px; border-radius:4px;">
        ResNet-50 &nbsp;·&nbsp; 9-CHANNEL IF
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="info-box">
    <strong>How to use:</strong> Upload any combination of the 9 IF channel images below.
    Missing channels are zero-filled automatically — more channels yield higher confidence.
    Recommend uploading ≥ 3 channels for reliable results.
    The model returns the top-3 most likely nephropathological diagnoses.
    IF results are automatically included in the Clinicopathological Observation and Grad-CAM tabs.
</div>
""", unsafe_allow_html=True)

    if_left, if_right = st.columns([2.5, 1.5], gap="large")

    with if_left:
        st.markdown('<div class="sec-label">Upload IF Images by Channel</div>', unsafe_allow_html=True)
        row1_cols = st.columns(3, gap="small")
        row2_cols = st.columns(3, gap="small")
        row3_cols = st.columns(3, gap="small")
        all_rows = [row1_cols, row2_cols, row3_cols]

        channel_files: dict = {}
        for idx, ch in enumerate(IF_CHANNELS):
            row_idx = idx // 3
            col_idx = idx % 3
            with all_rows[row_idx][col_idx]:
                st.markdown(
                    f'<div class="norm-panel-label">{ch}</div>',
                    unsafe_allow_html=True,
                )
                fs = st.file_uploader(
                    ch,
                    type=["jpg", "jpeg", "png"],
                    key=f"if_ch_{ch}",
                    label_visibility="collapsed",
                    accept_multiple_files=True,
                )
                if fs:
                    imgs = [Image.open(f).convert("RGB") for f in fs]
                    channel_files[ch] = imgs
                    for img in imgs:
                        st.image(img, use_container_width=True)

        # Persist channel images to session state; clear cached IF result if channels change.
        # A tiny thumbnail fingerprint is used as a stable, content-based cache key for the
        # uploaded IF channels (avoids re-running the classifier when nothing has changed).
        _if_bytes = b"".join(
            img.resize(_IF_HASH_THUMB_SIZE).tobytes()
            for ch in IF_CHANNELS if ch in channel_files
            for img in channel_files[ch]
        )
        _if_upload_key = hashlib.md5(_if_bytes).hexdigest() if _if_bytes else ""
        if st.session_state.get("_if_upload_key") != _if_upload_key:
            st.session_state["_if_upload_key"] = _if_upload_key
            # Store the first image per channel for Grad-CAM, PDF, and other consumers
            st.session_state.if_channel_imgs = (
                {ch: imgs[0] for ch, imgs in channel_files.items()} if channel_files else None
            )
            # Invalidate cached IF result, IF Grad-CAM, and LLM review when channels change
            old_key = st.session_state.get("_if_upload_key", "")
            for _k in list(st.session_state.keys()):
                if (
                    _k in ("if_result", "if_gcam_cache")
                    or _k.startswith("if_gcam_cache_")
                    or _k.startswith(f"if_llm_review_{old_key}")
                ):
                    st.session_state.pop(_k, None)

    with if_right:
        n_uploaded = len(channel_files)
        st.markdown(
            f'<div class="sec-label">{n_uploaded}/9 Channels Uploaded</div>',
            unsafe_allow_html=True,
        )

        run_if_btn = st.button(
            "▶  Run IF Classifier",
            use_container_width=True,
            key="if_run_btn",
            disabled=(n_uploaded == 0),
        )

        if run_if_btn and channel_files:
            if_model = load_if_model()
            if if_model is None:
                st.error("IF classifier model not loaded. Check checkpoint path.")
            else:
                # Average all images per channel for multi-image channels; use single image as-is
                inference_tmp_paths = {}
                tmp_to_delete = []
                for ch, imgs in channel_files.items():
                    if len(imgs) == 1:
                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                            imgs[0].save(tf.name, format="JPEG")
                            inference_tmp_paths[ch] = tf.name
                            tmp_to_delete.append(tf.name)
                    else:
                        # Pixel-average grayscale arrays across multiple images for this channel
                        arrays = [
                            np.array(img.convert("L").resize((224, 224)), dtype=np.float32) / 255.0
                            for img in imgs
                        ]
                        avg = np.mean(arrays, axis=0)
                        avg_uint8 = (np.clip(avg, 0, 1) * 255).astype(np.uint8)
                        avg_pil = Image.fromarray(avg_uint8, mode="L").convert("RGB")
                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                            avg_pil.save(tf.name, format="JPEG")
                            inference_tmp_paths[ch] = tf.name
                            tmp_to_delete.append(tf.name)
                try:
                    with st.spinner("Classifying IF pattern..."):
                        res = if_model.predict(inference_tmp_paths, top_k=3)
                    st.session_state["if_result"] = res
                    # Remove the cached report key so the Clinicopathological tab
                    # regenerates the LLM report with the new IF diagnosis included.
                    st.session_state.pop("report_key", None)
                except Exception as e:
                    st.error(f"Classification error: {str(e)}")
                finally:
                    for p in tmp_to_delete:
                        try:
                            os.unlink(p)
                        except OSError:
                            pass

        if "if_result" in st.session_state and st.session_state["if_result"] is not None:
            res = st.session_state["if_result"]

            if res.get("warning"):
                st.markdown(
                    f'<div class="info-box" style="border-left-color:#f59e0b;">⚠️ {res["warning"]}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<div class="sec-label" style="margin-top:12px;">Top Diagnoses</div>',
                unsafe_allow_html=True,
            )

            medals      = ["🥇", "🥈", "🥉"]
            diag_colors = ["#818cf8", "#a78bfa", "#c4b5fd"]

            for i, pred in enumerate(res["top_predictions"]):
                pct   = pred["pct"]
                color = diag_colors[i]
                st.markdown(f"""
<div class="card" style="padding:14px 16px; margin-bottom:10px;">
    <div style="font-family:'IBM Plex Mono',monospace; font-size:10px; color:#4a5880; margin-bottom:4px;">
        {medals[i]} RANK {i + 1}
    </div>
    <div style="font-size:14px; font-weight:700; color:{color}; margin-bottom:8px;">
        {pred["display"]}
    </div>
    <div class="prob-track" style="margin-bottom:4px;">
        <div class="prob-fill" style="width:{pct:.1f}%; background:{color};"></div>
    </div>
    <div style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#7a8aaa;">
        {pct:.1f}%
    </div>
</div>
""", unsafe_allow_html=True)

            used    = res["channels_used"]
            missing = res["channels_missing"]
            st.markdown(
                f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:10px; '
                f'color:#3d5070; margin-top:8px;">Channels used: {", ".join(used)}</div>',
                unsafe_allow_html=True,
            )
            if missing:
                st.markdown(
                    f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:10px; '
                    f'color:#2a3a56; margin-top:4px;">Zero-filled: {", ".join(missing)}</div>',
                    unsafe_allow_html=True,
                )

            # ── IF Panel Mosaic & LLM Safety Review ──────────────────────────
            if channel_files:
                st.markdown(
                    '<div class="sec-label" style="margin-top:16px;">IF Panel Mosaic</div>',
                    unsafe_allow_html=True,
                )
                mosaic = build_if_mosaic(channel_files)
                st.image(mosaic, caption="IF Panel Mosaic (sent to LLM reviewer)", use_container_width=True)

                # Build notes about channels that had multiple images averaged
                notes = [
                    f"{ch} had {len(imgs)} images \u2014 averaged into one panel cell"
                    for ch, imgs in channel_files.items() if len(imgs) > 1
                ]
                multi_channel_notes = ("Note: " + ". ".join(notes) + ".") if notes else ""

                llm_review_key = f"if_llm_review_{st.session_state.get('_if_upload_key', '')}"
                if llm_review_key not in st.session_state:
                    with st.spinner("Requesting LLM safety review of IF panel..."):
                        mosaic_b64 = pil_to_base64(mosaic)
                        review_text = llm_review_if_panel(
                            mosaic_b64,
                            res["top_predictions"],
                            res["channels_used"],
                            multi_channel_notes,
                        )
                    st.session_state[llm_review_key] = review_text
                else:
                    review_text = st.session_state[llm_review_key]

                st.markdown(
                    f'<div class="info-box" style="margin-top:10px;">'
                    f'<strong>🔬 LLM Safety Review</strong><br>{review_text}</div>',
                    unsafe_allow_html=True,
                )

                # ── IF Grad-CAM for top 1-2 uploaded channels ────────────────
                if_result_for_gcam = st.session_state.get("if_result")
                if_channel_imgs_for_gcam = st.session_state.get("if_channel_imgs")
                gcam_upload_key = st.session_state.get("_if_upload_key", "")
                gcam_cache_key = f"if_gcam_cache_{gcam_upload_key}"

                if (
                    if_result_for_gcam
                    and if_channel_imgs_for_gcam
                    and gcam_cache_key not in st.session_state
                ):
                    top_diag = if_result_for_gcam["top_predictions"][0]["diagnosis"]
                    try:
                        gcam_class_idx = IF_CLASSES.index(top_diag)
                    except (ImportError, ValueError):
                        gcam_class_idx = 0

                    gcam_channels = [
                        ch for ch in IF_CHANNELS if ch in if_channel_imgs_for_gcam
                    ][:2]

                    with st.spinner("Computing IF Grad-CAM..."):
                        computed_gcam = {}
                        for gcam_ch in gcam_channels:
                            try:
                                overlay_pil = compute_if_gradcam(
                                    if_channel_imgs_for_gcam[gcam_ch],
                                    gcam_ch,
                                    gcam_class_idx,
                                )
                                if overlay_pil is None:
                                    # Store a sentinel so we don't recompute
                                    computed_gcam[gcam_ch] = "unreliable"
                                else:
                                    computed_gcam[gcam_ch] = overlay_pil
                            except Exception:
                                pass
                    st.session_state[gcam_cache_key] = computed_gcam

                if gcam_cache_key in st.session_state and st.session_state[gcam_cache_key]:
                    cached_gcam = st.session_state[gcam_cache_key]
                    st.markdown(
                        '<div class="sec-label" style="margin-top:16px;">IF Grad-CAM — Top Channel(s)</div>',
                        unsafe_allow_html=True,
                    )
                    gcam_cols = st.columns(len(cached_gcam), gap="small")
                    for col, (gcam_ch, overlay) in zip(gcam_cols, cached_gcam.items()):
                        with col:
                            if overlay is None or isinstance(overlay, str):
                                st.markdown(
                                    f'<div style="text-align:center;padding:8px;color:#888;font-size:0.8rem;">'
                                    f'⚠️ {gcam_ch}<br>Signal too weak or activation outside tissue — Grad-CAM skipped</div>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.image(
                                    overlay,
                                    caption=f"{gcam_ch} — Grad-CAM",
                                    use_container_width=True,
                                )

        elif n_uploaded == 0:
            st.markdown("""
<div class="await-wrap">
    <div class="await-label">No Images Uploaded</div>
    <div class="await-sub">Upload at least one IF channel image to run the classifier.</div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Clinicopathological Observation
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

        # Cache key — invalidated whenever trichrome images OR IF data changes
        _thumb_bytes = b"".join(im.resize((8, 8)).tobytes() for im in st.session_state.imgs)
        _if_key_part = st.session_state.get("_if_upload_key", "")
        _if_ran      = "1" if st.session_state.get("if_result") is not None else "0"
        _report_key  = hashlib.md5(
            _thumb_bytes + _if_key_part.encode() + _if_ran.encode()
        ).hexdigest()

        st.markdown("""
<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:18px;">
    <div style="font-family:'Playfair Display',serif; font-size:18px; font-weight:700; color:#e0e6f0;">Clinicopathological Observation</div>
    <div style="font-family:'IBM Plex Mono',monospace; font-size:9px; font-weight:500; letter-spacing:0.08em;
                background:#1a2d1a; color:#6ee7b7; border:1px solid #1a4a35; padding:4px 10px; border-radius:4px;">
        LLAMA 4 SCOUT &nbsp;·&nbsp; VISION</div>
</div>
""", unsafe_allow_html=True)

        rcol1, rcol2 = st.columns([1, 2.4], gap="large")

        with rcol1:
            st.markdown('<div class="sec-label">Trichrome Images</div>', unsafe_allow_html=True)
            for i, im in enumerate(st.session_state.imgs):
                caption = (
                    f"Image {i+1} — {CLASS_NAMES[all_preds[i]]}"
                    if len(st.session_state.imgs) > 1
                    else "Analyzed biopsy image"
                )
                st.image(im, use_container_width=True, caption=caption)
                # Show Grad-CAM overlay if available from the cached report
                overlays = st.session_state.get("report_overlays")
                if overlays and i < len(overlays) and overlays[i] is not None:
                    st.image(overlays[i], use_container_width=True,
                             caption=f"Grad-CAM{' (Image ' + str(i+1) + ')' if len(st.session_state.imgs) > 1 else ''}")

            # Show IF channel images if available
            _if_imgs = st.session_state.get("if_channel_imgs")
            if _if_imgs:
                st.markdown('<div class="sec-label" style="margin-top:16px;">IF Channel Images</div>',
                            unsafe_allow_html=True)
                for ch in IF_CHANNELS:
                    if ch in _if_imgs:
                        st.image(_if_imgs[ch], use_container_width=True, caption=ch)

        with rcol2:
            # Show API key input if OPENROUTER_API_KEY is not configured server-side
            def _openrouter_key_from_secrets():
                try:
                    return st.secrets.get("OPENROUTER_API_KEY", "")
                except Exception:
                    return ""

            _openrouter_configured = bool(get_api_keys() or get_groq_keys())
            if not _openrouter_configured:
                _user_key = st.text_input(
                    "OpenRouter API Key",
                    value=st.session_state.get("openrouter_api_key_input", ""),
                    type="password",
                    placeholder="sk-or-...",
                    help="Enter your OpenRouter API key to generate the AI report. Get a free key at openrouter.ai",
                    key="openrouter_api_key_widget",
                )
                if _user_key:
                    st.session_state["openrouter_api_key_input"] = _user_key
                    # Invalidate cached report so it regenerates with the new key
                    if st.session_state.get("openrouter_key_used") != _user_key:
                        st.session_state.pop("report_key", None)
                    st.session_state["openrouter_key_used"] = _user_key
                elif not st.session_state.get("openrouter_api_key_input"):
                    st.info(
                        "Enter your [OpenRouter API key](https://openrouter.ai) above to generate the "
                        "AI clinicopathological report. A free key is available at openrouter.ai."
                    )

            # Only call the LLM (and rebuild the PDF) when images have changed
            _has_key = bool(get_api_keys() or get_groq_keys() or st.session_state.get("openrouter_api_key_input", ""))
            if _has_key and st.session_state.get("report_key") != _report_key:
                with st.spinner("Generating Clinicopathological Observation..."):
                    try:
                        # Compute Grad-CAM overlays for all images to pass to the LLM
                        overlay_images = []
                        for im in st.session_state.imgs:
                            try:
                                heatmap_rgb, overlay_np, _, _, _ = compute_gradcam(im, target_class=None)
                                overlay_images.append(Image.fromarray(overlay_np))
                            except Exception as cam_err:
                                st.warning(f"Grad-CAM overlay unavailable for one image: {cam_err}")
                                overlay_images.append(None)

                        report = get_unified_report(
                            images=st.session_state.imgs,
                            all_probs=all_probs,
                            all_preds=all_preds,
                            avg_probs=avg_probs,
                            consensus_pred=consensus_pred,
                            consensus_conf=consensus_conf,
                            overlay_images=overlay_images,
                            if_result=st.session_state.get("if_result"),
                            if_channel_imgs=st.session_state.get("if_channel_imgs"),
                        )

                        pdf_bytes = generate_pdf_report(
                            images=st.session_state.imgs,
                            overlay_images=overlay_images,
                            all_probs=all_probs,
                            all_preds=all_preds,
                            avg_probs=avg_probs,
                            consensus_pred=consensus_pred,
                            consensus_conf=consensus_conf,
                            report_text=report,
                            if_result=st.session_state.get("if_result"),
                            if_channel_imgs=st.session_state.get("if_channel_imgs"),
                            if_gradcam_overlays=st.session_state.get(
                                f"if_gcam_cache_{st.session_state.get('_if_upload_key', '')}"
                            ),
                        )
                        st.session_state.report_key      = _report_key
                        st.session_state.report_text     = report
                        st.session_state.report_overlays = overlay_images
                        st.session_state.report_pdf      = pdf_bytes
                    except requests.exceptions.HTTPError as e:
                        if e.response.status_code == 401:
                            st.error("OpenRouter API key missing. Add OPENROUTER_API_KEY to Streamlit secrets.")
                        elif e.response.status_code == 429:
                            st.warning("Rate limit reached. Please wait a moment and retry.")
                        else:
                            st.error(f"Report unavailable: {str(e)}")
                    except ValueError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"Report unavailable: {str(e)}")

            # Display cached report and download button
            if st.session_state.get("report_key") == _report_key and "report_text" in st.session_state:
                st.markdown('<div class="ai-body">', unsafe_allow_html=True)
                st.markdown(st.session_state.report_text)
                st.markdown('</div>', unsafe_allow_html=True)
                st.download_button(
                    label="⬇ Download PDF Report",
                    data=st.session_state.report_pdf,
                    file_name="kidney_fibrosis_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
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

        gcam_alpha_col, _ = st.columns([1, 2])
        with gcam_alpha_col:
            overlay_alpha = st.slider("Overlay opacity", 0.1, 0.9, 0.45, 0.05, key="gcam_alpha")

        # Only recompute when the image changes; the slider just reblends.
        # Use a small-thumbnail MD5 as a stable, content-based image fingerprint.
        _thumb = selected_img.resize((16, 16)).tobytes()
        gcam_img_key = hashlib.md5(_thumb).hexdigest()
        gcam_needs_recompute = (
            st.session_state.get("gcam_last_img_key") != gcam_img_key
        )

        if gcam_needs_recompute:
            with st.spinner("Computing Grad-CAM..."):
                try:
                    # target_class=None → automatically selects the predicted class
                    heatmap_rgb, _, pred_class, probs, cam_raw = compute_gradcam(
                        selected_img, target_class=None
                    )
                    st.session_state["gcam_last_img_key"] = gcam_img_key
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
            st.image(selected_img, use_container_width=True)
        with g2:
            st.markdown('<div class="norm-panel-label">Grad-CAM Heatmap</div>', unsafe_allow_html=True)
            st.image(heatmap_rgb, use_container_width=True)
        with g3:
            st.markdown('<div class="norm-panel-label">Overlay</div>', unsafe_allow_html=True)
            st.image(overlay_np, use_container_width=True)

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
            st.image(sn_source, use_container_width=True, caption="Source (original)")

    with sn_right:
        st.markdown('<div class="sec-label">Reference Image (target stain)</div>', unsafe_allow_html=True)
        sn_ref_upload = st.file_uploader(
            "Upload reference image", type=["jpg", "jpeg", "png"],
            key="sn_ref_upload", label_visibility="collapsed"
        )
        sn_reference = Image.open(sn_ref_upload).convert("RGB") if sn_ref_upload else None
        if sn_reference:
            st.image(sn_reference, use_container_width=True, caption="Reference (target lab)")

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
                st.image(sn_source, use_container_width=True)
            with r2:
                st.markdown('<div class="norm-panel-label">Reference</div>', unsafe_allow_html=True)
                st.image(sn_reference, use_container_width=True)
            with r3:
                st.markdown('<div class="norm-panel-label">Normalised Output</div>', unsafe_allow_html=True)
                st.image(result_img, use_container_width=True)

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
