import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import sys, os, io, base64, requests
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
.tpill-purple { color: #c084fc; border-color: #4a2a80; background: #2a1a4a; }

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
    max-height: 320px !important; object-fit: cover !important;
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

/* TABS */
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

/* STAIN NORM INFO BOX */
.norm-info {
    background: #1a2435; border: 1px solid #2a3a55;
    border-radius: 8px; padding: 14px 16px; margin-bottom: 14px;
    font-size: 12px; color: #7a8aaa; line-height: 1.7;
}
.norm-info strong { color: #a0b4cc; }

/* HEATMAP LEGEND */
.heatmap-legend {
    display: flex; align-items: center; gap: 8px;
    margin-top: 8px; font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #4a5470;
}
.legend-bar {
    flex: 1; height: 6px; border-radius: 3px;
    background: linear-gradient(to right, #000080, #0000ff, #00ffff, #ffff00, #ff0000);
}

/* SELECTBOX */
[data-testid="stSelectbox"] > div > div {
    background: #1a2435 !important;
    border: 1px solid #2a3a55 !important;
    color: #d0d6e0 !important;
    border-radius: 6px !important;
}

/* RADIO */
[data-testid="stRadio"] label { color: #8a9ab0 !important; font-size: 13px !important; }
[data-testid="stRadio"] [data-checked="true"] label { color: #e0e6f0 !important; }
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

# ── LAB colour helpers (pure numpy, matching OpenCV uint8 encoding) ──────────

def _rgb2lab_u8(img_np: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB HxWx3 → uint8 LAB HxWx3 using OpenCV's encoding.
    L channel: L_real * 255/100  (0–255)
    A channel: A_real + 128      (0–255)
    B channel: B_real + 128      (0–255)
    """
    rgb = img_np.astype(np.float32) / 255.0
    # sRGB linearisation
    lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    # Linear RGB → XYZ (D65)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    xyz = (lin.reshape(-1, 3) @ M.T).reshape(img_np.shape)
    # Normalise by D65 white point
    xyz /= np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    # f(t)
    eps = (6.0 / 29.0) ** 3
    f = np.where(xyz > eps, xyz ** (1.0 / 3.0), (xyz / (3.0 * (6.0/29.0)**2)) + 4.0/29.0)
    L = 116.0 * f[..., 1] - 16.0
    A = 500.0 * (f[..., 0] - f[..., 1])
    B = 200.0 * (f[..., 1] - f[..., 2])
    L_enc = np.clip(L * 255.0 / 100.0, 0, 255)
    A_enc = np.clip(A + 128.0, 0, 255)
    B_enc = np.clip(B + 128.0, 0, 255)
    return np.stack([L_enc, A_enc, B_enc], axis=-1).astype(np.uint8)


def _lab2rgb_u8(lab_u8: np.ndarray) -> np.ndarray:
    """Convert uint8 LAB HxWx3 (OpenCV encoding) → uint8 RGB HxWx3."""
    lab = lab_u8.astype(np.float32)
    L = lab[..., 0] * 100.0 / 255.0
    A = lab[..., 1] - 128.0
    B = lab[..., 2] - 128.0
    fy = (L + 16.0) / 116.0
    fx = A / 500.0 + fy
    fz = fy - B / 200.0
    eps = 6.0 / 29.0
    xyz = np.stack([
        np.where(fx > eps, fx ** 3, 3.0 * eps**2 * (fx - 4.0/29.0)),
        np.where(fy > eps, fy ** 3, 3.0 * eps**2 * (fy - 4.0/29.0)),
        np.where(fz > eps, fz ** 3, 3.0 * eps**2 * (fz - 4.0/29.0)),
    ], axis=-1)
    xyz *= np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    M_inv = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                      [-0.9692660,  1.8760108,  0.0415560],
                      [ 0.0556434, -0.2040259,  1.0572252]], dtype=np.float32)
    rgb = (xyz.reshape(-1, 3) @ M_inv.T).reshape(xyz.shape)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = np.where(rgb > 0.0031308, 1.055 * rgb ** (1.0/2.4) - 0.055, 12.92 * rgb)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


# ── Stain Normalization ───────────────────────────────────────────────────────

def macenko_normalize(img_np: np.ndarray, beta: float = 0.15, alpha: float = 1.0) -> np.ndarray:
    """
    Macenko stain normalization.
    Separates H&E (or trichrome) stain vectors via SVD on the optical density space,
    then projects to a reference stain matrix with fixed maximum concentrations.
    Works well for trichrome: separates haematoxylin (nuclei) from collagen/cytoplasm channels.
    """
    img = img_np.astype(np.float32)
    img = np.clip(img, 1, 255)
    OD = -np.log(img / 255.0)

    # Flatten and filter background (low OD = bright/white pixels)
    OD_flat = OD.reshape(-1, 3)
    mask = (OD_flat > beta).any(axis=1)
    OD_tissue = OD_flat[mask]

    if OD_tissue.shape[0] < 10:
        return img_np  # not enough tissue — return as-is

    # SVD to find stain plane
    _, _, V = np.linalg.svd(OD_tissue, full_matrices=False)
    V = V[:2].T  # first two principal components

    # Project onto plane and find angle of each pixel
    that = OD_tissue @ V
    phi = np.arctan2(that[:, 1], that[:, 0])

    # Reference stain vectors from percentile extremes
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    v1 = V @ np.array([np.cos(minPhi), np.sin(minPhi)])
    v2 = V @ np.array([np.cos(maxPhi), np.sin(maxPhi)])

    # Ensure v1 is the haematoxylin stain (larger OD in first channel)
    if v1[0] < v2[0]:
        v1, v2 = v2, v1

    HE = np.stack([v1, v2], axis=1)

    # Target reference stain matrix (Ruifrok & Johnston standard values)
    HE_ref = np.array([[0.5626, 0.2159],
                        [0.7201, 0.8012],
                        [0.4062, 0.5581]])

    # Normalise stain vectors
    HE     = HE     / (np.linalg.norm(HE,     axis=0) + 1e-6)
    HE_ref = HE_ref / (np.linalg.norm(HE_ref, axis=0) + 1e-6)

    # Solve for concentrations
    OD_all = OD_flat
    C, _, _, _ = np.linalg.lstsq(HE, OD_all.T, rcond=None)

    # Scale to reference max concentrations
    maxC = np.percentile(C, 99, axis=1, keepdims=True)
    maxC_ref = np.array([[1.9705], [1.0308]])
    C_norm = C * (maxC_ref / (maxC + 1e-6))

    # Reconstruct normalised image
    OD_norm = HE_ref @ C_norm
    img_norm = np.exp(-OD_norm.T) * 255.0
    img_norm = np.clip(img_norm, 0, 255).astype(np.uint8)
    img_norm = img_norm.reshape(img_np.shape)
    return img_norm


def reinhard_normalize(img_np: np.ndarray) -> np.ndarray:
    """
    Reinhard colour normalisation in LAB space.
    Matches the mean and std of L, A, B channels to a trichrome reference.
    Fast, good for correcting global illumination/scanner differences.
    Reference stats derived from a representative trichrome slide.
    """
    # Target LAB statistics (trichrome reference from literature)
    target_mean = np.array([74.02, 12.60, -6.48])
    target_std  = np.array([18.77,  8.32,  5.11])

    lab = _rgb2lab_u8(img_np).astype(np.float32)
    for i in range(3):
        src_mean = lab[:, :, i].mean()
        src_std  = lab[:, :, i].std() + 1e-6
        lab[:, :, i] = (lab[:, :, i] - src_mean) / src_std * target_std[i] + target_mean[i]

    lab = np.clip(lab, [0, -127, -127], [100, 127, 127]).astype(np.float32)
    lab_u8 = lab.astype(np.uint8)
    result = _lab2rgb_u8(lab_u8)
    return result


def vahadane_normalize(img_np: np.ndarray) -> np.ndarray:
    """
    Vahadane structure-preserving stain normalization (simplified version).
    Uses sparse non-negative matrix factorization concept approximated via
    iterative OD decomposition. Preserves tissue structure better than Reinhard,
    faster than full SPAMS-based NMF.
    """
    img = np.clip(img_np, 1, 255).astype(np.float32)
    OD = -np.log(img / 255.0 + 1e-6)

    OD_flat = OD.reshape(-1, 3)
    mask = (OD_flat > 0.15).any(axis=1)
    OD_tissue = OD_flat[mask]

    if OD_tissue.shape[0] < 10:
        return img_np

    # NMF-like: use k-means to find 2 stain prototypes
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=2, n_init=3, random_state=0)
    km.fit(OD_tissue)
    W = km.cluster_centers_.T  # 3×2

    # Sort: haematoxylin first (highest OD in blue channel)
    if W[2, 0] < W[2, 1]:
        W = W[:, [1, 0]]

    # Reference stain matrix
    W_ref = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
    W     = W     / (np.linalg.norm(W,     axis=0) + 1e-6)
    W_ref = W_ref / (np.linalg.norm(W_ref, axis=0) + 1e-6)

    C, _, _, _ = np.linalg.lstsq(W, OD_flat.T, rcond=None)
    maxC = np.percentile(C, 99, axis=1, keepdims=True)
    maxC_ref = np.array([[1.9705], [1.0308]])
    C_norm = C * (maxC_ref / (maxC + 1e-6))

    OD_norm = W_ref @ C_norm
    img_norm = np.exp(-OD_norm.T) * 255.0
    img_norm = np.clip(img_norm, 0, 255).astype(np.uint8)
    return img_norm.reshape(img_np.shape)


NORM_METHODS = {
    "None (original)": None,
    "Reinhard (LAB colour)": "reinhard",
    "Macenko (stain separation)": "macenko",
    "Vahadane (structure-preserving)": "vahadane",
}

NORM_DESCRIPTIONS = {
    "None (original)": "No normalisation applied. Use as a baseline.",
    "Reinhard (LAB colour)": "Fast global colour transfer in LAB space. Corrects scanner/illumination differences. Best for mild intensity shifts between sites.",
    "Macenko (stain separation)": "SVD-based stain vector decomposition. Separates haematoxylin and collagen channels independently. Recommended for trichrome slides with significant stain concentration variation.",
    "Vahadane (structure-preserving)": "NMF-based approach that preserves fine tissue structure during normalisation. Slower but most faithful to local histological detail.",
}


def apply_norm(img: Image.Image, method: str | None) -> Image.Image:
    if method is None:
        return img
    arr = np.array(img.convert("RGB"))
    if method == "reinhard":
        out = reinhard_normalize(arr)
    elif method == "macenko":
        out = macenko_normalize(arr)
    elif method == "vahadane":
        out = vahadane_normalize(arr)
    else:
        out = arr
    return Image.fromarray(out)


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

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
        score = output[0, class_idx]
        score.backward()

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1).squeeze(0)  # [H, W]
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.cpu().numpy()


def find_last_conv(module):
    """Recursively find the last Conv2d-containing module (layer4 equivalent)."""
    last = None
    for name, m in module.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last


def generate_gradcam_overlay(img: Image.Image, net, class_idx: int) -> Image.Image:
    """
    Runs Grad-CAM on the global branch of the vkola model.
    Returns a PIL Image with the heatmap blended over the original.
    """
    transform_gc = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    tensor = transform_gc(img).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_(False)

    # Try to hook layer4 of the global encoder (ResNet convention)
    target_layer = None
    model_inner = net.module if hasattr(net, 'module') else net
    for name, m in model_inner.named_modules():
        if 'layer4' in name and isinstance(m, torch.nn.Sequential):
            target_layer = m
    if target_layer is None:
        target_layer = find_last_conv(model_inner)
    if target_layer is None:
        return img  # fallback — no conv found

    gcam = GradCAM(model_inner, target_layer)

    try:
        cam = gcam.generate(tensor, class_idx)
    finally:
        gcam.remove()

    # Upsample CAM to image size
    cam_resized = np.array(
        Image.fromarray((cam * 255).clip(0, 255).astype(np.uint8)).resize(
            (img.width, img.height), Image.BICUBIC
        )
    ).astype(np.float32) / 255.0

    # Apply jet colormap
    colormap = cm.get_cmap('jet')
    heatmap = colormap(cam_resized)[:, :, :3]  # RGB, drop alpha
    heatmap_u8 = (heatmap * 255).astype(np.uint8)

    # Blend with original
    orig_arr = np.array(img.convert("RGB").resize((img.width, img.height)))
    overlay  = (0.55 * orig_arr + 0.45 * heatmap_u8).astype(np.uint8)
    return Image.fromarray(overlay)


# ── Model & Prediction ────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    net, _ = build_model(N_CLASS, mode=MODE, evaluation=True, path_g=MODEL_PATH)
    net.eval()
    return net

transform_infer = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def predict(img: Image.Image):
    tensor = transform_infer(img).unsqueeze(0).to(DEVICE)
    net = load_model()
    dummy_patches   = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    dummy_top_lefts = [(0, 0)]
    dummy_ratio     = (1.0, 1.0)
    with torch.no_grad():
        output, _ = net.module.forward(tensor, dummy_patches, dummy_top_lefts, dummy_ratio, mode=1)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
    return probs


# ── Groq Report ───────────────────────────────────────────────────────────────

def get_unified_report(images, all_probs, all_preds, avg_probs, consensus_pred, consensus_conf, norm_method_label):
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

    norm_note = f"\nStain normalisation applied before inference: {norm_method_label}." if norm_method_label != "None (original)" else ""

    prompt = f"""You are an expert nephropathologist and clinical AI assistant analyzing trichrome-stained kidney biopsy image(s).

Automated Model Output:
- Consensus Grade: {CLASS_NAMES[consensus_pred]} ({CLASS_RANGE[consensus_pred]})
- Consensus Confidence: {consensus_conf:.1f}%
- Averaged probability breakdown:
{avg_breakdown}
{multi_note}{norm_note}
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


# ── Session State ─────────────────────────────────────────────────────────────
for key in ["imgs", "all_probs", "all_preds", "norm_imgs", "last_norm", "gradcam_overlays"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── TOP BAR ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-brand">
        <div class="topbar-logo">KF</div>
        <div>
            <div class="topbar-name">Kidney Fibrosis Grader</div>
            <div class="topbar-desc">Automated Interstitial Fibrosis Analysis &nbsp;·&nbsp; ResNet-FPN + Grad-CAM</div>
        </div>
    </div>
    <div class="topbar-pills">
        <span class="tpill tpill-blue">ResNet-FPN</span>
        <span class="tpill tpill-green">95% Accuracy</span>
        <span class="tpill tpill-purple">Grad-CAM</span>
        <span class="tpill tpill-amber">Research Only</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Analysis", "Explainability", "Stain Normalisation", "Pathology Report"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Analysis (unchanged layout + norm selector)
# ─────────────────────────────────────────────────────────────────────────────
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

        # Stain norm selector (in the left column, compact)
        st.markdown('<div class="sec-label" style="margin-top:16px;">Stain Normalisation</div>', unsafe_allow_html=True)
        norm_label = st.selectbox(
            "Normalisation method",
            list(NORM_METHODS.keys()),
            label_visibility="collapsed",
            key="norm_selector"
        )
        norm_key = NORM_METHODS[norm_label]
        st.markdown(f'<div class="norm-info">{NORM_DESCRIPTIONS[norm_label]}</div>', unsafe_allow_html=True)

        if uploaded_files:
            raw_imgs = [Image.open(f).convert("RGB") for f in uploaded_files[:3]]

            # Recompute normalised images if inputs or method changed
            if raw_imgs != st.session_state.imgs or norm_label != st.session_state.last_norm:
                st.session_state.imgs = raw_imgs
                st.session_state.last_norm = norm_label
                st.session_state.all_probs = None
                st.session_state.all_preds = None
                st.session_state.gradcam_overlays = None
                with st.spinner("Applying stain normalisation..."):
                    st.session_state.norm_imgs = [apply_norm(im, norm_key) for im in raw_imgs]

            norm_imgs = st.session_state.norm_imgs

            st.markdown('<div class="sec-label" style="margin-top:14px;">Input Preview</div>', unsafe_allow_html=True)
            if len(norm_imgs) == 1:
                st.image(norm_imgs[0], use_column_width=True,
                         caption="Normalised" if norm_key else "Original")
            else:
                thumb_cols = st.columns(len(norm_imgs))
                for i, (tc, im) in enumerate(zip(thumb_cols, norm_imgs)):
                    with tc:
                        st.image(im, use_column_width=True, caption=f"Img {i+1}")
        else:
            st.session_state.imgs = None
            st.session_state.norm_imgs = None
            st.session_state.last_norm = None
            st.session_state.all_probs = None
            st.session_state.all_preds = None
            st.session_state.gradcam_overlays = None

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

        if st.session_state.norm_imgs is not None and st.session_state.all_probs is None:
            with st.spinner("Analyzing..."):
                try:
                    all_probs = [predict(im) for im in st.session_state.norm_imgs]
                    all_preds = [int(np.argmax(p)) for p in all_probs]
                    st.session_state.all_probs = all_probs
                    st.session_state.all_preds = all_preds
                except Exception as e:
                    st.error(f"Inference error: {str(e)}")

        if st.session_state.all_probs is not None:
            all_probs = st.session_state.all_probs
            all_preds = st.session_state.all_preds
            n = len(all_probs)

            avg_probs = np.mean(all_probs, axis=0)
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

        elif st.session_state.norm_imgs is None:
            st.markdown("""
<div class="await-wrap">
    <div class="await-label">No Result Yet</div>
    <div class="await-sub">Upload 1–3 biopsy images to run the analysis</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Grad-CAM Explainability
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("""
<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:18px;">
    <div>
        <div style="font-family:'Playfair Display',serif; font-size:18px; font-weight:700; color:#e0e6f0;">Grad-CAM Explainability</div>
        <div style="font-size:12px; color:#5a6480; margin-top:4px;">Gradient-weighted Class Activation Mapping — shows which regions drove the prediction</div>
    </div>
    <div style="font-family:'IBM Plex Mono',monospace; font-size:9px; font-weight:500; letter-spacing:0.08em;
                background:#2a1a4a; color:#c084fc; border:1px solid #4a2a80; padding:4px 10px; border-radius:4px;">
        GRAD-CAM &nbsp;·&nbsp; LAST CONV LAYER</div>
</div>
""", unsafe_allow_html=True)

    if st.session_state.norm_imgs is None or st.session_state.all_preds is None:
        st.markdown("""
<div class="await-wrap">
    <div class="await-label">Analysis Required</div>
    <div class="await-sub">Run an analysis in the Analysis tab first, then return here for Grad-CAM visualisation.</div>
</div>
""", unsafe_allow_html=True)
    else:
        norm_imgs  = st.session_state.norm_imgs
        all_preds  = st.session_state.all_preds
        n = len(norm_imgs)

        # Which class to visualise
        gc_col1, gc_col2 = st.columns([1.2, 2.8], gap="large")
        with gc_col1:
            st.markdown('<div class="sec-label">CAM Settings</div>', unsafe_allow_html=True)
            target_class_label = st.selectbox(
                "Target class for CAM",
                [f"{CLASS_NAMES[i]} — {CLASS_RANGE[i]}" for i in range(4)],
                index=int(np.argmax(np.mean(st.session_state.all_probs, axis=0))),
                label_visibility="collapsed",
                key="cam_class"
            )
            target_class_idx = [f"{CLASS_NAMES[i]} — {CLASS_RANGE[i]}" for i in range(4)].index(target_class_label)

            st.markdown(f"""
<div class="norm-info" style="margin-top:12px;">
    <strong>What you're seeing:</strong><br>
    Red/yellow regions had the highest gradient activation for the <em>{CLASS_NAMES[target_class_idx]}</em> class.
    These are the tissue areas that most strongly influenced the model's decision.
    Blue/cool areas had low influence.<br><br>
    <strong>For pathologists:</strong> check whether highlighted regions correspond to
    areas of collagen deposition, tubular atrophy, or other histological findings
    consistent with the predicted grade.
</div>
""", unsafe_allow_html=True)

            st.markdown("""
<div class="heatmap-legend">
    <span>Low</span>
    <div class="legend-bar"></div>
    <span>High</span>
</div>
<div style="font-family:'IBM Plex Mono',monospace; font-size:10px; color:#3a4460; margin-top:4px;">
Activation intensity
</div>
""", unsafe_allow_html=True)

        with gc_col2:
            st.markdown('<div class="sec-label">Activation Maps</div>', unsafe_allow_html=True)

            # Generate Grad-CAM (cached per run)
            cam_key = (id(norm_imgs[0]), target_class_idx, st.session_state.last_norm)
            if (st.session_state.gradcam_overlays is None or
                    st.session_state.get("last_cam_key") != cam_key):
                with st.spinner("Computing Grad-CAM..."):
                    net = load_model()
                    model_inner = net.module if hasattr(net, 'module') else net
                    overlays = []
                    for im in norm_imgs:
                        try:
                            overlay = generate_gradcam_overlay(im, model_inner, target_class_idx)
                        except Exception as e:
                            st.warning(f"Grad-CAM failed for one image: {e}")
                            overlay = im
                        overlays.append(overlay)
                    st.session_state.gradcam_overlays = overlays
                    st.session_state["last_cam_key"] = cam_key

            overlays = st.session_state.gradcam_overlays
            if len(overlays) == 1:
                oc1, oc2 = st.columns(2)
                with oc1:
                    st.image(norm_imgs[0], use_column_width=True, caption="Original (normalised)")
                with oc2:
                    st.image(overlays[0], use_column_width=True,
                             caption=f"Grad-CAM → {CLASS_NAMES[target_class_idx]}")
            else:
                for i, (orig, ov) in enumerate(zip(norm_imgs, overlays)):
                    oc1, oc2 = st.columns(2)
                    with oc1:
                        st.image(orig, use_column_width=True, caption=f"Image {i+1} — Original")
                    with oc2:
                        st.image(ov, use_column_width=True,
                                 caption=f"Image {i+1} — Grad-CAM ({CLASS_NAMES[target_class_idx]})")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Stain Normalisation Comparison
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:18px;">
    <div>
        <div style="font-family:'Playfair Display',serif; font-size:18px; font-weight:700; color:#e0e6f0;">Stain Normalisation Comparison</div>
        <div style="font-size:12px; color:#5a6480; margin-top:4px;">Compare all methods side-by-side and evaluate their effect on model confidence</div>
    </div>
</div>
""", unsafe_allow_html=True)

    if st.session_state.imgs is None:
        st.markdown("""
<div class="await-wrap">
    <div class="await-label">No Images Uploaded</div>
    <div class="await-sub">Upload images in the Analysis tab first.</div>
</div>
""", unsafe_allow_html=True)
    else:
        # Pick which image to compare (if multiple)
        raw_imgs = st.session_state.imgs
        if len(raw_imgs) > 1:
            img_idx = st.selectbox(
                "Select image to compare",
                [f"Image {i+1}" for i in range(len(raw_imgs))],
                label_visibility="collapsed",
                key="norm_compare_img"
            )
            img_to_compare = raw_imgs[int(img_idx.split()[-1]) - 1]
        else:
            img_to_compare = raw_imgs[0]

        st.markdown('<div class="sec-label" style="margin-bottom:14px;">Side-by-side visual comparison</div>', unsafe_allow_html=True)

        method_keys   = list(NORM_METHODS.keys())
        method_values = list(NORM_METHODS.values())

        with st.spinner("Rendering all normalisation methods..."):
            normed = []
            probs_per_method = []
            for label, key in zip(method_keys, method_values):
                nim = apply_norm(img_to_compare, key)
                normed.append(nim)
                p = predict(nim)
                probs_per_method.append(p)

        # Show images in a 2×2 grid
        row1 = st.columns(2)
        row2 = st.columns(2)
        rows = [row1[0], row1[1], row2[0], row2[1]]

        for i, (col, label, nim, probs) in enumerate(zip(rows, method_keys, normed, probs_per_method)):
            pred_idx  = int(np.argmax(probs))
            pred_conf = probs[pred_idx] * 100
            color     = CLASS_COLORS[pred_idx]
            with col:
                st.image(nim, use_column_width=True, caption=label)
                st.markdown(f"""
<div style="background:#1a2435; border:1px solid #2a3a55; border-radius:6px; padding:10px 12px; margin-top:2px;">
    <div style="font-family:'IBM Plex Mono',monospace; font-size:10px; color:#4a5470; margin-bottom:4px;">MODEL OUTPUT</div>
    <div style="font-size:13px; font-weight:600; color:{color};">{CLASS_NAMES[pred_idx]}</div>
    <div style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#5a6480; margin-top:2px;">Confidence: {pred_conf:.1f}%</div>
    {''.join(f'<div class="prob-row" style="margin-top:4px;"><div class="prob-name" style="width:58px;">{CLASS_NAMES[j][:3]}</div><div class="prob-track" style="flex:1;height:4px;background:#0e1825;border-radius:2px;overflow:hidden;"><div style="height:100%;width:{probs[j]*100:.1f}%;background:{CLASS_COLORS[j]};border-radius:2px;"></div></div><div class="prob-pct">{probs[j]*100:.0f}%</div></div>' for j in range(4))}
</div>
""", unsafe_allow_html=True)

        # Summary table
        st.markdown('<div class="sec-label" style="margin-top:24px; margin-bottom:12px;">Prediction summary across methods</div>', unsafe_allow_html=True)
        rows_html = ""
        for label, probs in zip(method_keys, probs_per_method):
            pi   = int(np.argmax(probs))
            conf = probs[pi] * 100
            c    = CLASS_COLORS[pi]
            rows_html += f"""
<tr>
  <td style="padding:8px 12px; font-family:'IBM Plex Mono',monospace; font-size:11px; color:#8a9ab0; border-bottom:1px solid #1e2a3a;">{label}</td>
  <td style="padding:8px 12px; font-size:12px; font-weight:600; color:{c}; border-bottom:1px solid #1e2a3a;">{CLASS_NAMES[pi]}</td>
  <td style="padding:8px 12px; font-family:'IBM Plex Mono',monospace; font-size:11px; color:#5a6480; border-bottom:1px solid #1e2a3a;">{conf:.1f}%</td>
  <td style="padding:8px 12px; font-family:'IBM Plex Mono',monospace; font-size:11px; color:#5a6480; border-bottom:1px solid #1e2a3a;">{CLASS_RANGE[pi]}</td>
</tr>"""
        st.markdown(f"""
<table style="width:100%; background:#202b3d; border:1px solid #2a3549; border-radius:8px; border-collapse:collapse;">
  <thead>
    <tr style="background:#1a2333;">
      <th style="padding:10px 12px; font-family:'IBM Plex Mono',monospace; font-size:10px; letter-spacing:0.1em; color:#4a5470; text-align:left; text-transform:uppercase; border-bottom:2px solid #2a3549;">Method</th>
      <th style="padding:10px 12px; font-family:'IBM Plex Mono',monospace; font-size:10px; letter-spacing:0.1em; color:#4a5470; text-align:left; text-transform:uppercase; border-bottom:2px solid #2a3549;">Grade</th>
      <th style="padding:10px 12px; font-family:'IBM Plex Mono',monospace; font-size:10px; letter-spacing:0.1em; color:#4a5470; text-align:left; text-transform:uppercase; border-bottom:2px solid #2a3549;">Confidence</th>
      <th style="padding:10px 12px; font-family:'IBM Plex Mono',monospace; font-size:10px; letter-spacing:0.1em; color:#4a5470; text-align:left; text-transform:uppercase; border-bottom:2px solid #2a3549;">Range</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Pathology Report (unchanged except passes norm_label)
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    if (st.session_state.norm_imgs is not None and
            st.session_state.all_probs is not None and
            st.session_state.all_preds is not None):

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
            for i, im in enumerate(st.session_state.norm_imgs):
                caption = f"Image {i+1} — {CLASS_NAMES[all_preds[i]]}" if len(st.session_state.norm_imgs) > 1 else "Analyzed biopsy image"
                st.image(im, use_column_width=True, caption=caption)

        with rcol2:
            with st.spinner("Generating pathology report..."):
                try:
                    norm_method_label = st.session_state.last_norm or "None (original)"
                    report = get_unified_report(
                        images=st.session_state.norm_imgs,
                        all_probs=all_probs,
                        all_preds=all_preds,
                        avg_probs=avg_probs,
                        consensus_pred=consensus_pred,
                        consensus_conf=consensus_conf,
                        norm_method_label=norm_method_label,
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

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <span>KIDNEY FIBROSIS GRADER &nbsp;·&nbsp; ResNet-FPN + Grad-CAM &nbsp;·&nbsp; 95% Test Accuracy</span>
    <span>Research use only — not validated for clinical diagnosis</span>
</div>
""", unsafe_allow_html=True)
