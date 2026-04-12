"""
inference.py — Model loading and inference for AUBMC Nephropathology App
========================================================================
Two models:
  1. IFTARegressor   — EfficientNet-B3, single trichrome image → IFTA % (0-100)
  2. IFClassifier    — ResNet-50 (9-channel IF stack) → diagnosis (9 classes)

Usage:
    from inference import IFClassifier

    clf_model  = IFClassifier("if_classifier_best.pth")
    result     = clf_model.predict({"IgG": "igg.jpg", "C3": "c3.jpg"})
"""

import re
import torch
import torch.nn as nn
import torchvision.models as tvm
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Union

# ── Constants ────────────────────────────────────────────────────────────────

IMG_SIZE = 224

IF_CHANNELS = ['IgG', 'IgA', 'IgM', 'C3', 'C1q', 'kappa', 'lambda', 'fibrinogen', 'albumin']
N_IF        = len(IF_CHANNELS)

IF_CLASSES = [
    'transplant',
    'membranous_nephropathy',
    'lupus_nephritis',
    'FSGS',
    'IgA_nephropathy',
    'amyloidosis',
    'crescentic_GN',
    'diabetic_nephropathy',
    'minimal_change_disease',
]

# Friendly display names
CLASS_DISPLAY = {
    'transplant':              'Transplant nephropathy',
    'membranous_nephropathy':  'Membranous nephropathy',
    'lupus_nephritis':         'Lupus nephritis',
    'FSGS':                    'FSGS',
    'IgA_nephropathy':         'IgA nephropathy',
    'amyloidosis':             'Amyloidosis',
    'crescentic_GN':           'Crescentic GN',
    'diabetic_nephropathy':    'Diabetic nephropathy',
    'minimal_change_disease':  'Minimal change disease',
}

# Filename → IF marker auto-detection
IF_FILENAME_PATTERNS = {
    'IgG':        re.compile(r'(?:^|\b)ig\s*g(?:\b|\s|\d|$)', re.IGNORECASE),
    'IgA':        re.compile(r'(?:^|\b)ig\s*a(?:\b|\s|\d|$)', re.IGNORECASE),
    'IgM':        re.compile(r'(?:^|\b)ig\s*m(?:\b|\s|\d|$)', re.IGNORECASE),
    'C1q':        re.compile(r'(?:^|\b)c\s*1\s*q(?:\b|\s|\d|$)', re.IGNORECASE),
    'C3':         re.compile(r'(?:^|\b)c\s*3(?:\b|\s|\d|$)', re.IGNORECASE),
    'kappa':      re.compile(r'(?:^|\b)kappa(?:\b|\s|\d|$)', re.IGNORECASE),
    'lambda':     re.compile(r'(?:^|\b)lambda(?:\b|\s|\d|$)', re.IGNORECASE),
    'fibrinogen': re.compile(r'(?:^|\b)fibrin(?:ogen)?(?:\b|\s|\d|$)', re.IGNORECASE),
    'albumin':    re.compile(r'(?:^|\b)albumin(?:\b|\s|$)', re.IGNORECASE),
}


def detect_marker_from_filename(path: Union[str, Path]) -> Optional[str]:
    """Auto-detect IF marker from filename. C1q checked before C3."""
    stem = Path(path).stem
    priority = ['C1q', 'IgG', 'IgA', 'IgM', 'C3', 'kappa', 'lambda', 'fibrinogen', 'albumin']
    for marker in priority:
        if IF_FILENAME_PATTERNS[marker].search(stem):
            return marker
    return None


# ── Image utilities ──────────────────────────────────────────────────────────

def load_gray_normalized(path: Union[str, Path], size: int = IMG_SIZE) -> np.ndarray:
    """Load image as grayscale float32 array, normalized on non-background pixels."""
    img = Image.open(path).convert('L').resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    nz  = arr[arr > 0.05]
    if len(nz) > 50:
        arr = (arr - nz.mean()) / (nz.std() + 1e-6)
    return arr


# ── Model 2: IF Diagnosis Classifier ────────────────────────────────────────

class IFClassifier:
    """
    ResNet-50 with 9-channel IF input → diagnosis (9 classes).
    Checkpoint: if_classifier_best.pth (saved by Colab training script).
    Missing channels are filled with zero planes.
    """

    def __init__(self, checkpoint_path: Union[str, Path], device: Optional[str] = None):
        self.device  = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.classes = IF_CLASSES
        self.model   = self._build()
        self._load(checkpoint_path)
        self.model.eval()

    def _build(self) -> nn.Module:
        model   = tvm.resnet50(weights=None)
        old     = model.conv1
        new     = nn.Conv2d(N_IF, old.out_channels,
                            kernel_size=old.kernel_size,
                            stride=old.stride,
                            padding=old.padding, bias=False)
        model.conv1 = new
        model.fc    = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.fc.in_features, len(self.classes))
        )
        return model.to(self.device)

    def _load(self, path: Union[str, Path]):
        ckpt  = torch.load(path, map_location=self.device)
        state = ckpt.get('model_state_dict', ckpt)
        # If checkpoint recorded classes/channels, use them
        if isinstance(ckpt, dict) and 'classes' in ckpt:
            self.classes = ckpt['classes']
        self.model.load_state_dict(state)

    def _build_stack(self, channel_paths: dict) -> torch.Tensor:
        """
        channel_paths: {marker: image_path} for however many channels are present.
        Returns (1, N_IF, H, W) tensor with zeros for missing channels.
        """
        planes = []
        for ch in IF_CHANNELS:
            path = channel_paths.get(ch)
            if path is not None:
                planes.append(load_gray_normalized(path))
            else:
                planes.append(np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32))
        stack = np.stack(planes, axis=0)                          # (N_IF, H, W)
        return torch.from_numpy(stack).unsqueeze(0).to(self.device)  # (1, N_IF, H, W)

    @torch.no_grad()
    def predict(
        self,
        channel_paths: dict,
        top_k: int = 3,
    ) -> dict:
        """
        Args:
            channel_paths: {marker: filepath}  e.g. {'IgG': '/tmp/igg.jpg', 'C3': '/tmp/c3.jpg'}
            top_k: number of top predictions to return

        Returns:
            {
              'top_predictions': [
                  {'diagnosis': 'IgA_nephropathy', 'display': 'IgA nephropathy',
                   'confidence': 0.72, 'pct': 72.0},
                  ...
              ],
              'channels_used': ['IgG', 'C3', 'kappa'],
              'channels_missing': ['IgA', 'IgM', 'C1q', 'lambda', 'fibrinogen', 'albumin'],
              'n_channels': 3,
              'warning': None  # or string if < 3 channels provided
            }
        """
        channels_used    = [ch for ch in IF_CHANNELS if ch in channel_paths]
        channels_missing = [ch for ch in IF_CHANNELS if ch not in channel_paths]

        warning = None
        if len(channels_used) < 3:
            warning = (f"Only {len(channels_used)} channel(s) provided. "
                       f"Confidence may be low. Recommend ≥3 channels for reliable results.")

        tensor  = self._build_stack(channel_paths)
        logits  = self.model(tensor).squeeze(0)
        probs   = torch.softmax(logits, dim=0).cpu().numpy()

        top_idx = np.argsort(probs)[::-1][:top_k]
        top_preds = [
            {
                'diagnosis':  self.classes[i],
                'display':    CLASS_DISPLAY.get(self.classes[i], self.classes[i]),
                'confidence': round(float(probs[i]), 4),
                'pct':        round(float(probs[i]) * 100, 1),
            }
            for i in top_idx
        ]

        return {
            'top_predictions':  top_preds,
            'channels_used':    channels_used,
            'channels_missing': channels_missing,
            'n_channels':       len(channels_used),
            'warning':          warning,
        }

    def predict_from_files(
        self,
        image_paths: list,
        top_k: int = 3,
    ) -> dict:
        """
        Convenience method: auto-detect marker from each filename.
        Unrecognized filenames are ignored.
        """
        channel_paths = {}
        unrecognized  = []
        for p in image_paths:
            marker = detect_marker_from_filename(p)
            if marker and marker not in channel_paths:
                channel_paths[marker] = p
            elif not marker:
                unrecognized.append(Path(p).name)

        result = self.predict(channel_paths, top_k=top_k)
        if unrecognized:
            result['unrecognized_files'] = unrecognized
        return result
