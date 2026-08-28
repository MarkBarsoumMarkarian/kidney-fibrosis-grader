# Updates:
Moved Project to HuggingFace for faster processing speeds: [Link](https://huggingface.co/spaces/MarkBarsoum/Kidney-IS-Predictor)
(Currently testing Precision medicine models, where the model will predict if the patient needs immunosupressive drugs or not. Will connect everything together soon))

# Kidney Fibrosis Grader

AI-assisted kidney biopsy analysis app for:
- **Trichrome fibrosis grading** (4 classes)
- **Immunofluorescence (IF) diagnosis** (9-channel classifier)
- **Clinicopathological report generation** (multimodal LLM)
- **Stain normalisation** (Macenko / Reinhard / Vahadane)

> Research and educational use only. Not validated for clinical diagnosis.

---

## What this repository contains

This project ships a Streamlit application (`app.py`) that combines classical pathology workflows with deep learning and LLM-assisted report drafting.

### Main capabilities

1. **Analysis & IF Diagnosis tab**
   - Upload **1-3 trichrome biopsy images** (`jpg/jpeg/png`)
   - Run fibrosis grading with a patched **ResNet-FPN** model
   - View consensus prediction, class probabilities, and confidence
   - Upload IF channels (IgG, IgA, IgM, C3, C1q, kappa, lambda, fibrinogen, albumin)
   - Run IF classifier (ResNet-50, 9-channel input, top-3 outputs)
   - Auto zero-fill missing IF channels
   - Generate IF mosaic and optional LLM safety review of IF pattern

2. **Clinicopathological Observation tab**
   - Generates Grad-CAM overlays for trichrome images
   - Sends trichrome image(s), overlays, and IF mosaic to an LLM
   - Produces a structured multimodal nephropathology-style report
   - Exports a multi-page PDF containing:
     - fibrosis summary + probability bars
     - IF diagnosis summary (if available)
     - image and Grad-CAM panels
     - IF mosaic page (if IF data uploaded)
     - report text page

3. **Stain Normalisation tab**
   - Source image + reference image workflow
   - Methods included:
     - Macenko (SVD stain separation)
     - Reinhard (LAB statistics transfer)
     - Vahadane-style LAB histogram matching
   - Shows before/after visual comparison
   - Computes stain metrics (LAB means, color spread)
   - Runs fibrosis inference before/after normalisation for comparison

---

## Models and outputs

### Trichrome fibrosis model
- Architecture: patched **ResNet-FPN** from the AJPA/vkola-lab lineage
- Classes:
  - Minimal (<10% fibrosis)
  - Mild (10-25%)
  - Moderate (25-50%)
  - Severe (>50%)
- Weights file: `global_only.pth`

### IF classifier
- Architecture: **ResNet-50** with 9-channel grayscale IF stack
- Output classes:
  - transplant
  - membranous_nephropathy
  - lupus_nephritis
  - FSGS
  - IgA_nephropathy
  - amyloidosis
  - crescentic_GN
  - diabetic_nephropathy
  - minimal_change_disease
- Weights file: `if_classifier_best.pth`

### Explainability and reporting
- Grad-CAM for trichrome model region saliency
- LLM report generation through **Groq API** (`meta-llama/llama-4-scout-17b-16e-instruct`)
- PDF export via `fpdf2`

---

## Repository structure

```text
.
├── app.py                      # Streamlit app (UI + inference + Grad-CAM + LLM + PDF)
├── inference.py                # IF classifier module + channel parsing helpers
├── download_model.py           # Downloads trichrome weights (global_only.pth)
├── requirements.txt            # Python dependencies
├── models/
│   ├── resnet.py
│   ├── resnet_fpn.py
│   └── resnet_fpn_patched.py
├── utils/
│   ├── metrics.py
│   ├── model_builder.py
│   └── trainer_patched.py
├── .streamlit/
│   └── secrets.toml.example
└── .devcontainer/
    └── devcontainer.json
```

---

## Setup

### 1) Clone

```bash
git clone https://github.com/MarkBarsoumMarkarian/kidney-fibrosis-grader.git
cd kidney-fibrosis-grader
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Prepare model weights

Option A (manual trichrome download helper):

```bash
python download_model.py
```

Option B (automatic):
- Running `app.py` auto-downloads missing weights for:
  - `global_only.pth`
  - `if_classifier_best.pth`

### 4) Configure secrets

Copy template:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then set keys as needed.

#### Required for LLM report and IF LLM safety review

```toml
GROQ_API_KEY = "your_groq_key"
```

You can also set numbered keys for rotation/fallback:

```toml
GROQ_API_KEY_1 = "..."
GROQ_API_KEY_2 = "..."
```

#### Optional / legacy key path still present in code

```toml
OPENROUTER_API_KEY = "your_openrouter_key"
```

### 5) Run app

```bash
streamlit run app.py
```

Default local URL is typically `http://localhost:8501`.

---

## Dependency stack

From `requirements.txt`:
- torch
- torchvision
- streamlit
- numpy
- Pillow
- opencv-python-headless
- gdown
- google-generativeai
- fpdf2
- timm

---

## Standalone IF classifier usage

`inference.py` exposes `IFClassifier` and helpers for programmatic use.

Supported marker channels:
`IgG, IgA, IgM, C3, C1q, kappa, lambda, fibrinogen, albumin`

It can:
- infer from explicit channel-to-file mappings
- auto-detect channels from filenames
- return top-k predictions, used channels, missing channels, and warnings

---

## Notes, limitations, and safety

- Research use only; not for clinical deployment.
- Predictions depend on stain quality, scanner differences, and image selection.
- Missing IF channels are zero-filled, which can lower confidence.
- LLM output can be wrong or incomplete; pathologist review is mandatory.
- If LLM features are used, image-derived data is sent to external API services.

---

## Troubleshooting

- **Model file missing**: run `python download_model.py` (trichrome) or launch `app.py` for auto-download.
- **LLM report unavailable**: check `GROQ_API_KEY` / `GROQ_API_KEY_1` in Streamlit secrets.
- **Slow startup**: first run may take longer due to model downloads.
- **No IF result**: ensure at least one IF channel image is uploaded and supported file types are used.

---

## Devcontainer / Codespaces

A `.devcontainer/devcontainer.json` is included:
- Python 3.11 base image
- installs `requirements.txt`
- auto-runs Streamlit app on attach
- forwards port `8501`

---

## Related project

- [trichrome-analyzer](https://github.com/MarkBarsoumMarkarian/trichrome-analyzer) — companion tool for pixel-level fibrosis area quantification.

---

## License

MIT
