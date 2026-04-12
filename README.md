# Kidney Fibrosis Grader

**Deep learning classifier for kidney biopsy fibrosis grading with AI-generated pathology reports**

> Upload a trichrome-stained kidney biopsy image. Get a fibrosis grade (0-3), confidence scores, and a structured pathology report written by an LLM in seconds.

---

## What it does

Interstitial fibrosis and tubular atrophy (IFTA) grading is a critical step in kidney biopsy assessment, but it is time-consuming, subjective, and requires a trained pathologist. This tool automates the visual grading step using a ResNet-FPN deep learning model and generates a clinical-style report via a large language model.

| Component | Details |
|---|---|
| Architecture | ResNet-50 Feature Pyramid Network |
| Task | 4-class fibrosis grading: Grade 0 / 1 / 2 / 3 |
| Input | Trichrome-stained kidney biopsy image (JPG or PNG) |
| Stain normalisation | Macenko · Reinhard · Vahadane (built-in, no extra dependencies) |
| Report generation | OpenRouter API with Llama 4 Scout |
| Interface | Streamlit web app |

**Grading scale:**
- **Grade 0**: No fibrosis (less than 5% cortical area)
- **Grade 1**: Mild fibrosis (5-25%)
- **Grade 2**: Moderate fibrosis (26-50%)
- **Grade 3**: Severe fibrosis (more than 50%)

---

## Architecture

Adapts the [vkola-lab/ajpa2021](https://github.com/vkola-lab/ajpa2021) ResNet-FPN, originally built for whole-slide images (.svs), for standard JPG/PNG input. OpenSlide replaced with PIL, no WSI dependencies, runs on standard hardware including CPU-only.

```
Input image (JPG/PNG)
    -> [Optional] Stain normalisation (Macenko / Reinhard / Vahadane)
    -> PIL preprocessing + transforms
    -> ResNet-50 backbone
    -> Feature Pyramid Network
    -> Global average pooling
    -> 4-class softmax classifier
    -> Grade + confidence scores
    -> OpenRouter / Llama 4 Scout
    -> Structured pathology report
```

---

## Stain Normalisation

Trichrome staining intensity varies across laboratories, which can shift colour distributions at the same fibrosis grade and cause the model to mis-predict. The built-in **Stain Normalisation** tab lets you transfer a reference lab's stain profile onto your source image before running inference, reducing cross-site domain shift.

Three algorithms are available, all implemented from scratch with NumPy/OpenCV (no additional dependencies):

| Method | Approach | Reference |
|---|---|---|
| **Macenko** | SVD-based stain-vector separation in optical-density space, most accurate for trichrome | Macenko et al., *A method for normalizing histology slides for quantitative analysis*, ISBI 2009 · [doi:10.1109/ISBI.2009.5193250](https://doi.org/10.1109/ISBI.2009.5193250) |
| **Reinhard** | Per-channel mean/std transfer in LAB colour space, fast and simple | Reinhard et al., *Color transfer between images*, IEEE CG&A 2001 · [doi:10.1109/38.946629](https://doi.org/10.1109/38.946629) |
| **Vahadane** | Full LAB histogram matching via quantile interpolation, robust to outliers | Vahadane et al., *Structure-Preserving Color Normalization and Sparse Stain Separation for Histological Images*, IEEE TMI 2016 · [doi:10.1109/TMI.2016.2529665](https://doi.org/10.1109/TMI.2016.2529665) |

---

## How to run

### Clone and run locally

```bash
git clone https://github.com/MarkBarsoumMarkarian/kidney-fibrosis-grader
cd kidney-fibrosis-grader
pip install -r requirements.txt
python download_model.py        # downloads weights from Google Drive
streamlit run app.py
```

### Environment variables

```bash
OPENROUTER_API_KEY=your_key_here    # required for LLM report generation and IF panel safety review
```

Get a free OpenRouter API key at [openrouter.ai](https://openrouter.ai)

For local development, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in your key.  
For Streamlit Cloud deployments, add the key under **Settings → Secrets**.

A `.devcontainer` config is included for VS Code devcontainer or GitHub Codespaces.

---

## Limitations

- Research and educational tool only, not validated for clinical use
- Performance depends on image quality and staining consistency
- LLM-generated reports should not replace pathologist review

---

## Related

[trichrome-analyzer](https://github.com/MarkBarsoumMarkarian/trichrome-analyzer) — companion tool for pixel-level fibrosis area quantification from trichrome images.

---

## Stack

Python · PyTorch · ResNet-FPN · Streamlit · OpenRouter API · Llama 4 Scout · PIL · NumPy · OpenCV

**License:** MIT
