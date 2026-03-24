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
| Report generation | Groq API with Llama 4 Scout |
| Interface | Streamlit web app |

**Grading scale:**
- **Grade 0** — No fibrosis (less than 5% cortical area)
- **Grade 1** — Mild fibrosis (5-25%)
- **Grade 2** — Moderate fibrosis (26-50%)
- **Grade 3** — Severe fibrosis (more than 50%)

---

## Architecture

Adapts the [vkola-lab/ajpa2021](https://github.com/vkola-lab/ajpa2021) ResNet-FPN, originally built for whole-slide images (.svs), for standard JPG/PNG input. OpenSlide replaced with PIL — no WSI dependencies, runs on standard hardware including CPU-only.

```
Input image (JPG/PNG)
    -> PIL preprocessing + transforms
    -> ResNet-50 backbone
    -> Feature Pyramid Network
    -> Global average pooling
    -> 4-class softmax classifier
    -> Grade + confidence scores
    -> Groq / Llama 4 Scout
    -> Structured pathology report
```

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
GROQ_API_KEY=your_key_here     # required for LLM report generation
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)

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

Python · PyTorch · ResNet-FPN · Streamlit · Groq API · Llama 4 Scout · PIL

**License:** MIT
