# ExplainableCropAI

**Interpretable Deep Learning for Agricultural Disease Detection and Decision Support**

[![GitHub Repository](https://img.shields.io/badge/GitHub-ExplainableCropAI-blue)](https://github.com/santoshray02/ExplainableCropAI)

ExplainableCropAI detects paddy (rice) crop diseases and, alongside each prediction,
produces three complementary explanations: a **Grad-CAM++ visual heatmap**, a
**Monte Carlo Dropout uncertainty** estimate, and an **LLM-generated, audience-adaptive
natural-language explanation** grounded in a curated treatment knowledge base. It is the
reference implementation for the paper submitted to *The Visual Computer* (Springer Nature).

> **Status:** under major revision. Some experimental numbers in the paper are being
> finalised from the scripts in [`experiments/`](experiments/README.md). See
> [Reproducing the results](#reproducing-the-results-paper-tables).

## Key features

- **ConvNeXt V2-Tiny backbone** (`src/models/convnext_v2.py`, `ConvNeXtV2Classifier`) — modern CNN, 28.6M parameters. A legacy ResNet50 path (`src/base_model.py`, `PaddyDiseaseClassifier`) is retained as a baseline.
- **Grad-CAM++ visual explanations** (`src/explainers/gradcam_plus.py`), with LayerCAM as an alternative and legacy Grad-CAM (`src/explainers/gradcam.py`).
- **Monte Carlo Dropout uncertainty** (`src/uncertainty/mc_dropout.py`) — 30 stochastic passes; predictive entropy + epistemic (mutual information) decomposition.
- **LLM natural-language generation** (`src/nlg/llm_generator.py`) using Phi-3-mini, with a template-based fallback (`src/nlg/template_generator.py`). Treatment content is drawn from a **curated knowledge base**, not free-generated (see [Safety](#safety--scope)).
- **Recommendation engine** (`src/recommendations/`) — severity-based treatment guidance keyed by disease.
- **Unified interface** (`src/explainable_system.py`, `ExplainableCropAI`) tying the components together; supports both the legacy and modern backbones via configuration.

## Dataset

Trained and evaluated on the public **[Paddy Doctor](https://www.kaggle.com/competitions/paddy-disease-classification) dataset** (10,407 images, 10 classes). The official class set is:

`bacterial_leaf_blight`, `bacterial_leaf_streak`, `bacterial_panicle_blight`, `blast`,
`brown_spot`, `dead_heart`, `downy_mildew`, `hispa`, `normal` (healthy), `tungro`.

> **Note:** the code uses **"Bacterial Panicle Blight"** for this class, matching the
> official dataset. Its knowledge-base treatment entry is intentionally conservative and
> referral-based (no specific dosages) — have a plant pathologist / extension service
> verify region-appropriate management before field use.

The dataset itself is **not redistributed** here (Kaggle license). To reproduce splits,
use `experiments/create_splits.py` (see below).

## Project structure

```
ExplainableCropAI/
├── src/
│   ├── models/convnext_v2.py        # ConvNeXtV2Classifier (current backbone)
│   ├── base_model.py                # PaddyDiseaseClassifier (legacy ResNet50 baseline)
│   ├── explainers/gradcam_plus.py   # Grad-CAM++ and LayerCAM
│   ├── explainers/gradcam.py        # legacy Grad-CAM
│   ├── uncertainty/mc_dropout.py    # Monte Carlo Dropout
│   ├── nlg/llm_generator.py         # Phi-3-mini LLM generator (+ template fallback)
│   ├── recommendations/             # treatment_db.py (knowledge base) + engine
│   ├── evaluation/explanation_metrics.py
│   ├── visualization/               # figure-generation scripts (illustrative)
│   └── explainable_system.py        # ExplainableCropAI — unified interface
├── experiments/                     # reproducible scripts for the paper tables (see its README)
├── notebooks/system_demo.ipynb      # end-to-end demo
├── paper/                           # LaTeX manuscript (sn-jnl.cls) + response docs
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/santoshray02/ExplainableCropAI.git
cd ExplainableCropAI
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.8+; a CUDA-capable GPU is recommended for training and for the
LLM/uncertainty passes (CPU works for small tests).

## Quick start

End-to-end demo:

```bash
jupyter notebook notebooks/system_demo.ipynb
```

Generate the illustrative interface/figure renders (these use demo data, not the trained
model outputs):

```bash
python src/visualization/generate_user_interface.py
python src/visualization/generate_gradcam_examples.py
```

## Reproducing the results (paper tables)

The scripts in [`experiments/`](experiments/README.md) regenerate the manuscript's
experimental tables on any CUDA-capable GPU. Run `create_splits.py` first; the others
consume its `splits.csv`:

| Script | Reviewer point | Paper output |
|---|---|---|
| `create_splits.py` | #6 | leakage-controlled 70/15/15 split + per-class-per-split counts |
| `run_explanation_comparison.py` | #7 | 6-method attribution comparison table |
| `run_calibration.py` | #8 | ECE/MCE/Brier + reliability diagram |
| `run_baselines_ablation.py` | #11 | baselines and ablation tables |

See `experiments/README.md` for exact commands and the output→paper mapping.

## Safety & scope

This system is **decision support, not a prescription authority**. Treatment text is
sourced from a curated knowledge base (`src/recommendations/treatment_db.py`); the LLM
rephrases that content rather than inventing dosages. When prediction confidence or
agreement is low, the system surfaces an explicit **"consult an agricultural expert"**
referral and does not assert firm treatment advice. Pesticide/chemical guidance should be
verified against local agricultural-extension recommendations before field use.

## Compiling the paper

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Citation

A DOI-archived release (Zenodo) will accompany the published paper. Until then:

```bibtex
@article{explainablecropai2026,
  title   = {ExplainableCropAI: Interpretable Deep Learning for Agricultural
             Disease Detection and Decision Support},
  author  = {Kumar, Santosh and Kumar, Sumit and Nandan, Durgesh},
  journal = {The Visual Computer (under review)},
  year    = {2026},
  note    = {ConvNeXt V2-Tiny, Grad-CAM++, Monte Carlo Dropout, Phi-3-mini}
}
```

## Contact

- **Santosh Kumar** — santoshray02@gmail.com — [@santoshray02](https://github.com/santoshray02)
- Symbiosis Institute of Technology, Symbiosis International (Deemed University), Pune

## License

Academic research use. Please cite if you build on this work.
