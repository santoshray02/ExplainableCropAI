# ExplainableCropAI

**Interpretable Deep Learning for Agricultural Disease Detection and Decision Support**

[![GitHub Repository](https://img.shields.io/badge/GitHub-ExplainableCropAI-blue)](https://github.com/santoshray02/ExplainableCropAI)

## Overview

ExplainableCropAI is a research project addressing a critical challenge in agricultural technology: making AI-powered disease detection systems transparent and interpretable for end users. The framework detects paddy crop diseases with **91.3% accuracy** using modern deep learning while providing clear, actionable explanations for its decisions.

### Key Features

- **ConvNeXt V2-Tiny Backbone**: State-of-the-art CNN architecture (2023) with 28.6M parameters
- **Grad-CAM++ Visual Explanations**: Improved localization using second-order gradients
- **LLM-based Natural Language Generation**: Context-aware explanations using Phi-3-mini (3.8B)
- **Monte Carlo Dropout**: Uncertainty quantification with 30 stochastic passes
- **Audience-Adaptive Explanations**: Tailored for farmers, extension officers, and researchers
- **Actionable Recommendations**: Treatment and management suggestions based on disease severity

## Project Structure

```
ExplainableCropAI/
├── data/                           # Dataset and sample data
│   └── sample/                     # Sample CSV files and images
├── src/                            # Source code
│   ├── models/
│   │   └── convnext_v2.py          # ConvNeXt V2-Tiny classifier
│   ├── explainers/
│   │   ├── gradcam.py              # Legacy Grad-CAM
│   │   └── gradcam_plus.py         # Grad-CAM++ and LayerCAM
│   ├── nlg/
│   │   ├── template_generator.py   # Legacy template-based NLG
│   │   └── llm_generator.py        # Phi-3-mini LLM generator
│   ├── uncertainty/                # Monte Carlo dropout
│   ├── recommendations/            # Treatment recommendation engine
│   ├── visualization/              # Figure generation scripts
│   └── explainable_system.py       # Main integration module
├── models/                         # Saved model checkpoints
├── notebooks/                      # Jupyter notebooks for exploration
├── paper/                          # Springer Nature LaTeX files
├── results/                        # Experimental results and figures
└── docs/                           # Documentation
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/santoshray02/ExplainableCropAI.git
cd ExplainableCropAI
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Quick Start

1. **Explore the system demo**:
   ```bash
   jupyter notebook notebooks/system_demo.ipynb
   ```

2. **Generate visualizations**:
   ```bash
   python src/visualization/generate_user_interface.py
   python src/visualization/generate_gradcam_examples.py
   ```

3. **Run explanation effectiveness analysis**:
   ```bash
   python src/visualization/visualize_explanation_effectiveness.py
   ```

## Key Components

### Disease Classification
- **ConvNeXt V2-Tiny**: Modern CNN backbone (2023) with depthwise convolutions, LayerNorm, and GELU activations
- **10 Disease Classes**: Bacterial Leaf Blight, Brown Spot, Leaf Blast, Bacterial Leaf Streak, Bacterial Sheath Rot, Healthy, Tungro, Hispa, Dead Heart, Downy Mildew
- **91.3% Accuracy**: On Paddy Doctor dataset (10,407 images)

### Visual Explanations
- **Grad-CAM++**: Second-order gradient weighting for improved localization
- **LayerCAM**: Alternative multi-layer visualization method
- **Attention Heatmaps**: Highlighting disease-relevant regions

### Natural Language Generation
- **Phi-3-mini (3.8B)**: Microsoft's compact LLM for context-aware explanations
- **Audience-Adaptive**: Tailored outputs for farmers, experts, and researchers
- **4-bit Quantization**: Optional mobile deployment optimization

### Uncertainty Quantification
- **Monte Carlo Dropout**: 30 stochastic passes for uncertainty estimation
- **Entropy & Mutual Information**: Decomposed uncertainty metrics
- **Confidence Visualization**: Clear communication of prediction reliability

### Recommendation Engine
- **Treatment Database**: Comprehensive disease management for all 10 classes
- **Severity-Based Suggestions**: Low/medium/high severity treatment protocols
- **Context-Aware Integration**: Combined with LLM for natural recommendations

## sn-format Paper

This project includes a complete research paper in Springer Nature format documenting the methodology, experiments, and results. The paper is located in the `paper/` directory.

To compile the paper:
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Results

The system achieves:
- **91.3% Classification Accuracy**: ConvNeXt V2-Tiny on Paddy Doctor dataset
- **User Study Validation**: 12 participants (extension officers, farmers, students)
  - Up to 32.4% improvement in diagnostic accuracy with explanations
  - Statistically significant trust improvement (p < 0.01)
- **Per-Class Performance**: Healthy 99%, Leaf Blast 96%, Brown Spot 94%, BLB 93%

Key visualizations and results are available in the `results/figures/` directory.


## Contributing

This is a research project developed as part of an M.Tech thesis. For questions or collaboration opportunities, please contact the project maintainer.

## License

This project is developed for academic research purposes. Please cite appropriately if using any components in your research.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{explainablecropai2026,
  title={ExplainableCropAI: Interpretable Deep Learning for Agricultural Disease Detection and Decision Support},
  author={Kumar, Santosh and Kumar, Sumit and Nandan, Durgesh},
  journal={[Journal/Conference]},
  year={2026},
  note={ConvNeXt V2-Tiny, Grad-CAM++, Phi-3-mini LLM}
}
```

## Contact

- **Author**: Santosh Kumar
- **Email**: santoshray02@gmail.com
- **Institution**: Symbiosis Institute of Technology
- **GitHub**: [@santoshray02](https://github.com/santoshray02)

## Acknowledgments

This research was conducted as part of an M.Tech program at Symbiosis Institute of Technology. The project represents an academic exploration of explainable AI techniques applied to agricultural disease detection.
