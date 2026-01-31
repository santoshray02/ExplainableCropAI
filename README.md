# ExplainableCropAI

**Interpretable Deep Learning for Agricultural Disease Detection and Decision Support**

[![GitHub Repository](https://img.shields.io/badge/GitHub-ExplainableCropAI-blue)](https://github.com/santoshray02/ExplainableCropAI)

## Overview

ExplainableCropAI is a cutting-edge research project that addresses a critical challenge in agricultural technology: making AI-powered disease detection systems transparent and interpretable for end users. This project develops a novel framework that not only detects paddy crop diseases with high accuracy but also provides clear, actionable explanations for its decisions.

### Key Features

- **Multi-level Explanation System**: Visual heatmaps, feature importance, and natural language descriptions
- **Domain-Specific Templates**: Agricultural knowledge integrated into explanation generation
- **Uncertainty-Aware Predictions**: Model confidence communication for risk-informed decisions
- **Actionable Recommendations**: Treatment and management suggestions paired with disease explanations
- **User-Centered Design**: Designed with practical agricultural applications in mind

## Project Structure

```
ExplainableCropAI/
├── data/                           # Dataset and sample data
│   └── sample/                     # Sample CSV files and images
├── src/                            # Source code
│   ├── data/                       # Data processing utilities
│   ├── evaluation/                 # Explanation metrics
│   ├── explainers/                 # Explainability modules (Grad-CAM, etc.)
│   ├── nlg/                        # Natural language generation
│   ├── recommendations/            # Treatment recommendation engine
│   ├── uncertainty/                # Monte Carlo dropout implementation
│   └── visualization/              # Figure generation scripts
├── models/                         # Saved model checkpoints
├── notebooks/                      # Jupyter notebooks for exploration
├── paper/                          # sn-format Paper LaTeX files
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

### Explainability Modules
- **Grad-CAM**: Visual attention heatmaps highlighting disease-relevant regions
- **Feature Importance**: Quantitative analysis of input features
- **Natural Language Generation**: Human-readable explanations using domain templates

### Uncertainty Quantification
- **Monte Carlo Dropout**: Uncertainty estimation for model predictions
- **Confidence Visualization**: Clear communication of prediction reliability

### Recommendation Engine
- **Treatment Database**: Comprehensive disease management strategies
- **Context-Aware Suggestions**: Recommendations based on disease type and severity

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
- **High Detection Accuracy**: Competitive performance with state-of-the-art models
- **Meaningful Explanations**: User study validation of explanation quality
- **Practical Utility**: Actionable insights for agricultural decision-making

Key visualizations and results are available in the `results/figures/` directory.


## Contributing

This is a research project developed as part of an M.Tech thesis. For questions or collaboration opportunities, please contact the project maintainer.

## License

This project is developed for academic research purposes. Please cite appropriately if using any components in your research.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{explainablecropai2025,
  title={ExplainableCropAI: Interpretable Deep Learning for Agricultural Disease Detection and Decision Support},
  author={Santosh Kumar},
  journal={[Journal/Conference]},
  year={2025}
}
```

## Contact

- **Author**: Santosh Kumar
- **Email**: santoshray02@gmail.com
- **Institution**: Symbiosis Institute of Technology
- **GitHub**: [@santoshray02](https://github.com/santoshray02)

## Acknowledgments

This research was conducted as part of an M.Tech program at Symbiosis Institute of Technology. The project represents an academic exploration of explainable AI techniques applied to agricultural disease detection.
