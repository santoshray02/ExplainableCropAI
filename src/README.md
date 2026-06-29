# Source Code Structure

This directory contains the implementation of the ExplainableCropAI system.

## Files and Modules

- `data_loader.py` - Functions for loading and preprocessing agricultural image data
- `base_model.py` - Implementation of the base disease detection model
- `explainers/` - Directory containing various explainability techniques
  - `gradcam.py` - Grad-CAM implementation for visual explanations
  - `lime_explainer.py` - LIME implementation for local feature importance
  - `shap_explainer.py` - SHAP implementation for global feature importance
- `nlg/` - Natural Language Generation for explanations
  - `template_generator.py` - Template-based explanation generation
  - `disease_templates.py` - Disease-specific explanation templates
- `uncertainty/` - Uncertainty quantification
  - `mc_dropout.py` - Monte Carlo dropout for uncertainty estimation
  - `ensemble.py` - Ensemble methods for prediction variance
- `recommendations/` - Recommendation engine
  - `treatment_db.py` - Database of treatments for different diseases
  - `recommendation_engine.py` - Logic for generating context-aware recommendations
- `evaluation/` - Evaluation metrics and tools
  - `explanation_metrics.py` - Metrics for evaluating explanation quality (insertion/deletion faithfulness, runtime)
- `visualization/` - Visualization tools for explanations
  - `heatmap_visualizer.py` - Tools for visualizing attention heatmaps
  - `dashboard.py` - Interactive dashboard for explanation exploration

## How to Run

Instructions for running the models and generating explanations will be added here.
