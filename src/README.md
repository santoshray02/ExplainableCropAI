# Source code structure

- `models/convnext_v2.py` — `ConvNeXtV2Classifier`, the ConvNeXt V2-Tiny backbone with a 10-way head, a dropout layer before the classifier (kept active at inference for Monte Carlo Dropout) and hooks on the last stage for Grad-CAM++. `create_model(config)` builds it.
- `base_model.py` — `PaddyDiseaseClassifier`, the original ResNet50 model, retained only as a baseline.
- `data_loader.py` — `PaddyDiseaseDataset` and `get_transforms(mode, input_size)`; train mode applies random-resized-crop, flip, rotation and colour jitter, test mode applies resize and centre crop. Inputs are normalised to ImageNet statistics; no other preprocessing is applied.
- `explainers/gradcam_plus.py` — `GradCAMPlusPlus` and `LayerCAM`; `explainers/gradcam.py` — standard Grad-CAM.
- `uncertainty/mc_dropout.py` — `MCDropout` (30 stochastic passes by default) with predictive-entropy / mutual-information decomposition, and `uncertainty_to_confidence`.
- `nlg/llm_generator.py` — Phi-3-mini explanation generator with farmer and extension-officer prompt modes; `nlg/template_generator.py` — deterministic fallback that reproduces the same knowledge-base content.
- `recommendations/treatment_db.py` — curated knowledge base (treatment, prevention, severity levels per disease); `recommendations/recommendation_engine.py` — selects knowledge-base content by disease and severity.
- `evaluation/explanation_metrics.py` — insertion / deletion faithfulness metrics and runtime.
- `explainable_system.py` — `ExplainableCropAI`, the unified pipeline: classification, Grad-CAM++, MC Dropout, reduction of the attribution map to three typed descriptors (`_cam_descriptors`: dominant region, coverage, multiplicity), language generation and recommendation.
- `visualization/` — scripts that render the figures used in the paper from the outputs in `experiments/outputs/`.

Entry points for the experiments are in `../experiments/`.
