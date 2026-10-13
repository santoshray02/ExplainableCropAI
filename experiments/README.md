# Experiments — reproducing every number in the paper

All scripts share `--seed 42` and write to `experiments/outputs/` (default). The
committed files in that directory are the exact outputs behind the manuscript's
tables; re-running a script overwrites them. A CUDA GPU is assumed; every script
accepts `--cpu` for smoke tests.

## Setup

```bash
pip install -r requirements.txt            # torch, timm, scikit-learn, pandas, imagehash, Pillow, opencv-python
export DATA=/path/to/paddy-doctor/train_images   # one sub-folder per class (Kaggle release)
# trained checkpoint: GitHub release v1.1 -> experiments/outputs/baseline_convnextv2_tiny.pth
```

Paths in the committed `splits.csv` are relative to the dataset root; run the
scripts from the repository root with the dataset at `train_images/`, or
regenerate the split with your own path (step 1).

## Scripts, in the order the paper uses them

| Step | Script | Produces | Paper |
|---|---|---|---|
| 1 | `create_splits.py --data-dir $DATA` | `splits.csv`, `split_counts.csv`, `split_summary.txt` — stratified 70/15/15 image-level split with perceptual-hash near-duplicate groups kept together | §2.5, Table 2 |
| 2 | `run_baselines_ablation.py --mode baselines` / `--mode ablation` | `baselines.csv`, `ablation.csv`, `baseline_<backbone>.pth` — seven backbones trained under one protocol | Tables 6–7 |
| 3 | `run_eval.py --checkpoint outputs/baseline_convnextv2_tiny.pth` | `headline_metrics.json`, `per_class_metrics.csv`, `confusion_matrix*.csv`, `predictions.csv` | §3.1, Table 3, Fig. 4 |
| 4 | `run_confidence_intervals.py outputs/predictions.csv` | `confidence_intervals.txt` — bootstrap 95% CIs, confidence bands, referral curve | §3.1–3.2 |
| 5 | `run_calibration.py --checkpoint ...` | `calibration_metrics.csv`, `reliability_diagram.png`, `uncertainty_decomposition.csv`, `referral_policy.csv` | §3.2, Fig. 6 |
| 6 | `run_uncertainty_error.py --checkpoint ...` | `uncertainty_error.csv`, `referral_by_signal.csv`, `calibration_ci.csv`, `mc_predictions.csv` — AUROC of each uncertainty signal for identifying errors, CIs for ECE / Brier / referral | §3.2, Table 4 |
| 7 | `run_explanation_faithfulness.py --checkpoint ...` | `explanation_comparison.csv`, `explanation_per_image.csv` — insertion / deletion / runtime for six attribution methods on 30 sampled test images | Table 5 |
| 8 | `run_baseline_eval.py` | `baselines_extended.csv`, `mcnemar.csv`, `predictions_<backbone>.csv` — per-model temperature scaling, ECE, Brier, AUROC, exact McNemar tests vs. ConvNeXt V2-Tiny | Table 6 |
| 9 | `run_robustness.py --backbones convnextv2_tiny resnet50 ...` | `robustness.csv`, `robustness_summary.csv` — 8 corruptions × 5 severities for every model | §3.5, Tables 10–11 |
| 10 | `run_background_test.py --checkpoint ...` | `background_test_hsv.csv`, `background_cam_energy_hsv.csv` — background replacement / leaf removal on the colour-segmentable subset | §3.6, Table 12 |
| 11 | `run_external_eval.py` (see `run_external.sh`) | `external_<dataset>_metrics.json`, confusion matrices, predictions — zero-shot evaluation on RiceLeafBD and Sethy et al. | §3.4, Table 8 |
| 12 | `run_external_finetune.py` (see `run_finetune.sh`, `run_finetune2.sh`) | `external_<dataset>_finetune.csv` — few-shot adaptation from the Paddy Doctor model and from ImageNet weights; Sethy et al. split group-wise by base image | §3.4, Table 9 |

`run_round2_batch.sh` runs steps 6, 8, 9 and 10 in sequence. `make_confusion_figure.py`
renders Fig. 4 from `confusion_matrix_pct.csv`.

## External datasets (step 11–12)

Download from Mendeley Data (both CC BY 4.0) and point `--data-dir` at the class folders:

- RiceLeafBD — <https://doi.org/10.17632/kx9rx8p2mz.1> (`Original Images/`)
- Rice Leaf Disease Image Samples, Sethy et al. — <https://doi.org/10.17632/fwcj7stb8r.2> (`Rice Leaf Disease Images/`)

The class-name mappings to Paddy Doctor labels are given in the two shell runners.
Note that the Sethy et al. files are augmented copies of about 900 base photographs
(`BLAST3_045.jpg`, `brownspot_orig_045.jpg`, … share a base image); the
`--group-regex '_([0-9]+).'` option keeps all copies of one base image on one side of the
hold-out. `external_sethy_finetune.csv` is the result *without* that grouping and is kept
only to document the leakage it causes (100% accuracy); the paper reports
`external_sethy_grouped_finetune.csv`.

## Notes

- `run_explanation_comparison.py` is the older harness that also accepted expert symptom
  masks; no such masks exist for this work and the paper reports only the model-based
  metrics from `run_explanation_faithfulness.py`.
- Logs (`*.log`) and checkpoints (`*.pth`) are not committed.
