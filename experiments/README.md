# Experiments — reproducing the reviewer-requested results

These scripts regenerate the numbers in the manuscript. Run them on any
CUDA-capable GPU machine, then copy the emitted values/tables into the paper.

**Run order:** `create_splits.py` first (everything else consumes `splits.csv`),
then the other three in any order.

All scripts share: `--seed 42`, `--output-dir experiments/outputs` (default), and a
`--cpu` flag for smoke-testing without a GPU.

---

## 0. Setup

```bash
cd ExplainableCropAI
pip install -r requirements.txt          # torch, timm, scikit-learn, pandas, imagehash, Pillow, matplotlib
export DATA=/data/paddy-doctor/train_images   # one sub-folder per class
export CKPT=/path/to/convnextv2_tiny_best.pth # trained classifier checkpoint
```

---

## 1. `create_splits.py` — Reviewer #6 (data splitting & leakage)

```bash
python experiments/create_splits.py \
    --data-dir "$DATA" \
    --train 0.70 --val 0.15 --test 0.15 --seed 42 --phash-threshold 5
```

| Output | Goes into paper |
|---|---|
| `outputs/splits.csv` | consumed by all other scripts (and training) |
| `outputs/split_counts.csv` | **per-class × per-split count table** (the new #6 table) |
| `outputs/split_summary.txt` | dedup stats + split level → §Experimental Design prose |

After running, paste `split_counts.csv` as the per-split table and confirm the
train/val/test totals match the manuscript.

---

## 2. `run_explanation_comparison.py` — Reviewer #7 (attribution comparison)

Needs expert symptom masks (`--masks-dir`) aligned to the annotated test subset.

```bash
python experiments/run_explanation_comparison.py \
    --checkpoint "$CKPT" \
    --splits-csv experiments/outputs/splits.csv \
    --masks-dir /data/paddy-doctor/expert_masks \
    --backbone convnextv2_tiny --num-classes 10 \
    --threshold 0.5 \
    --methods gradcam gradcampp layercam ig lime shap
```

| Output | Goes into paper |
|---|---|
| `outputs/explanation_comparison.csv` | **Table `tab:visual-explanation`** (IoU / insertion / deletion / point-game / runtime) |
| `outputs/table_explanation_comparison.tex` | drop-in LaTeX table — replaces the provisional 6-method table |
| `outputs/explanation_per_image.csv` | per-image scores → success/failure figure selection |

---

## 3. `run_calibration.py` — Reviewer #8 (uncertainty & calibration)

```bash
python experiments/run_calibration.py \
    --checkpoint "$CKPT" \
    --splits-csv experiments/outputs/splits.csv \
    --backbone convnextv2_tiny --num-classes 10 \
    --mc-samples 30 --n-bins 15
```

| Output | Goes into paper |
|---|---|
| `outputs/calibration_metrics.csv` | **ECE / MCE / Brier** (before & after temperature scaling) → §Results |
| `outputs/reliability_diagram.png` | new **reliability-diagram figure** |
| `outputs/uncertainty_decomposition.csv` | epistemic vs aleatoric split (≈60/40) |
| `outputs/referral_policy.csv` | uncertainty → {accept / reduce-confidence / refer} thresholds (#8/#9 decision policy) |

---

## 4. `run_baselines_ablation.py` — Reviewer #11 (baselines + ablation)

Two modes; run both.

```bash
# Baselines (Table tab:baselines)
python experiments/run_baselines_ablation.py --mode baselines \
    --splits-csv experiments/outputs/splits.csv --epochs 40 \
    --backbones convnextv2_tiny efficientformer_l1 inception_v3 \
                resnet50 mobilevit_s efficientnet_b0 vgg16

# Ablation (Table tab:ablation)
python experiments/run_baselines_ablation.py --mode ablation \
    --splits-csv experiments/outputs/splits.csv --epochs 40
```

| Output | Goes into paper |
|---|---|
| `outputs/baselines.csv` | **Table `tab:baselines`** (params / accuracy / macro-F1 / latency) |
| `outputs/ablation.csv` | **Table `tab:ablation`** (per-component ΔAcc) |
| `outputs/baseline_*.pth` | trained baseline checkpoints |

---

## After all runs

1. Replace every `% PROVISIONAL` table/number in `paper/main.tex` with the emitted values.
2. Re-check internal consistency (IoU, ECE, BLEU, accuracies) across Abstract / Results / Discussion / Conclusion.
3. Update `paper/REVISION_RESPONSE_TRACKER.md` statuses 🟡 → 🟢.
4. Tag a release and archive to Zenodo → paste the DOI into the Code-availability statement.
