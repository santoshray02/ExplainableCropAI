# GPU-run checklist — replace provisional numbers with real results

Run on a CUDA machine (the project's GPU box). Each script's output maps to a
specific `% PROVISIONAL` block or synthetic figure in `paper/main.tex`. Tick items
as you replace them, then flip the matching status in `paper/REVISION_RESPONSE_TRACKER.md`
to 🟢 and re-run the consistency sweep (bottom).

Commands and output files: see `experiments/README.md`.

---

## 0. Prerequisites
- [ ] Dataset available (official Paddy Doctor, 10,407 imgs, 10 classes) at `$DATA`
- [ ] Trained **ConvNeXt V2-Tiny** checkpoint at `$CKPT` (needed for #7, #8, and the headline metrics)
- [ ] Expert symptom **masks** at `$MASKS` (needed for #7 IoU/point-game)
- [ ] `pip install -r requirements.txt`

---

## 1. Splits — `create_splits.py`  (reviewer #6)
- [ ] Run; produces `outputs/splits.csv`, `split_counts.csv`, `split_summary.txt`
- [ ] **Confirm/replace** Table `tab:split-counts` (`main.tex:312`, `% PROVISIONAL`) with `split_counts.csv` — my figures are train 7,286 / val 1,562 / test 1,559; rounding may shift ±1
- [ ] Report the **# near-duplicates folded** (from `split_summary.txt`) in §Experimental Design
- [ ] Commit `splits.csv` to the repo (reproducibility, #14)

## 2. Explanation comparison — `run_explanation_comparison.py`  (reviewer #7)
- [ ] Run with `--masks-dir $MASKS`; produces `explanation_comparison.csv`, `table_explanation_comparison.tex`, `explanation_per_image.csv`
- [ ] **Replace** Table `tab:visual-explanation` (`main.tex:400`, `% PROVISIONAL`, 6 methods + point-game + runtime) with `table_explanation_comparison.tex`
- [ ] Update in-text §Explainability Analysis numbers to match: **IoU 0.68**, insertion **0.72**, deletion **0.21**, point-game **81.4%**, disease-specific **0.74 / 0.58**
- [ ] Fill the real **Fleiss' κ** and **# annotated images** into the annotation-protocol sentence (§Experimental Design)
- [x] Success/failure figure **added** (`fig:cam-examples`) from the project's real field-image demo (2 confident + 1 uncertain/referral case). Optional: regenerate/extend with quantitative IoU per panel once the GPU run is available.

## 3. Calibration & uncertainty — `run_calibration.py`  (reviewer #8)
- [ ] Run; produces `calibration_metrics.csv`, `reliability_diagram.png`, `uncertainty_decomposition.csv`, `referral_policy.csv`
- [ ] Update **ECE (0.087)** and add the real **MCE** and **Brier** values in §Explainability Analysis / Discussion (currently only ECE is given)
- [ ] Report the fitted **temperature T** (before/after) in §Explainability Mechanism
- [ ] Confirm the **epistemic/aleatoric ≈ 60/40** split against `uncertainty_decomposition.csv`
- [ ] **Add the reliability-diagram figure** — none is currently in the paper. Copy `reliability_diagram.png`→`paper/figures/`, add `\includegraphics` + `\label{fig:reliability}`, reference in §Explainability Analysis
- [ ] Cross-check the referral thresholds in §LLM-NLG safety paragraph against `referral_policy.csv`

## 4. Baselines + ablation — `run_baselines_ablation.py`  (reviewer #11)
- [ ] Run `--mode baselines`; produces `baselines.csv` → **replace** Table `tab:baselines` (`main.tex:421`, `% PROVISIONAL`)
- [ ] Run `--mode ablation`; produces `ablation.csv` → **replace** Table `tab:ablation` (`main.tex:446`, `% PROVISIONAL`)
- [ ] Update the comparative prose (§Comparative Analysis) accuracies to match the real `baselines.csv`
- [ ] Update the ablation prose (per-component ΔAcc, ECE 0.142→0.087) to match `ablation.csv`

## 5. Headline classification metrics + confusion matrix  (from `$CKPT` evaluation)
- [ ] Evaluate the trained model on the test split → confirm/replace: **accuracy 91.3%**, precision 90.7 / recall 90.2 / F1 90.4, **5-fold CV 90.8 ± 1.1**, top-2 **96.8** / top-3 **98.4**, and the per-class precision/recall/F1
- [ ] **Regenerate `fig:confusion-matrix` from real predictions** (current `confusion_matrix.pdf` is synthetic/illustrative). Update `src/visualization/generate_confusion_matrix.py` to read real predictions, or export the matrix from the eval script
- [ ] BLEU **0.73** + Flesch–Kincaid readability: recompute against the expert reference explanations (NLG eval) if available

---

## 6. After all replacements — re-verify
- [ ] No `% PROVISIONAL` left: `grep -n PROVISIONAL paper/main.tex` → empty
- [ ] Numbers agree across Abstract / Results / Discussion / Conclusion (re-run the consistency check)
- [ ] `cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex` → **zero undefined references/citations**
- [ ] Cite/define cross-check: every `\cite` key defined, every entry cited
- [ ] Flip 🟡 → 🟢 in `REVISION_RESPONSE_TRACKER.md` for #7, #8, #11 (and #6 counts confirmed)
- [ ] Tag a release → archive on Zenodo → paste the **DOI** into the Code-availability statement (#14)
- [ ] Professional language edit (#13), then final proofread
