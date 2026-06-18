# ExplainableCropAI — manuscript (The Visual Computer, Springer Nature)

LaTeX sources for the paper *"ExplainableCropAI: Interpretable Deep Learning for
Agricultural Disease Detection and Decision Support"*, submitted to **The Visual
Computer** (Springer Nature, `sn-jnl.cls`). **Status: major revision in progress.**

## Contents

| File | Purpose |
|------|---------|
| `main.tex` | Main manuscript (single-file, SN `sn-mathphys-num` style) |
| `references.bib` | Bibliography |
| `sn-jnl.cls`, `bst/` | Springer Nature class + bibliography styles |
| `figures/` | All figures (PDF + PNG) — referenced from `main.tex` as `figures/...` |
| `Revisionfeedback.md` | Reviewer/editor feedback (points #6–#16 + 3 required questions) |
| `RESPONSE_TO_REVIEWERS.md` | Point-by-point response letter for resubmission |
| `REVISION_RESPONSE_TRACKER.md` | Per-point action tracker / status |
| `PUBLICATION_CHECKLIST.md` | Pre-submission checklist |

## Compiling

```bash
cd paper
pdflatex main.tex     # 1st pass — generates .aux
bibtex main           # process bibliography (needs bst/ + .bst in dir)
pdflatex main.tex     # 2nd pass — include bibliography
pdflatex main.tex     # 3rd pass — resolve cross-references
```

Confirm **zero undefined references** before submitting (grep the `.log` for
`undefined` / `??`). Overleaf: upload the whole `paper/` folder including `bst/` and
`figures/`. VS Code: LaTeX Workshop "Build LaTeX project".

## Reference style

`sn-mathphys-num` (Math & Physical Sciences, numbered). To change, edit the document
class option in `main.tex` (e.g. `sn-basic`, `sn-vancouver-num`, `sn-mathphys-ay`).

## Revision notes (current round)

- Backbone/results updated to **ConvNeXt V2-Tiny, 91.3%** (ResNet50/85.2% is a baseline).
- Dataset table corrected to the **official Paddy Doctor distribution**; class
  **Bacterial Panicle Blight** (not "Bacterial Sheath Rot").
- New tables (6-method explanation comparison, baselines, ablation, per-split counts)
  carry **provisional numbers** marked `% PROVISIONAL` — replace with the outputs of
  `../experiments/*.py` before submission.
- Reproducibility, calibration, and uncertainty additions per reviewer points #6–#15.

## Contact

Corresponding author: **Sumit Kumar** — `er.sumitkumar21@gmail.com` —
Symbiosis Institute of Technology, Symbiosis International (Deemed University), Pune.
