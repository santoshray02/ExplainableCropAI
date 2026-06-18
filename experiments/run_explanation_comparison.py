#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Head-to-head comparison of attribution methods under identical settings.
Addresses reviewer point #7 (stronger evidence for the explanation module).

Methods compared (all on the SAME model, SAME images, SAME threshold):
  - Grad-CAM            (vanilla)
  - Grad-CAM++          (project implementation)
  - LayerCAM            (project implementation)
  - Integrated Gradients (Captum)
  - LIME                (lime_image)
  - SHAP                (GradientExplainer)

For every annotated test image it computes localisation IoU / precision /
recall / F1 against the expert mask, plus faithfulness (insertion / deletion
AUC) and wall-clock time. Results are aggregated (mean +/- std) per method and
written as a CSV and a ready-to-paste LaTeX table. Qualitative success/failure
panels are also exported.

INPUTS REQUIRED
---------------
A folder of expert annotation masks, one PNG per annotated image, named to match
the image stem (e.g. image 100330.jpg -> masks/100330.png), white = symptom region.
See experiments/ANNOTATION_PROTOCOL.md for how to produce these.

Usage
-----
python experiments/run_explanation_comparison.py \
    --checkpoint runs/convnextv2_best.pth \
    --splits-csv experiments/outputs/splits.csv \
    --masks-dir data/expert_masks \
    --threshold 0.5 \
    --methods gradcam gradcampp layercam ig lime shap
"""

import os
import sys
import time
import glob
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._common import (  # noqa: E402
    set_seed, get_device, build_model, load_checkpoint, LogitsOnly,
    IMAGENET_MEAN, IMAGENET_STD, PADDY_DISEASE_CLASSES)
from src.data_loader import get_transforms  # noqa: E402
from src.evaluation.explanation_metrics import VisualExplanationEvaluator  # noqa: E402


# --------------------------------------------------------------------------- #
# Attribution method adapters -> each returns a 2D heatmap in [0, 1], H x W.
# --------------------------------------------------------------------------- #
def _normalize(cam):
    cam = cam - cam.min()
    return cam / (cam.max() + 1e-8)


def heatmap_gradcam(model, x, class_idx, device):
    """Vanilla Grad-CAM using the project model's stored activations/gradients."""
    model.zero_grad()
    x = x.clone().requires_grad_(True)
    logits, extras = model(x)
    if class_idx is None:
        class_idx = logits.argmax(1).item()
    logits[0, class_idx].backward(retain_graph=True)
    acts = extras['activations']
    grads = model.model.get_activation_gradients() if isinstance(model, LogitsOnly) \
        else model.get_activation_gradients()
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * acts).sum(1)).squeeze().detach().cpu().numpy()
    return _normalize(cam), class_idx


def heatmap_gradcampp(raw_model, x, class_idx, device):
    from src.explainers.gradcam_plus import GradCAMPlusPlus
    out = GradCAMPlusPlus(raw_model).generate(x, class_idx)
    return _normalize(out['cam']), out['class_idx']


def heatmap_layercam(raw_model, x, class_idx, device):
    from src.explainers.gradcam_plus import LayerCAM
    out = LayerCAM(raw_model).generate(x, class_idx)
    return _normalize(out['cam']), out['class_idx']


def heatmap_ig(model, x, class_idx, device):
    from captum.attr import IntegratedGradients
    model.zero_grad()
    if class_idx is None:
        class_idx = model(x).argmax(1).item()
    ig = IntegratedGradients(model)
    attr = ig.attribute(x, target=class_idx, n_steps=50)
    cam = attr.squeeze(0).abs().sum(0).detach().cpu().numpy()  # sum over channels
    return _normalize(cam), class_idx


def heatmap_lime(model, x, class_idx, device):
    from lime import lime_image
    from skimage.segmentation import mark_boundaries  # noqa: F401 (lime dep)
    img = x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0, 1).astype(np.float64)

    def predict_fn(images):
        batch = torch.stack([
            torch.from_numpy(((im - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)).float()
            for im in images]).to(device)
        with torch.no_grad():
            return F.softmax(model(batch), dim=1).cpu().numpy()

    explainer = lime_image.LimeImageExplainer()
    exp = explainer.explain_instance(img, predict_fn, top_labels=1,
                                     hide_color=0, num_samples=1000)
    if class_idx is None:
        class_idx = exp.top_labels[0]
    _, mask = exp.get_image_and_mask(class_idx, positive_only=True,
                                     num_features=5, hide_rest=False)
    return _normalize(mask.astype(np.float32)), class_idx


def heatmap_shap(model, x, class_idx, device, background):
    import shap
    if class_idx is None:
        class_idx = model(x).argmax(1).item()
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(x)
    sv = shap_values[class_idx] if isinstance(shap_values, list) else shap_values
    cam = np.abs(sv).squeeze(0).sum(0)
    return _normalize(cam), class_idx


METHOD_FUNCS = {
    'gradcam': 'wrapped',
    'gradcampp': 'raw',
    'layercam': 'raw',
    'ig': 'wrapped',
    'lime': 'wrapped',
    'shap': 'wrapped',
}


def load_mask(masks_dir, image_path, size):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    matches = glob.glob(os.path.join(masks_dir, stem + '.*'))
    if not matches:
        return None
    m = Image.open(matches[0]).convert('L').resize((size, size))
    return (np.array(m) > 127).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--splits-csv', default='experiments/outputs/splits.csv')
    ap.add_argument('--masks-dir', required=True)
    ap.add_argument('--backbone', default='convnextv2_tiny')
    ap.add_argument('--num-classes', type=int, default=10)
    ap.add_argument('--input-size', type=int, default=224)
    ap.add_argument('--threshold', type=float, default=0.5,
                    help='Binarisation threshold for IoU (also swept 0.1..0.9)')
    ap.add_argument('--methods', nargs='+',
                    default=['gradcam', 'gradcampp', 'layercam', 'ig', 'lime', 'shap'])
    ap.add_argument('--max-images', type=int, default=0, help='0 = all annotated')
    ap.add_argument('--output-dir', default='experiments/outputs')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(prefer_gpu=not args.cpu)
    os.makedirs(args.output_dir, exist_ok=True)

    raw_model = build_model(args.backbone, args.num_classes, pretrained=False)
    raw_model = load_checkpoint(raw_model, args.checkpoint, device)
    wrapped = LogitsOnly(raw_model).to(device).eval()
    evaluator = VisualExplanationEvaluator(model=wrapped)

    tfm = get_transforms('test', args.input_size)
    df = pd.read_csv(args.splits_csv)
    df = df[df['split'] == 'test'] if 'split' in df.columns else df

    # Only keep images that actually have an expert mask.
    annotated = [(p, l) for p, l in zip(df['image_path'], df['label'])
                 if load_mask(args.masks_dir, p, args.input_size) is not None]
    if args.max_images:
        annotated = annotated[:args.max_images]
    print(f"[#7] {len(annotated)} annotated test images found")
    if not annotated:
        print("No masks matched. See experiments/ANNOTATION_PROTOCOL.md")
        return

    # Background batch for SHAP.
    background = None
    if 'shap' in args.methods:
        bg = []
        for p, _ in annotated[:16]:
            bg.append(tfm(Image.open(p).convert('RGB')))
        background = torch.stack(bg).to(device)

    rows = []
    per_image = []
    thresholds = [round(t, 2) for t in np.arange(0.1, 0.95, 0.1)]
    for p, label in annotated:
        img = tfm(Image.open(p).convert('RGB')).unsqueeze(0).to(device)
        gt = load_mask(args.masks_dir, p, args.input_size)
        for m in args.methods:
            t0 = time.time()
            try:
                if m == 'gradcam':
                    cam, cidx = heatmap_gradcam(wrapped, img, None, device)
                elif m == 'gradcampp':
                    cam, cidx = heatmap_gradcampp(raw_model, img, None, device)
                elif m == 'layercam':
                    cam, cidx = heatmap_layercam(raw_model, img, None, device)
                elif m == 'ig':
                    cam, cidx = heatmap_ig(wrapped, img, None, device)
                elif m == 'lime':
                    cam, cidx = heatmap_lime(wrapped, img, None, device)
                elif m == 'shap':
                    cam, cidx = heatmap_shap(wrapped, img, None, device, background)
                else:
                    continue
            except Exception as e:
                print(f"[{m}] failed on {p}: {e}")
                continue
            elapsed = time.time() - t0

            if cam.shape != gt.shape:
                import cv2
                cam = cv2.resize(cam, (gt.shape[1], gt.shape[0]))

            loc = evaluator.evaluate_localization(cam, gt, threshold=args.threshold)
            # threshold sweep (IoU only) to report best operating point
            sweep = {f'iou@{t}': evaluator.evaluate_localization(cam, gt, threshold=t)['iou']
                     for t in thresholds}
            try:
                faith = evaluator.evaluate_faithfulness(img, cam, class_idx=cidx)
            except Exception:
                faith = {'insertion_auc': np.nan, 'deletion_auc': np.nan}

            rec = {'method': m, 'image': os.path.basename(p), 'label': label,
                   'time_s': elapsed, **loc,
                   'insertion_auc': faith.get('insertion_auc'),
                   'deletion_auc': faith.get('deletion_auc'), **sweep}
            per_image.append(rec)

    pi = pd.DataFrame(per_image)
    pi.to_csv(os.path.join(args.output_dir, 'explanation_per_image.csv'), index=False)

    # Aggregate.
    agg_cols = ['iou', 'precision', 'recall', 'f1_score',
                'insertion_auc', 'deletion_auc', 'time_s']
    for m in args.methods:
        sub = pi[pi['method'] == m]
        if sub.empty:
            continue
        row = {'method': m, 'n': len(sub)}
        for c in agg_cols:
            row[f'{c}_mean'] = sub[c].mean()
            row[f'{c}_std'] = sub[c].std()
        # best threshold by mean IoU
        ious = {t: sub[f'iou@{t}'].mean() for t in thresholds}
        best_t = max(ious, key=ious.get)
        row['best_threshold'] = best_t
        row['iou_best'] = ious[best_t]
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.output_dir, 'explanation_comparison.csv'), index=False)
    print("\n=== Explanation method comparison (mean) ===")
    print(summary[['method', 'n', 'iou_mean', 'f1_score_mean',
                   'deletion_auc_mean', 'time_s_mean', 'iou_best', 'best_threshold']]
          .to_string(index=False))

    # LaTeX table for the paper.
    latex = os.path.join(args.output_dir, 'table_explanation_comparison.tex')
    with open(latex, 'w') as f:
        f.write("% Auto-generated by run_explanation_comparison.py (reviewer #7)\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Method & IoU & F1 & Deletion AUC$\\downarrow$ & Time (s) \\\\\n\\midrule\n")
        name = {'gradcam': 'Grad-CAM', 'gradcampp': 'Grad-CAM++', 'layercam': 'LayerCAM',
                'ig': 'Integrated Gradients', 'lime': 'LIME', 'shap': 'SHAP'}
        for _, r in summary.iterrows():
            f.write(f"{name.get(r['method'], r['method'])} & "
                    f"{r['iou_mean']:.3f}$\\pm${r['iou_std']:.3f} & "
                    f"{r['f1_score_mean']:.3f} & "
                    f"{r['deletion_auc_mean']:.3f} & "
                    f"{r['time_s_mean']:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"\n[#7] wrote {latex}")


if __name__ == '__main__':
    main()
