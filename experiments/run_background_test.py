#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Background-dependence test (Round-2 point R2.9).

Does the classifier rely on the leaf, or on the background? For every test
image a foreground (leaf) mask is obtained by colour segmentation that is
independent of the classifier: HSV thresholding for green/yellow tissue,
morphological closing, largest connected component. Images whose mask covers
less than 5% or more than 95% of the frame are excluded as segmentation
failures and counted.

Conditions evaluated on the remaining images:
  original            unchanged
  bg_grey             background replaced by mid-grey
  bg_noise            background replaced by uniform random noise
  bg_swap             background replaced by a randomly chosen image of a
                      DIFFERENT class (the leaf is pasted onto it)
  fg_masked           leaf replaced by mid-grey, background kept

For each condition: accuracy, mean confidence, and the fraction of predictions
that agree with the prediction on the original image. On a random subset the
Grad-CAM++ energy fraction inside the leaf mask is also reported.

Usage (deepbeastx, SUPIR env):
  python experiments/run_background_test.py --checkpoint experiments/outputs/baseline_convnextv2_tiny.pth
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._common import (  # noqa: E402
    set_seed, get_device, build_model, load_checkpoint, PADDY_DISEASE_CLASSES)
from experiments.run_calibration import fit_temperature, collect_logits  # noqa: E402
from experiments._common import LogitsOnly, build_loader  # noqa: E402
from src.data_loader import get_transforms  # noqa: E402
from src.explainers.gradcam_plus import GradCAMPlusPlus  # noqa: E402

CONDITIONS = ['original', 'bg_grey', 'bg_noise', 'bg_swap', 'fg_masked']


def leaf_mask(rgb):
    """Colour-based leaf segmentation; returns uint8 mask in {0,1}."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((h >= 15) & (h <= 95) & (s >= 35) & (v >= 35)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    # keep components that are at least 20% of the largest (leaves can be several blades)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, a in enumerate(areas) if a >= 0.2 * areas.max()]
    return np.isin(lab, keep).astype(np.uint8)


def compose(rgb, mask, cond, rng, other=None):
    m = mask[..., None].astype(np.float32)
    if cond == 'original':
        return rgb
    if cond == 'bg_grey':
        bg = np.full_like(rgb, 128)
    elif cond == 'bg_noise':
        bg = rng.randint(0, 256, rgb.shape).astype(np.uint8)
    elif cond == 'bg_swap':
        bg = cv2.resize(other, (rgb.shape[1], rgb.shape[0]))
    elif cond == 'fg_masked':
        return (rgb * (1 - m) + 128 * m).astype(np.uint8)
    else:
        raise ValueError(cond)
    return (rgb * m + bg * (1 - m)).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--backbone', default='convnextv2_tiny')
    ap.add_argument('--splits-csv', default='experiments/outputs/splits.csv')
    ap.add_argument('--num-classes', type=int, default=10)
    ap.add_argument('--input-size', type=int, default=224)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--cam-images', type=int, default=300)
    ap.add_argument('--mask-dir', default=None,
                    help='directory of precomputed foreground masks named <image basename>.png '
                         '(e.g. U2-Net salient-object masks); if omitted, HSV colour segmentation is used')
    ap.add_argument('--tag', default='', help='suffix for output file names')
    ap.add_argument('--output-dir', default='experiments/outputs')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    set_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = get_device(prefer_gpu=not args.cpu)
    os.makedirs(args.output_dir, exist_ok=True)
    raw = load_checkpoint(build_model(args.backbone, args.num_classes, pretrained=False),
                          args.checkpoint, device)
    model = LogitsOnly(raw).to(device).eval()
    vl, vy = collect_logits(model, build_loader(args.splits_csv, 'val', args.batch_size), device)
    T = fit_temperature(vl, vy, device)
    tf = get_transforms('test', args.input_size)

    df = pd.read_csv(args.splits_csv)
    test = df[df['split'] == 'test'].reset_index(drop=True)
    paths = list(test['image_path']); labels = np.array([PADDY_DISEASE_CLASSES.index(l) for l in test['label']])

    # Load, resize to a common working size, and segment.
    W = 320
    imgs, masks, cover = [], [], []
    for p in paths:
        im = np.asarray(Image.open(p).convert('RGB'))
        im = cv2.resize(im, (W, int(W * im.shape[0] / im.shape[1])))
        if args.mask_dir:
            mp = os.path.join(args.mask_dir, os.path.basename(p) + '.png')
            m = (np.asarray(Image.open(mp).convert('L')) > 127).astype(np.uint8)
            m = cv2.resize(m, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            m = leaf_mask(im)
        imgs.append(im); masks.append(m); cover.append(float(m.mean()))
    cover = np.array(cover)
    valid = (cover >= 0.05) & (cover <= 0.95)
    print(f'{len(paths)} test images; mask coverage mean={cover.mean():.2f}; '
          f'excluded as segmentation failures: {(~valid).sum()}', flush=True)
    idx = np.where(valid)[0]

    @torch.no_grad()
    def predict(arrays):
        probs = []
        for i in range(0, len(arrays), args.batch_size):
            x = torch.stack([tf(Image.fromarray(a)) for a in arrays[i:i + args.batch_size]]).to(device)
            probs.append(F.softmax(model(x) / T, 1).cpu())
        return torch.cat(probs).numpy()

    results, base_pred = [], None
    for cond in CONDITIONS:
        arrays = []
        for i in idx:
            other = None
            if cond == 'bg_swap':
                cands = idx[labels[idx] != labels[i]]
                other = imgs[int(rng.choice(cands))]
            arrays.append(compose(imgs[i], masks[i], cond, rng, other))
        probs = predict(arrays)
        pred = probs.argmax(1); conf = probs.max(1); y = labels[idx]
        if cond == 'original':
            base_pred = pred
        row = {'condition': cond, 'n': len(idx), 'accuracy': round(float((pred == y).mean()) * 100, 2),
               'mean_confidence': round(float(conf.mean()), 3),
               'agreement_with_original': round(float((pred == base_pred).mean()) * 100, 2),
               'referral_rate_conf<0.9': round(float((conf < 0.9).mean()) * 100, 2)}
        results.append(row); print(row, flush=True)
    res = pd.DataFrame(results)
    res.to_csv(os.path.join(args.output_dir, f'background_test{args.tag}.csv'), index=False)
    pd.DataFrame([{'n_test': len(paths), 'mask_coverage_mean': round(float(cover.mean()), 3),
                   'excluded_lt5pct': int((cover < 0.05).sum()), 'excluded_gt95pct': int((cover > 0.95).sum()),
                   'n_analysed': int(valid.sum())}]).to_csv(
        os.path.join(args.output_dir, f'background_mask_stats{args.tag}.csv'), index=False)

    # Grad-CAM++ energy inside the leaf mask on a random subset of originals.
    sub = rng.choice(idx, size=min(args.cam_images, len(idx)), replace=False)
    cam_gen = GradCAMPlusPlus(raw)
    fracs = []
    for i in sub:
        x = tf(Image.fromarray(imgs[i])).unsqueeze(0).to(device)
        out = cam_gen.generate(x, None)
        cam = np.asarray(out['cam'], dtype=np.float32)
        cam = cam - cam.min(); cam = cam / (cam.max() + 1e-8)
        # bring the mask through the same resize + centre-crop as the test transform
        mh, mw = masks[i].shape
        short = int(args.input_size * 1.14)
        scale = short / min(mh, mw)
        m = cv2.resize(masks[i], (int(round(mw * scale)), int(round(mh * scale))), interpolation=cv2.INTER_NEAREST)
        top = (m.shape[0] - args.input_size) // 2; left = (m.shape[1] - args.input_size) // 2
        m = m[top:top + args.input_size, left:left + args.input_size]
        m = cv2.resize(m, (cam.shape[1], cam.shape[0]), interpolation=cv2.INTER_NEAREST)
        fracs.append(float((cam * m).sum() / (cam.sum() + 1e-8)))
    fracs = np.array(fracs)
    cam_row = {'n_images': len(sub), 'mean_cam_energy_in_leaf': round(float(fracs.mean()), 3),
               'median_cam_energy_in_leaf': round(float(np.median(fracs)), 3),
               'mean_leaf_area_fraction': round(float(cover[sub].mean()), 3)}
    pd.DataFrame([cam_row]).to_csv(os.path.join(args.output_dir, f'background_cam_energy{args.tag}.csv'), index=False)
    print(cam_row)
    print(res.to_string(index=False))


if __name__ == '__main__':
    main()
