#!/bin/bash
# Round-2 experiment batch: E1 baseline re-evaluation, E5 uncertainty-vs-error,
# E4 background test, E2 robustness (primary model first, then baselines).
cd ~/projects/ExplainableCropAI
P=~/miniconda3/envs/SUPIR/bin/python
CK=experiments/outputs/baseline_convnextv2_tiny.pth
echo "### E1 baseline eval $(date)"
$P experiments/run_baseline_eval.py
echo "### E5 uncertainty-error $(date)"
$P experiments/run_uncertainty_error.py --checkpoint $CK
echo "### E4 background test $(date)"
$P experiments/run_background_test.py --checkpoint $CK
echo "### E2 robustness $(date)"
$P experiments/run_robustness.py --backbones convnextv2_tiny resnet50 vgg16 efficientformer_l1 mobilevit_s efficientnet_b0 inception_v3
echo "ALLDONE $(date)"
