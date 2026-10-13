#!/bin/bash
# Second batch: group-wise hold-out for Sethy et al. (each base image + its augmented copies share the numeric suffix, e.g. BLAST3_045 ~ brownspot_orig_045) + ImageNet-init controls.
cd ~/projects/ExplainableCropAI
P=~/miniconda3/envs/SUPIR/bin/python
CK=experiments/outputs/baseline_convnextv2_tiny.pth
$P experiments/run_external_finetune.py --checkpoint $CK \
  --data-dir "/home/santosh/data/external/sethy/Rice Leaf Disease Images" \
  --class-map '{"Bacterialblight":"bacterial_leaf_blight","Blast":"blast","Brownspot":"brown_spot","Tungro":"tungro"}' \
  --name sethy_grouped --group-regex '_([0-9]+).' \
  --shots 0 5 10 25 50 all --epochs 15 --compare-imagenet --imagenet-shots 5 10 25 50 all
$P experiments/run_external_finetune.py --checkpoint $CK \
  --data-dir "/home/santosh/data/external/riceleafbd/Original Images" \
  --class-map '{"Bacterial Leaf Blight":"bacterial_leaf_blight","Brown Spot":"brown_spot","Healthy Leaf":"normal","Tungro Virus":"tungro"}' \
  --name riceleafbd_imnet --shots 0 --epochs 15 --compare-imagenet --imagenet-shots 5 10 25 50
echo ALLDONE
