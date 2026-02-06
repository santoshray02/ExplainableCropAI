#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Explainers module for ExplainableCropAI.

This module provides visual explanation methods for CNN predictions:
- GradCAM: Gradient-weighted Class Activation Mapping (baseline)
- GradCAMPlusPlus: Improved Grad-CAM with better localization
- LayerCAM: Layer-wise CAM with element-wise gradient weighting
"""

from .gradcam import GradCAM, visualize_explanation
from .gradcam_plus import GradCAMPlusPlus, LayerCAM, visualize_gradcam_plus, compare_cam_methods

__all__ = [
    'GradCAM',
    'GradCAMPlusPlus',
    'LayerCAM',
    'visualize_explanation',
    'visualize_gradcam_plus',
    'compare_cam_methods'
]
