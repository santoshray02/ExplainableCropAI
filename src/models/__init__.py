#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Models module for ExplainableCropAI.

This module contains various model architectures for paddy disease classification.
"""

from .convnext_v2 import ConvNeXtV2Classifier, create_model

__all__ = ['ConvNeXtV2Classifier', 'create_model']
