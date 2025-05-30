#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base disease detection model for ExplainableCropAI.

This module implements the core CNN architecture for paddy disease detection,
designed to be compatible with various explainability techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PaddyDiseaseClassifier(nn.Module):
    """
    CNN-based classifier for paddy disease detection.
    
    This model uses a pre-trained ResNet50 backbone with a custom classification
    head, modified to preserve activation maps for explainability techniques.
    """
    
    def __init__(self, num_classes=10, pretrained=True):
        """
        Initialize the PaddyDiseaseClassifier.
        
        Args:
            num_classes (int): Number of disease classes to predict.
            pretrained (bool): Whether to use pre-trained weights.
        """
        super(PaddyDiseaseClassifier, self).__init__()
        
        # Load pre-trained ResNet50 without the classification head
        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.fc = nn.Linear(2048, num_classes)
        
        # Activation maps for explainability
        self.gradients = None
        self.activations = None
        
    def activations_hook(self, grad):
        """Store gradients for Grad-CAM."""
        self.gradients = grad
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input image batch.
            
        Returns:
            torch.Tensor: Class logits.
            dict: Additional outputs for explainability.
        """
        # Extract features
        x = self.backbone(x)
        
        # Save activations for explainability
        self.activations = x
        
        # Register hook for Grad-CAM
        if self.training or x.requires_grad:
            h = x.register_hook(self.activations_hook)
        
        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        logits = self.fc(x)
        
        # Additional outputs for explainability
        additional_outputs = {
            'activations': self.activations,
            'features': x
        }
        
        return logits, additional_outputs
    
    def get_activations(self):
        """Get the activations from the last convolutional layer."""
        return self.activations
    
    def get_activation_gradients(self):
        """Get the gradients of the activations."""
        return self.gradients


def create_model(config):
    """
    Factory function to create a model instance based on configuration.
    
    Args:
        config (dict): Model configuration parameters.
        
    Returns:
        PaddyDiseaseClassifier: Instantiated model.
    """
    model = PaddyDiseaseClassifier(
        num_classes=config.get('num_classes', 10),
        pretrained=config.get('pretrained', True)
    )
    return model


if __name__ == "__main__":
    # Quick test to ensure model can be instantiated
    model = PaddyDiseaseClassifier()
    x = torch.randn(2, 3, 224, 224)
    logits, outputs = model(x)
    print(f"Output shape: {logits.shape}")
    print(f"Activation shape: {outputs['activations'].shape}")
