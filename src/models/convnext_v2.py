#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ConvNeXt V2-Tiny model for paddy disease classification.

This module implements a ConvNeXt V2-Tiny classifier using pre-trained weights
from the timm library, with custom classification head and Grad-CAM++ compatibility.

Reference:
    Woo, S., et al. (2023). ConvNeXt V2: Co-designing and Scaling ConvNets with
    Masked Autoencoders. CVPR 2023.
"""

from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
import timm


# Paddy Doctor dataset disease classes
PADDY_DISEASE_CLASSES = [
    'bacterial_leaf_blight',
    'bacterial_leaf_streak',
    'bacterial_panicle_blight',
    'blast',
    'brown_spot',
    'dead_heart',
    'downy_mildew',
    'hispa',
    'normal',
    'tungro'
]


class ConvNeXtV2Classifier(nn.Module):
    """
    ConvNeXt V2-Tiny classifier for paddy disease detection.

    This model uses a pre-trained ConvNeXt V2-Tiny backbone with a custom
    classification head. It is designed to be compatible with Grad-CAM++
    and other explainability techniques.

    Attributes:
        num_classes (int): Number of disease classes (default: 10).
        backbone (nn.Module): ConvNeXt V2-Tiny feature extractor.
        classifier (nn.Module): Custom classification head.
        gradients (torch.Tensor): Stored gradients for Grad-CAM.
        activations (torch.Tensor): Stored activations for Grad-CAM.
    """

    # Model configuration
    MODEL_NAME = 'convnextv2_tiny.fcmae_ft_in22k_in1k'
    FEATURE_DIM = 768  # ConvNeXt V2-Tiny output dimension

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        freeze_backbone: bool = False
    ):
        """
        Initialize the ConvNeXtV2Classifier.

        Args:
            num_classes: Number of disease classes to predict.
            pretrained: Whether to use pre-trained weights from ImageNet-22k/1k.
            drop_rate: Dropout rate for the classifier head.
            drop_path_rate: Drop path (stochastic depth) rate.
            freeze_backbone: Whether to freeze the backbone weights.
        """
        super(ConvNeXtV2Classifier, self).__init__()

        self.num_classes = num_classes
        self.pretrained = pretrained

        # Load pre-trained ConvNeXt V2-Tiny backbone
        self.backbone = timm.create_model(
            self.MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate
        )

        # Get the actual feature dimension from the model
        self.feature_dim = self.backbone.num_features

        # Global Average Pooling (backup, backbone usually includes this)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Custom classification head with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate) if drop_rate > 0 else nn.Identity(),
            nn.Linear(self.feature_dim, num_classes)
        )

        # Storage for Grad-CAM compatibility
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Hook handles for cleanup
        self._forward_hook_handle = None
        self._backward_hook_handle = None

        # Register hooks for Grad-CAM
        self._register_hooks()

        # Optionally freeze backbone
        if freeze_backbone:
            self._freeze_backbone()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks for Grad-CAM compatibility."""
        target_layer = self.get_cam_target_layer()

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self._forward_hook_handle = target_layer.register_forward_hook(forward_hook)
        self._backward_hook_handle = target_layer.register_full_backward_hook(backward_hook)

    def _freeze_backbone(self) -> None:
        """Freeze backbone weights for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone weights for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_cam_target_layer(self) -> nn.Module:
        """
        Get the target layer for Grad-CAM++ visualization.

        Returns the last convolutional layer of the ConvNeXt V2 backbone,
        which is the last stage's last block.

        Returns:
            nn.Module: The target layer for CAM-based visualizations.
        """
        # ConvNeXt V2 structure: stages[3] is the last stage
        # Each stage contains blocks, we want the last block's conv
        return self.backbone.stages[-1]

    def get_features(
        self,
        x: torch.Tensor,
        return_intermediate: bool = False
    ) -> torch.Tensor:
        """
        Extract features from the backbone without classification.

        Args:
            x: Input image tensor of shape (B, 3, H, W).
            return_intermediate: If True, return features before global pooling.

        Returns:
            Feature tensor of shape (B, feature_dim) or (B, C, H, W) if intermediate.
        """
        # Get features from backbone (already pooled by default)
        features = self.backbone.forward_features(x)

        if return_intermediate:
            return features

        # Apply global pooling if features are spatial
        if features.dim() == 4:
            features = self.global_pool(features)
            features = features.flatten(1)

        return features

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through the network.

        Args:
            x: Input image tensor of shape (B, 3, H, W).
            return_features: Whether to return intermediate features.

        Returns:
            Tuple containing:
                - logits: Class logits of shape (B, num_classes).
                - additional_outputs: Dictionary with activations and features
                  for explainability.
        """
        # Extract features using backbone
        features = self.backbone.forward_features(x)

        # Handle spatial features
        if features.dim() == 4:
            pooled_features = self.global_pool(features)
            pooled_features = pooled_features.flatten(1)
        else:
            pooled_features = features

        # Classification
        logits = self.classifier(pooled_features)

        # Prepare additional outputs for explainability
        additional_outputs = {
            'activations': self.activations,
            'features': pooled_features,
            'gradients': self.gradients
        }

        if return_features:
            additional_outputs['spatial_features'] = features

        return logits, additional_outputs

    def get_activations(self) -> Optional[torch.Tensor]:
        """
        Get the activations from the target layer.

        Returns:
            Activation tensor or None if not available.
        """
        return self.activations

    def get_activation_gradients(self) -> Optional[torch.Tensor]:
        """
        Get the gradients of the target layer activations.

        Returns:
            Gradient tensor or None if not available.
        """
        return self.gradients

    def get_class_names(self) -> list:
        """
        Get the list of class names for the Paddy Doctor dataset.

        Returns:
            List of disease class names.
        """
        return PADDY_DISEASE_CLASSES.copy()

    def predict(
        self,
        x: torch.Tensor,
        return_probs: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions on input images.

        Args:
            x: Input image tensor of shape (B, 3, H, W).
            return_probs: If True, return probabilities; otherwise, return logits.

        Returns:
            Tuple of (predicted_classes, confidence_scores).
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)

            if return_probs:
                probs = torch.softmax(logits, dim=1)
                confidence, predicted = torch.max(probs, dim=1)
            else:
                confidence, predicted = torch.max(logits, dim=1)

        return predicted, confidence

    def cleanup_hooks(self) -> None:
        """Remove registered hooks to prevent memory leaks."""
        if self._forward_hook_handle is not None:
            self._forward_hook_handle.remove()
            self._forward_hook_handle = None

        if self._backward_hook_handle is not None:
            self._backward_hook_handle.remove()
            self._backward_hook_handle = None

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.cleanup_hooks()


def create_model(config: Optional[Dict[str, Any]] = None) -> ConvNeXtV2Classifier:
    """
    Factory function to create a ConvNeXtV2Classifier instance.

    Args:
        config: Optional dictionary with model configuration parameters.
            Supported keys:
                - num_classes (int): Number of output classes (default: 10).
                - pretrained (bool): Use pre-trained weights (default: True).
                - drop_rate (float): Dropout rate (default: 0.0).
                - drop_path_rate (float): Drop path rate (default: 0.0).
                - freeze_backbone (bool): Freeze backbone weights (default: False).

    Returns:
        Instantiated ConvNeXtV2Classifier model.

    Example:
        >>> config = {'num_classes': 10, 'pretrained': True, 'drop_rate': 0.2}
        >>> model = create_model(config)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> logits, outputs = model(x)
    """
    if config is None:
        config = {}

    model = ConvNeXtV2Classifier(
        num_classes=config.get('num_classes', 10),
        pretrained=config.get('pretrained', True),
        drop_rate=config.get('drop_rate', 0.0),
        drop_path_rate=config.get('drop_path_rate', 0.0),
        freeze_backbone=config.get('freeze_backbone', False)
    )

    return model


def load_pretrained(
    checkpoint_path: str,
    config: Optional[Dict[str, Any]] = None,
    map_location: str = 'cpu'
) -> ConvNeXtV2Classifier:
    """
    Load a pre-trained ConvNeXtV2Classifier from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        config: Optional model configuration (used if not in checkpoint).
        map_location: Device to load the model to.

    Returns:
        Loaded ConvNeXtV2Classifier model.
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # Get config from checkpoint or use provided config
    model_config = checkpoint.get('config', config) or {}

    # Create model
    model = create_model(model_config)

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    return model


if __name__ == "__main__":
    """Quick test to verify model functionality."""
    import sys

    print("=" * 60)
    print("ConvNeXt V2-Tiny Classifier Test")
    print("=" * 60)

    # Test with pretrained=False to avoid downloading weights during testing
    print("\n[1] Creating model (pretrained=False for quick test)...")
    try:
        model = create_model({
            'num_classes': 10,
            'pretrained': False,
            'drop_rate': 0.1
        })
        print(f"    Model created successfully!")
        print(f"    Feature dimension: {model.feature_dim}")
        print(f"    Number of classes: {model.num_classes}")
    except Exception as e:
        print(f"    Error creating model: {e}")
        sys.exit(1)

    # Test forward pass
    print("\n[2] Testing forward pass...")
    x = torch.randn(2, 3, 224, 224)
    try:
        logits, outputs = model(x)
        print(f"    Input shape: {x.shape}")
        print(f"    Output logits shape: {logits.shape}")
        print(f"    Features shape: {outputs['features'].shape}")
        if outputs['activations'] is not None:
            print(f"    Activations shape: {outputs['activations'].shape}")
    except Exception as e:
        print(f"    Error in forward pass: {e}")
        sys.exit(1)

    # Test feature extraction
    print("\n[3] Testing feature extraction...")
    try:
        features = model.get_features(x)
        print(f"    Pooled features shape: {features.shape}")

        spatial_features = model.get_features(x, return_intermediate=True)
        print(f"    Spatial features shape: {spatial_features.shape}")
    except Exception as e:
        print(f"    Error in feature extraction: {e}")
        sys.exit(1)

    # Test Grad-CAM compatibility
    print("\n[4] Testing Grad-CAM compatibility...")
    try:
        target_layer = model.get_cam_target_layer()
        print(f"    CAM target layer: {type(target_layer).__name__}")

        # Test gradient flow
        model.train()
        x.requires_grad = True
        logits, outputs = model(x)
        loss = logits[:, 0].sum()
        loss.backward()

        gradients = model.get_activation_gradients()
        activations = model.get_activations()

        if gradients is not None:
            print(f"    Gradients shape: {gradients.shape}")
        if activations is not None:
            print(f"    Activations shape: {activations.shape}")
    except Exception as e:
        print(f"    Error in Grad-CAM test: {e}")
        sys.exit(1)

    # Test prediction
    print("\n[5] Testing prediction...")
    try:
        model.eval()
        x_test = torch.randn(4, 3, 224, 224)
        predicted, confidence = model.predict(x_test)
        print(f"    Predicted classes: {predicted.tolist()}")
        print(f"    Confidence scores: {[f'{c:.4f}' for c in confidence.tolist()]}")
    except Exception as e:
        print(f"    Error in prediction: {e}")
        sys.exit(1)

    # Display class names
    print("\n[6] Disease classes (Paddy Doctor dataset):")
    for i, name in enumerate(model.get_class_names()):
        print(f"    {i}: {name}")

    # Cleanup
    model.cleanup_hooks()

    print("\n" + "=" * 60)
    print("All tests passed successfully!")
    print("=" * 60)
