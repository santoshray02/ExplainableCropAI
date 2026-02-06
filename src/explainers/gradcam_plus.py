#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Grad-CAM++ and LayerCAM implementations for ExplainableCropAI.

This module implements improved versions of Grad-CAM for better localization:
- Grad-CAM++: Uses second-order gradients for improved localization
- LayerCAM: Combines positive gradients with activations for cleaner heatmaps

Both methods are designed to work with ConvNeXt V2 architecture.

References:
    - Grad-CAM++: Chattopadhyay et al., "Grad-CAM++: Improved Visual Explanations
      for Deep Convolutional Networks" (WACV 2018)
    - LayerCAM: Jiang et al., "LayerCAM: Exploring Hierarchical Class Activation
      Maps for Localization" (IEEE TIP 2021)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from matplotlib import cm
from typing import Optional, Dict, Any, Union, Tuple


class GradCAMPlusPlus:
    """
    Grad-CAM++ (Gradient-weighted Class Activation Mapping++).

    An improved version of Grad-CAM that uses second-order gradients
    to provide better localization and handle multiple instances of
    the same class in an image.

    The key difference from Grad-CAM is the weight calculation:
    alpha_k = (d^2 y^c / dA_k^2) / (2*(d^2 y^c / dA_k^2) + sum(A_k * d^3 y^c / dA_k^3))

    This provides pixel-wise weighting instead of global average pooling,
    resulting in more focused and accurate heatmaps.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """
        Initialize GradCAMPlusPlus.

        Args:
            model: PyTorch model (supports ConvNeXt V2 and other CNN architectures)
            target_layer: Layer to use for Grad-CAM++. If None, attempts to
                         automatically detect the last convolutional layer.
        """
        self.model = model
        self.target_layer = target_layer

        # Set model to evaluation mode
        self.model.eval()

        # Placeholder for hooks and stored tensors
        self.hooks = []
        self.activations = None
        self.gradients = None

        # Auto-detect target layer for ConvNeXt V2 if not provided
        if self.target_layer is None:
            self.target_layer = self._find_target_layer()

    def _find_target_layer(self) -> Optional[nn.Module]:
        """
        Automatically find the target layer for CAM generation.

        Searches for ConvNeXt V2 specific layers first, then falls back
        to standard convolutional layers.

        Returns:
            The detected target layer or None if using model's stored activations.
        """
        # Check for ConvNeXt V2 architecture
        for name, module in self.model.named_modules():
            # ConvNeXt V2 uses LayerNorm after the last stage
            if 'stages' in name and isinstance(module, nn.Sequential):
                # Return the last block in the last stage
                children = list(module.children())
                if children:
                    return children[-1]

        # Check for standard ResNet/ConvNext backbone
        for name, module in self.model.named_modules():
            if 'layer4' in name or 'stage4' in name:
                if isinstance(module, nn.Sequential):
                    children = list(module.children())
                    if children:
                        return children[-1]

        # If no specific layer found, return None to use model's stored activations
        return None

    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""
        # Remove existing hooks if any
        self._remove_hooks()

        if self.target_layer is None:
            # No need for hooks if the model stores activations directly
            return

        def forward_hook(module, input, output):
            self.activations = output.detach().clone()
            # Enable gradient computation for activations
            self.activations.requires_grad_(True)

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach().clone()

        # Register the hooks
        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

        self.hooks = [forward_handle, backward_handle]

    def _remove_hooks(self):
        """Remove registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def _compute_weights(
        self,
        gradients: torch.Tensor,
        activations: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Compute Grad-CAM++ weights using second-order gradients.

        The weight formula is:
        alpha_k^c = (d^2 y^c / dA_k^2) /
                    (2 * (d^2 y^c / dA_k^2) + sum_ij(A_k_ij * d^3 y^c / dA_k^3))

        Args:
            gradients: First-order gradients (dY/dA)
            activations: Feature map activations
            eps: Small constant for numerical stability

        Returns:
            Tensor of weights with shape (batch, channels, 1, 1)
        """
        # First-order gradients: dy/dA
        grad_1 = gradients

        # Second-order gradients: d^2y/dA^2 (approximated as grad^2)
        grad_2 = grad_1.pow(2)

        # Third-order gradients: d^3y/dA^3 (approximated as grad^3)
        grad_3 = grad_1.pow(3)

        # Compute the sum of activations weighted by third-order gradients
        # Shape: (batch, channels, height, width)
        sum_activations_grad3 = (activations * grad_3).sum(dim=(2, 3), keepdim=True)

        # Compute alpha weights
        # alpha = grad_2 / (2 * grad_2 + sum(A * grad_3) + eps)
        denominator = 2 * grad_2 + sum_activations_grad3.expand_as(grad_2) + eps
        alpha = grad_2 / denominator

        # Apply ReLU to first-order gradients (positive influence only)
        # This ensures we only consider features that positively influence the class
        relu_grad = F.relu(grad_1)

        # Compute final weights by summing alpha * relu_grad over spatial dimensions
        # Shape: (batch, channels, 1, 1)
        weights = (alpha * relu_grad).sum(dim=(2, 3), keepdim=True)

        return weights

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        overlay_alpha: float = 0.5
    ) -> Dict[str, Any]:
        """
        Generate Grad-CAM++ visualization.

        Args:
            input_tensor: Input image tensor of shape (C, H, W) or (B, C, H, W)
            class_idx: Index of the class to generate CAM for.
                      If None, uses the predicted class.
            overlay_alpha: Transparency factor for the heatmap overlay.

        Returns:
            Dictionary containing:
                - class_idx: The target class index
                - cam: Raw class activation map
                - heatmap: Colorized heatmap
                - overlay: Overlay of heatmap on original image
                - original_img: Original image (denormalized)
        """
        # Register hooks
        self._register_hooks()

        # Ensure input has batch dimension
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # Clone input and enable gradients
        input_tensor = input_tensor.clone().requires_grad_(True)

        # Forward pass
        try:
            # Handle models that return tuple (logits, additional_outputs)
            output = self.model(input_tensor)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
        except Exception as e:
            self._remove_hooks()
            raise RuntimeError(f"Forward pass failed: {e}")

        # Determine class index
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()

        # Clear previous gradients
        self.model.zero_grad()

        # Create one-hot encoding for backward pass
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1

        # Backward pass
        logits.backward(gradient=one_hot, retain_graph=True)

        # Get activations and gradients
        if self.target_layer is not None:
            activations = self.activations
            gradients = self.gradients
        else:
            # Use model's stored activations and gradients
            activations = self.model.get_activations()
            gradients = self.model.get_activation_gradients()

        # Compute Grad-CAM++ weights
        weights = self._compute_weights(gradients, activations)

        # Generate weighted activation map
        # Shape: (batch, channels, height, width) * (batch, channels, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)

        # Apply ReLU to focus on positive contributions
        cam = F.relu(cam)

        # Normalize the CAM
        cam = cam.squeeze()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-10)
        cam = cam.cpu().detach().numpy()

        # Clean up hooks
        self._remove_hooks()

        # Create visualization
        heatmap, overlay, original_img = self._create_visualization(
            cam, input_tensor, overlay_alpha
        )

        return {
            'class_idx': class_idx,
            'cam': cam,
            'heatmap': heatmap,
            'overlay': overlay,
            'original_img': original_img,
            'method': 'Grad-CAM++'
        }

    def _create_visualization(
        self,
        cam: np.ndarray,
        input_tensor: torch.Tensor,
        overlay_alpha: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create visualization from CAM.

        Args:
            cam: Raw class activation map
            input_tensor: Original input tensor
            overlay_alpha: Transparency for overlay

        Returns:
            Tuple of (heatmap, overlay, original_img)
        """
        # Resize CAM to input size
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        heatmap = cv2.resize(cam, (w, h))
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap = cm.jet(heatmap)[:, :, :3]

        # Denormalize original image
        original_img = input_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        original_img = std * original_img + mean
        original_img = np.clip(original_img, 0, 1)

        # Create overlay
        overlay = (heatmap * overlay_alpha + original_img * (1 - overlay_alpha))
        overlay = (overlay * 255).astype(np.uint8)

        return heatmap, overlay, (original_img * 255).astype(np.uint8)


class LayerCAM:
    """
    LayerCAM: Layer-wise Class Activation Mapping.

    An alternative to Grad-CAM that combines positive gradients with
    activations at each spatial location, providing cleaner and more
    detailed heatmaps.

    Key differences from Grad-CAM:
    - Uses element-wise product of positive gradients and activations
    - No global average pooling of gradients
    - Better preservation of spatial details

    Particularly effective for ConvNeXt V2 due to its hierarchical
    feature extraction approach.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """
        Initialize LayerCAM.

        Args:
            model: PyTorch model (supports ConvNeXt V2 and other CNN architectures)
            target_layer: Layer to use for LayerCAM. If None, attempts to
                         automatically detect the last convolutional layer.
        """
        self.model = model
        self.target_layer = target_layer

        # Set model to evaluation mode
        self.model.eval()

        # Placeholder for hooks and stored tensors
        self.hooks = []
        self.activations = None
        self.gradients = None

        # Auto-detect target layer if not provided
        if self.target_layer is None:
            self.target_layer = self._find_target_layer()

    def _find_target_layer(self) -> Optional[nn.Module]:
        """
        Automatically find the target layer for CAM generation.

        Returns:
            The detected target layer or None if using model's stored activations.
        """
        # Check for ConvNeXt V2 architecture
        for name, module in self.model.named_modules():
            if 'stages' in name and isinstance(module, nn.Sequential):
                children = list(module.children())
                if children:
                    return children[-1]

        # Check for standard ResNet backbone
        for name, module in self.model.named_modules():
            if 'layer4' in name or 'stage4' in name:
                if isinstance(module, nn.Sequential):
                    children = list(module.children())
                    if children:
                        return children[-1]

        return None

    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""
        self._remove_hooks()

        if self.target_layer is None:
            return

        def forward_hook(module, input, output):
            self.activations = output.detach().clone()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach().clone()

        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

        self.hooks = [forward_handle, backward_handle]

    def _remove_hooks(self):
        """Remove registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def _compute_weights(
        self,
        gradients: torch.Tensor,
        activations: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute LayerCAM weights.

        LayerCAM uses element-wise product of positive gradients and activations,
        which preserves spatial information better than Grad-CAM.

        Args:
            gradients: Gradient tensor
            activations: Activation tensor

        Returns:
            Weighted activation map
        """
        # Apply ReLU to gradients to keep only positive gradients
        # This focuses on features that positively contribute to the class
        positive_gradients = F.relu(gradients)

        # Element-wise product of positive gradients and activations
        # This preserves spatial detail better than global average pooling
        weighted_activations = positive_gradients * activations

        return weighted_activations

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
        overlay_alpha: float = 0.5
    ) -> Dict[str, Any]:
        """
        Generate LayerCAM visualization.

        Args:
            input_tensor: Input image tensor of shape (C, H, W) or (B, C, H, W)
            class_idx: Index of the class to generate CAM for.
                      If None, uses the predicted class.
            overlay_alpha: Transparency factor for the heatmap overlay.

        Returns:
            Dictionary containing:
                - class_idx: The target class index
                - cam: Raw class activation map
                - heatmap: Colorized heatmap
                - overlay: Overlay of heatmap on original image
                - original_img: Original image (denormalized)
        """
        # Register hooks
        self._register_hooks()

        # Ensure input has batch dimension
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # Clone input and enable gradients
        input_tensor = input_tensor.clone().requires_grad_(True)

        # Forward pass
        try:
            output = self.model(input_tensor)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
        except Exception as e:
            self._remove_hooks()
            raise RuntimeError(f"Forward pass failed: {e}")

        # Determine class index
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()

        # Clear previous gradients
        self.model.zero_grad()

        # Create one-hot encoding for backward pass
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1

        # Backward pass
        logits.backward(gradient=one_hot, retain_graph=True)

        # Get activations and gradients
        if self.target_layer is not None:
            activations = self.activations
            gradients = self.gradients
        else:
            activations = self.model.get_activations()
            gradients = self.model.get_activation_gradients()

        # Compute LayerCAM weighted activations
        weighted_activations = self._compute_weights(gradients, activations)

        # Sum over channels
        cam = weighted_activations.sum(dim=1, keepdim=True)

        # Apply ReLU
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-10)
        cam = cam.cpu().detach().numpy()

        # Clean up hooks
        self._remove_hooks()

        # Create visualization
        heatmap, overlay, original_img = self._create_visualization(
            cam, input_tensor, overlay_alpha
        )

        return {
            'class_idx': class_idx,
            'cam': cam,
            'heatmap': heatmap,
            'overlay': overlay,
            'original_img': original_img,
            'method': 'LayerCAM'
        }

    def _create_visualization(
        self,
        cam: np.ndarray,
        input_tensor: torch.Tensor,
        overlay_alpha: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create visualization from CAM.

        Args:
            cam: Raw class activation map
            input_tensor: Original input tensor
            overlay_alpha: Transparency for overlay

        Returns:
            Tuple of (heatmap, overlay, original_img)
        """
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        heatmap = cv2.resize(cam, (w, h))
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap = cm.jet(heatmap)[:, :, :3]

        original_img = input_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        original_img = std * original_img + mean
        original_img = np.clip(original_img, 0, 1)

        overlay = (heatmap * overlay_alpha + original_img * (1 - overlay_alpha))
        overlay = (overlay * 255).astype(np.uint8)

        return heatmap, overlay, (original_img * 255).astype(np.uint8)


def visualize_gradcam_plus(
    explanation: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
    title_prefix: str = ""
) -> None:
    """
    Visualize Grad-CAM++ or LayerCAM explanation.

    Creates a three-panel visualization showing the original image,
    activation map, and overlay.

    Args:
        explanation: Explanation dictionary from GradCAMPlusPlus.generate()
                    or LayerCAM.generate()
        save_path: Path to save visualization. If None, displays inline.
        figsize: Figure size as (width, height)
        title_prefix: Prefix to add to subplot titles

    Returns:
        None
    """
    import matplotlib.pyplot as plt

    method = explanation.get('method', 'CAM')

    fig, ax = plt.subplots(1, 3, figsize=figsize)

    # Plot original image
    ax[0].imshow(explanation['original_img'])
    ax[0].set_title(f'{title_prefix}Original Image')
    ax[0].axis('off')

    # Plot heatmap
    ax[1].imshow(explanation['heatmap'])
    ax[1].set_title(f'{title_prefix}Activation Map ({method})')
    ax[1].axis('off')

    # Plot overlay
    ax[2].imshow(explanation['overlay'])
    ax[2].set_title(f'{title_prefix}{method} Overlay')
    ax[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compare_cam_methods(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_idx: Optional[int] = None,
    save_path: Optional[str] = None,
    target_layer: Optional[nn.Module] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compare different CAM methods on the same input.

    Generates visualizations for Grad-CAM++, LayerCAM, and creates
    a side-by-side comparison.

    Args:
        model: PyTorch model
        input_tensor: Input image tensor
        class_idx: Target class index (if None, uses predicted class)
        save_path: Path to save comparison visualization
        target_layer: Target layer for CAM computation

    Returns:
        Dictionary containing explanations from each method
    """
    import matplotlib.pyplot as plt

    # Generate explanations
    gradcam_pp = GradCAMPlusPlus(model, target_layer)
    layercam = LayerCAM(model, target_layer)

    explanations = {
        'gradcam_pp': gradcam_pp.generate(input_tensor, class_idx),
        'layercam': layercam.generate(input_tensor, class_idx)
    }

    # Create comparison visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, (name, exp) in enumerate(explanations.items()):
        method = exp['method']

        axes[idx, 0].imshow(exp['original_img'])
        axes[idx, 0].set_title(f'Original (Class: {exp["class_idx"]})')
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(exp['heatmap'])
        axes[idx, 1].set_title(f'{method} Heatmap')
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(exp['overlay'])
        axes[idx, 2].set_title(f'{method} Overlay')
        axes[idx, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return explanations


if __name__ == "__main__":
    # Example usage demonstrating both Grad-CAM++ and LayerCAM
    import sys
    import os

    # Add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from base_model import PaddyDiseaseClassifier

    print("Creating model...")
    model = PaddyDiseaseClassifier(num_classes=10)

    # Create dummy input
    print("Creating dummy input...")
    dummy_input = torch.randn(1, 3, 224, 224)

    # Test Grad-CAM++
    print("\nTesting Grad-CAM++...")
    gradcam_pp = GradCAMPlusPlus(model)
    explanation_pp = gradcam_pp.generate(dummy_input)
    print(f"  Class index: {explanation_pp['class_idx']}")
    print(f"  CAM shape: {explanation_pp['cam'].shape}")
    print(f"  Method: {explanation_pp['method']}")

    # Test LayerCAM
    print("\nTesting LayerCAM...")
    layercam = LayerCAM(model)
    explanation_lc = layercam.generate(dummy_input)
    print(f"  Class index: {explanation_lc['class_idx']}")
    print(f"  CAM shape: {explanation_lc['cam'].shape}")
    print(f"  Method: {explanation_lc['method']}")

    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'results')
    os.makedirs(results_dir, exist_ok=True)

    # Visualize Grad-CAM++
    print("\nSaving Grad-CAM++ visualization...")
    visualize_gradcam_plus(
        explanation_pp,
        save_path=os.path.join(results_dir, 'gradcam_plus_example.png')
    )

    # Visualize LayerCAM
    print("Saving LayerCAM visualization...")
    visualize_gradcam_plus(
        explanation_lc,
        save_path=os.path.join(results_dir, 'layercam_example.png')
    )

    # Compare methods
    print("\nComparing CAM methods...")
    compare_cam_methods(
        model,
        dummy_input,
        save_path=os.path.join(results_dir, 'cam_comparison.png')
    )

    print("\nDone! Visualizations saved to:", results_dir)
