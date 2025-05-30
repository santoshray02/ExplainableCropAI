#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Grad-CAM implementation for ExplainableCropAI.

This module implements Gradient-weighted Class Activation Mapping (Grad-CAM)
to visualize important regions in images for model predictions.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from matplotlib import cm


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    
    Generates visual explanations for CNN-based models by highlighting
    important regions in the image for prediction.
    """
    
    def __init__(self, model, target_layer=None):
        """
        Initialize GradCAM.
        
        Args:
            model: PyTorch model (should support activation and gradient access)
            target_layer: Layer to use for Grad-CAM (if None, uses the last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Placeholder for hooks
        self.hooks = []
        
    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""
        
        # Remove existing hooks if any
        self._remove_hooks()
        
        # If target layer is not provided, use model's activations directly
        # This assumes the model is designed to store activations (as in base_model.py)
        if self.target_layer is None:
            # No need for hooks if the model already stores activations and gradients
            return
        
        # Register hooks for specified target layer
        self.activations = None
        self.gradients = None
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Register the hooks
        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_backward_hook(backward_hook)
        
        # Save handles for later removal
        self.hooks = [forward_handle, backward_handle]
    
    def _remove_hooks(self):
        """Remove registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _get_cam(self, class_idx=None):
        """
        Generate a class activation map.
        
        Args:
            class_idx (int): Index of the class to generate CAM for.
                             If None, uses the predicted class.
                             
        Returns:
            np.ndarray: Class activation map.
        """
        # Get activations and gradients
        if self.target_layer is not None:
            activations = self.activations
            gradients = self.gradients
        else:
            # Use model's stored activations and gradients
            activations = self.model.get_activations()
            gradients = self.model.get_activation_gradients()
        
        # Take the mean of gradients over the batch dimension
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        # Weight the channels by corresponding gradients
        # (batch, channels, height, width)
        for i in range(activations.size(1)):
            activations[:, i, :, :] *= pooled_gradients[i]
        
        # Average the channels and apply ReLU
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        
        # Normalize the heatmap
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-10)
        
        return heatmap.cpu().numpy()
    
    def generate(self, input_tensor, class_idx=None, overlay_alpha=0.5):
        """
        Generate Grad-CAM visualization.
        
        Args:
            input_tensor (torch.Tensor): Input image tensor.
            class_idx (int): Index of the class to generate CAM for.
                             If None, uses the predicted class.
            overlay_alpha (float): Transparency factor for the heatmap overlay.
            
        Returns:
            dict: Visualization results including raw CAM, heatmap, and overlay.
        """
        # Register hooks
        self._register_hooks()
        
        # Ensure input has batch dimension
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Forward pass with gradients
        input_tensor.requires_grad = True
        
        # Get model prediction
        logits, outputs = self.model(input_tensor)
        
        # If class_idx is not provided, use the predicted class
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()
        
        # Clear previous gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1
        
        # Backward pass
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Generate class activation map
        cam = self._get_cam(class_idx)
        
        # Clean up
        self._remove_hooks()
        
        # Convert CAM to heatmap
        heatmap = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        heatmap = (heatmap * 255).astype(np.uint8)
        heatmap = cm.jet(heatmap)[:, :, :3]  # Apply colormap
        
        # Get original image (denormalize if needed)
        original_img = input_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        original_img = std * original_img + mean
        original_img = np.clip(original_img, 0, 1)
        
        # Create overlay
        overlay = (heatmap * overlay_alpha + original_img * (1 - overlay_alpha))
        overlay = (overlay * 255).astype(np.uint8)
        
        return {
            'class_idx': class_idx,
            'cam': cam,
            'heatmap': heatmap,
            'overlay': overlay,
            'original_img': (original_img * 255).astype(np.uint8)
        }


def visualize_explanation(explanation, save_path=None):
    """
    Visualize Grad-CAM explanation.
    
    Args:
        explanation (dict): Explanation dictionary from GradCAM.generate().
        save_path (str): Path to save visualization. If None, displays inline.
        
    Returns:
        None
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    ax[0].imshow(explanation['original_img'])
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    # Plot heatmap
    ax[1].imshow(explanation['heatmap'])
    ax[1].set_title('Activation Map')
    ax[1].axis('off')
    
    # Plot overlay
    ax[2].imshow(explanation['overlay'])
    ax[2].set_title('Grad-CAM Overlay')
    ax[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')  # Add parent directory to path
    
    from base_model import PaddyDiseaseClassifier
    
    # Create model
    model = PaddyDiseaseClassifier(num_classes=10)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Initialize GradCAM
    grad_cam = GradCAM(model)
    
    # Generate explanation
    explanation = grad_cam.generate(dummy_input)
    
    # Visualize
    visualize_explanation(explanation, save_path='../results/gradcam_example.png')
