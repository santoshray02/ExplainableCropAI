#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Monte Carlo Dropout for uncertainty estimation in ExplainableCropAI.

This module implements Monte Carlo Dropout for uncertainty quantification
in deep learning models for paddy disease classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter


class MCDropout:
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    Uses multiple forward passes with dropout enabled to estimate
    prediction uncertainty.
    """
    
    def __init__(self, model, num_samples=30, dropout_rate=0.5):
        """
        Initialize MC Dropout.
        
        Args:
            model: PyTorch model with dropout layers
            num_samples (int): Number of forward passes
            dropout_rate (float): Dropout rate (if model needs modification)
        """
        self.model = model
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate
        
        # Enable dropout during inference
        self._enable_dropout(model)
    
    def _enable_dropout(self, model):
        """
        Enable dropout layers during inference.
        
        Args:
            model: PyTorch model
        """
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Set dropout layers to training mode
    
    def predict(self, input_tensor):
        """
        Perform multiple forward passes to estimate uncertainty.
        
        Args:
            input_tensor (torch.Tensor): Input image tensor
            
        Returns:
            dict: Prediction results with uncertainty metrics
        """
        # Store predictions from multiple forward passes
        predictions = []
        
        # Ensure input has batch dimension
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Multiple forward passes
        for _ in range(self.num_samples):
            with torch.no_grad():
                logits, _ = self.model(input_tensor)
                probabilities = F.softmax(logits, dim=1)
                predictions.append(probabilities.cpu().numpy())
        
        # Stack predictions
        predictions = np.stack(predictions)  # Shape: (num_samples, batch_size, num_classes)
        
        # Calculate mean prediction (expected probabilities)
        mean_probabilities = np.mean(predictions, axis=0)  # Shape: (batch_size, num_classes)
        
        # Get predicted class (highest mean probability)
        predicted_class = np.argmax(mean_probabilities, axis=1)  # Shape: (batch_size,)
        
        # Calculate uncertainty metrics
        uncertainty_metrics = self._calculate_uncertainty(predictions, mean_probabilities)
        
        # Count predicted classes across samples for the first image
        pred_classes = np.argmax(predictions[:, 0, :], axis=1)  # Shape: (num_samples,)
        class_counts = Counter(pred_classes)
        
        # Calculate frequency of the most predicted class
        most_common_class, most_common_count = class_counts.most_common(1)[0]
        class_agreement = most_common_count / self.num_samples
        
        return {
            'predicted_class': predicted_class[0],
            'mean_probabilities': mean_probabilities[0],
            'entropy': uncertainty_metrics['entropy'][0],
            'predictive_variance': uncertainty_metrics['predictive_variance'][0],
            'mutual_information': uncertainty_metrics['mutual_information'][0],
            'class_agreement': class_agreement,
            'class_distribution': dict(class_counts),
        }
    
    def _calculate_uncertainty(self, predictions, mean_probabilities):
        """
        Calculate uncertainty metrics from predictions.
        
        Args:
            predictions (np.ndarray): Predictions from multiple forward passes
                                     Shape: (num_samples, batch_size, num_classes)
            mean_probabilities (np.ndarray): Mean probabilities
                                           Shape: (batch_size, num_classes)
                                           
        Returns:
            dict: Uncertainty metrics
        """
        # Predictive entropy (total uncertainty)
        # H[p(y|x)] = -∑_c p(y=c|x) log p(y=c|x)
        entropy = -np.sum(mean_probabilities * np.log(mean_probabilities + 1e-10), axis=1)
        
        # Expected entropy (aleatoric uncertainty)
        # E_p(w|D)[H[p(y|x,w)]]
        expected_entropy = -np.mean(
            np.sum(predictions * np.log(predictions + 1e-10), axis=2),
            axis=0
        )
        
        # Mutual information (epistemic uncertainty)
        # I[y,w|x,D] = H[p(y|x)] - E_p(w|D)[H[p(y|x,w)]]
        mutual_information = entropy - expected_entropy
        
        # Predictive variance
        predictive_variance = np.mean(np.var(predictions, axis=0), axis=1)
        
        return {
            'entropy': entropy,
            'expected_entropy': expected_entropy,
            'mutual_information': mutual_information,
            'predictive_variance': predictive_variance
        }


def uncertainty_to_confidence(uncertainty_info, method='mutual_information'):
    """
    Convert uncertainty metrics to a confidence score.
    
    Args:
        uncertainty_info (dict): Uncertainty metrics from MCDropout.predict()
        method (str): Method to use for confidence calculation
                     Options: 'mutual_information', 'entropy', 'class_agreement'
                     
    Returns:
        float: Confidence score (0-1)
    """
    if method == 'mutual_information':
        # Normalize mutual information (lower is more confident)
        # Assuming mutual_information is in range [0, log(num_classes)]
        # For 10 classes, log(10) ≈ 2.3
        max_mi = 2.3  # Approximate log of number of classes
        confidence = 1.0 - min(uncertainty_info['mutual_information'] / max_mi, 1.0)
    
    elif method == 'entropy':
        # Normalize entropy (lower is more confident)
        # Assuming entropy is in range [0, log(num_classes)]
        max_entropy = 2.3  # Approximate log of number of classes
        confidence = 1.0 - min(uncertainty_info['entropy'] / max_entropy, 1.0)
    
    elif method == 'class_agreement':
        # Class agreement (higher is more confident)
        confidence = uncertainty_info['class_agreement']
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return confidence


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')  # Add parent directory to path
    
    from base_model import PaddyDiseaseClassifier
    
    # Create model
    model = PaddyDiseaseClassifier(num_classes=10)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Initialize MC Dropout
    mc_dropout = MCDropout(model, num_samples=20)
    
    # Predict with uncertainty
    uncertainty_info = mc_dropout.predict(dummy_input)
    
    # Convert to confidence score
    confidence = uncertainty_to_confidence(uncertainty_info)
    
    print(f"Predicted class: {uncertainty_info['predicted_class']}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Entropy: {uncertainty_info['entropy']:.4f}")
    print(f"Mutual information: {uncertainty_info['mutual_information']:.4f}")
    print(f"Class agreement: {uncertainty_info['class_agreement']:.4f}")
    print(f"Class distribution: {uncertainty_info['class_distribution']}")
