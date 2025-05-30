#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main integration module for ExplainableCropAI.

This module integrates all components of the ExplainableCropAI system,
providing a unified interface for paddy disease detection with explanations.
"""

import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Import components
from base_model import PaddyDiseaseClassifier
from explainers.gradcam import GradCAM, visualize_explanation
from nlg.template_generator import NLExplanationGenerator, ExplanationWithVisualEvidence
from uncertainty.mc_dropout import MCDropout, uncertainty_to_confidence
from recommendations.recommendation_engine import RecommendationEngine
from recommendations.treatment_db import PaddyDiseaseKnowledgeBase


class ExplainableCropAI:
    """
    Integrated system for explainable paddy disease detection.
    
    Combines disease detection, visual explanations, uncertainty estimation,
    natural language explanations, and recommendations.
    """
    
    def __init__(self, model_path=None, num_classes=10, device=None):
        """
        Initialize the ExplainableCropAI system.
        
        Args:
            model_path (str): Path to trained model weights
            num_classes (int): Number of disease classes
            device (torch.device): Device to run model on
        """
        # Set device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = PaddyDiseaseClassifier(num_classes=num_classes)
        
        # Load model weights if provided
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Class names (placeholder, should be loaded from config)
        self.class_names = [
            "Bacterial Leaf Blight",
            "Brown Spot",
            "Leaf Blast",
            "Bacterial Leaf Streak", 
            "Sheath Blight",
            "Healthy",
            "Unknown Disease 1",
            "Unknown Disease 2",
            "Unknown Disease 3",
            "Unknown Disease 4"
        ]
        
        # Initialize components
        self.visual_explainer = GradCAM(self.model)
        self.uncertainty_estimator = MCDropout(self.model, num_samples=30)
        self.nlg_generator = ExplanationWithVisualEvidence(
            NLExplanationGenerator()
        )
        self.recommendation_engine = RecommendationEngine(
            knowledge_base=PaddyDiseaseKnowledgeBase()
        )
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess_image(self, image_path):
        """
        Preprocess an image for model input.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            tuple: (tensor for model input, PIL image for display)
        """
        # Load image
        pil_image = Image.open(image_path).convert('RGB')
        
        # Transform for model
        input_tensor = self.transform(pil_image)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        
        return input_tensor, pil_image
    
    def predict(self, image_path, with_explanation=True):
        """
        Predict disease and generate explanations.
        
        Args:
            image_path (str): Path to image file
            with_explanation (bool): Whether to generate explanations
            
        Returns:
            dict: Prediction results and explanations
        """
        # Preprocess image
        input_tensor, original_image = self.preprocess_image(image_path)
        
        # Base prediction
        with torch.no_grad():
            logits, outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            
            # Get top prediction
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_class = predicted_idx.item()
            confidence_value = confidence.item()
        
        # Create prediction info
        prediction_info = {
            "class_name": self.class_names[predicted_class],
            "class_id": predicted_class,
            "confidence": confidence_value,
            "probabilities": probabilities[0].cpu().numpy()
        }
        
        # Return just prediction if explanation not requested
        if not with_explanation:
            return prediction_info
        
        # Generate visual explanation (Grad-CAM)
        visual_explanation = self.visual_explainer.generate(input_tensor, class_idx=predicted_class)
        
        # Estimate uncertainty
        uncertainty_info = self.uncertainty_estimator.predict(input_tensor)
        adjusted_confidence = uncertainty_to_confidence(uncertainty_info)
        
        # Update prediction with uncertainty-adjusted confidence
        prediction_info["raw_confidence"] = prediction_info["confidence"]
        prediction_info["confidence"] = adjusted_confidence
        
        # Detect visual features from explanation
        visual_features = self._extract_visual_features(visual_explanation, prediction_info["class_name"])
        
        # Generate natural language explanation
        disease_info = self.recommendation_engine.kb.get_disease_info(prediction_info["class_name"])
        nl_explanation = self.nlg_generator.generate_explanation(
            prediction_info=prediction_info,
            visual_features=visual_features,
            disease_info=disease_info
        )
        
        # Generate recommendations
        metadata = {
            "affected_area_percentage": self._estimate_affected_area(visual_explanation)
        }
        recommendations = self.recommendation_engine.generate_recommendations(
            prediction=prediction_info,
            uncertainty=uncertainty_info,
            metadata=metadata
        )
        
        # Combine all results
        result = {
            "prediction": prediction_info,
            "uncertainty": uncertainty_info,
            "visual_explanation": visual_explanation,
            "nl_explanation": nl_explanation,
            "recommendations": recommendations,
            "visual_features": visual_features
        }
        
        return result
    
    def _extract_visual_features(self, visual_explanation, disease_name):
        """
        Extract visual features from explanation based on disease.
        
        Args:
            visual_explanation (dict): Visual explanation from Grad-CAM
            disease_name (str): Predicted disease name
            
        Returns:
            list: Detected visual features
        """
        # This is a placeholder implementation
        # In a real system, this would analyze the activation maps
        # and map them to domain-specific visual features
        
        # Simple mapping based on disease name
        feature_mapping = {
            "Bacterial Leaf Blight": ["leaf_lesions", "leaf_discoloration"],
            "Brown Spot": ["leaf_spots", "leaf_discoloration"],
            "Leaf Blast": ["leaf_lesions", "leaf_discoloration"],
            "Bacterial Leaf Streak": ["leaf_lesions"],
            "Sheath Blight": ["stem_lesions", "leaf_discoloration"],
            "Healthy": []
        }
        
        # Return mapped features or empty list
        return feature_mapping.get(disease_name, ["leaf_discoloration"])
    
    def _estimate_affected_area(self, visual_explanation):
        """
        Estimate percentage of affected area from visual explanation.
        
        Args:
            visual_explanation (dict): Visual explanation from Grad-CAM
            
        Returns:
            float: Estimated percentage of affected area (0-100)
        """
        # Get raw CAM and apply threshold
        cam = visual_explanation['cam']
        threshold = 0.5
        binary_mask = (cam > threshold).astype(np.float32)
        
        # Calculate percentage of affected area
        affected_percentage = np.mean(binary_mask) * 100
        
        return affected_percentage
    
    def visualize_results(self, result, output_path=None):
        """
        Visualize prediction results with explanations.
        
        Args:
            result (dict): Prediction results from predict()
            output_path (str): Path to save visualization
            
        Returns:
            None
        """
        # Extract data from result
        prediction = result["prediction"]
        visual_explanation = result["visual_explanation"]
        nl_explanation = result["nl_explanation"]
        recommendations = result["recommendations"]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 10))
        
        # 1. Original image
        ax1 = plt.subplot2grid((3, 6), (0, 0), colspan=2, rowspan=1)
        ax1.imshow(visual_explanation["original_img"])
        ax1.set_title("Original Image")
        ax1.axis("off")
        
        # 2. Grad-CAM visualization
        ax2 = plt.subplot2grid((3, 6), (0, 2), colspan=2, rowspan=1)
        ax2.imshow(visual_explanation["overlay"])
        ax2.set_title("Visual Explanation (Grad-CAM)")
        ax2.axis("off")
        
        # 3. Uncertainty visualization
        ax3 = plt.subplot2grid((3, 6), (0, 4), colspan=2, rowspan=1)
        self._plot_uncertainty(ax3, result["uncertainty"])
        
        # 4. Prediction and confidence
        ax4 = plt.subplot2grid((3, 6), (1, 0), colspan=3, rowspan=1)
        self._plot_prediction(ax4, prediction)
        
        # 5. Visual features and evidence
        ax5 = plt.subplot2grid((3, 6), (1, 3), colspan=3, rowspan=1)
        self._plot_visual_features(ax5, result["visual_features"], nl_explanation["visual_evidence"])
        
        # 6. Natural language explanation
        ax6 = plt.subplot2grid((3, 6), (2, 0), colspan=3, rowspan=1)
        self._plot_explanation(ax6, nl_explanation)
        
        # 7. Recommendations
        ax7 = plt.subplot2grid((3, 6), (2, 3), colspan=3, rowspan=1)
        self._plot_recommendations(ax7, recommendations)
        
        # Add title
        plt.suptitle(f"ExplainableCropAI: {prediction['class_name']} Detection", fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save or show
        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            plt.close()
        else:
            plt.show()
    
    def _plot_uncertainty(self, ax, uncertainty):
        """Plot uncertainty information."""
        ax.axis("off")
        ax.set_title("Prediction Uncertainty")
        
        # Create text content
        content = [
            f"Entropy: {uncertainty['entropy']:.4f}",
            f"Mutual Information: {uncertainty['mutual_information']:.4f}",
            f"Class Agreement: {uncertainty['class_agreement']:.2f}",
            "\nClass Distribution:"
        ]
        
        # Add class distribution
        for cls, count in uncertainty["class_distribution"].items():
            if cls < len(self.class_names):
                content.append(f"- {self.class_names[cls]}: {count}")
        
        # Add text to plot
        ax.text(0.5, 0.5, "\n".join(content), ha="center", va="center", fontsize=9)
    
    def _plot_prediction(self, ax, prediction):
        """Plot prediction information."""
        ax.axis("off")
        ax.set_title("Disease Prediction")
        
        # Create content
        content = [
            f"Predicted Disease: {prediction['class_name']}",
            f"Confidence: {prediction['confidence']:.2f}",
            "\nTop Predictions:"
        ]
        
        # Add top 3 predictions
        probs = prediction["probabilities"]
        top_indices = np.argsort(probs)[::-1][:3]
        
        for idx in top_indices:
            content.append(f"- {self.class_names[idx]}: {probs[idx]:.2f}")
        
        # Add text to plot
        ax.text(0.5, 0.5, "\n".join(content), ha="center", va="center", fontsize=10)
    
    def _plot_visual_features(self, ax, features, evidence):
        """Plot visual features."""
        ax.axis("off")
        ax.set_title("Visual Evidence")
        
        # Create content
        content = [
            "Detected Visual Features:",
            ", ".join(f"'{feature}'" for feature in features),
            "\nVisual Evidence:",
            evidence
        ]
        
        # Add text to plot
        ax.text(0.5, 0.5, "\n".join(content), ha="center", va="center", fontsize=10, wrap=True)
    
    def _plot_explanation(self, ax, explanation):
        """Plot natural language explanation."""
        ax.axis("off")
        ax.set_title("Explanation")
        
        # Get full explanation
        content = explanation["full_explanation"]
        
        # Add text to plot
        ax.text(0.5, 0.5, content, ha="center", va="center", fontsize=9, wrap=True)
    
    def _plot_recommendations(self, ax, recommendations):
        """Plot recommendations."""
        ax.axis("off")
        ax.set_title("Recommendations")
        
        # Create content
        content = [
            f"Disease: {recommendations['disease']}",
            f"Severity: {recommendations['severity']}",
            f"Risk Level: {recommendations['risk_level']}",
            "\nTreatment:",
            recommendations['recommendations']['treatment'],
            "\nPrevention:",
            recommendations['recommendations']['prevention']
        ]
        
        # Add verification note if present
        if recommendations['recommendations']['verification']:
            content.append("\nNote:")
            content.append(recommendations['recommendations']['verification'])
        
        # Add text to plot
        ax.text(0.5, 0.5, "\n".join(content), ha="center", va="center", fontsize=9, wrap=True)


if __name__ == "__main__":
    # Example usage
    system = ExplainableCropAI()
    
    # Sample image path (this should be an actual path to a test image)
    sample_image_path = "../data/sample_image.jpg"
    
    # Check if the file exists (for this example)
    if not os.path.exists(sample_image_path):
        print(f"Sample image not found at {sample_image_path}")
        print("This is just an example. Please provide a valid image path.")
    else:
        # Predict with explanations
        result = system.predict(sample_image_path)
        
        # Visualize results
        system.visualize_results(result, output_path="../results/sample_explanation.png")
        
        print(f"Predicted disease: {result['prediction']['class_name']}")
        print(f"Confidence: {result['prediction']['confidence']:.4f}")
        print(f"Explanation: {result['nl_explanation']['full_explanation']}")
        print("\nRecommendations:")
        print(f"Treatment: {result['recommendations']['recommendations']['treatment']}")
