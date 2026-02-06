#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main integration module for ExplainableCropAI.

This module integrates all components of the ExplainableCropAI system,
providing a unified interface for paddy disease detection with explanations.

Supports both legacy (ResNet50/GradCAM) and modern (ConvNeXtV2/GradCAM++)
architectures through configuration options.
"""

import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from typing import Optional, Dict, Any, Union

# Import legacy components (for backward compatibility)
from base_model import PaddyDiseaseClassifier
from explainers.gradcam import GradCAM, visualize_explanation
from nlg.template_generator import NLExplanationGenerator, ExplanationWithVisualEvidence

# Import modern components
from models.convnext_v2 import ConvNeXtV2Classifier
from explainers.gradcam_plus import GradCAMPlusPlus, LayerCAM
from nlg.llm_generator import LLMExplanationGenerator, ExplanationWithLLM, AudienceLevel

# Import shared components
from uncertainty.mc_dropout import MCDropout, uncertainty_to_confidence
from recommendations.recommendation_engine import RecommendationEngine
from recommendations.treatment_db import PaddyDiseaseKnowledgeBase


class ExplainableCropAI:
    """
    Integrated system for explainable paddy disease detection.

    Combines disease detection, visual explanations, uncertainty estimation,
    natural language explanations, and recommendations.

    Supports two configurations:
    - Legacy: ResNet50 + GradCAM + Template-based NLG
    - Modern: ConvNeXtV2 + GradCAM++ + LLM-based NLG (Phi-3)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_classes: int = 10,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the ExplainableCropAI system.

        Args:
            model_path (str): Path to trained model weights
            num_classes (int): Number of disease classes
            device (torch.device): Device to run model on
            config (dict): Configuration options including:
                - use_convnext (bool): Use ConvNeXtV2 instead of ResNet50 (default: True)
                - use_gradcam_plus (bool): Use GradCAM++ instead of GradCAM (default: True)
                - use_llm (bool): Use LLM-based explanation generation (default: True)
                - llm_model (str): HuggingFace model name for LLM (default: Phi-3-mini)
                - llm_load_in_4bit (bool): Load LLM in 4-bit quantization (default: False)
                - llm_preload (bool): Preload LLM model at init (default: False)
                - audience (str): Default audience level for explanations (default: 'farmer')
                - model_size (str): ConvNeXtV2 size: 'tiny', 'small', 'base' (default: 'tiny')
                - cam_method (str): CAM method: 'gradcam', 'gradcam++', 'layercam' (default: 'gradcam++')
        """
        # Parse configuration with defaults
        if config is None:
            config = {}

        self.config = {
            'use_convnext': config.get('use_convnext', True),
            'use_gradcam_plus': config.get('use_gradcam_plus', True),
            'use_llm': config.get('use_llm', True),
            'llm_model': config.get('llm_model', 'microsoft/Phi-3-mini-4k-instruct'),
            'llm_load_in_4bit': config.get('llm_load_in_4bit', False),
            'llm_preload': config.get('llm_preload', False),
            'audience': config.get('audience', 'farmer'),
            'model_size': config.get('model_size', 'tiny'),
            'cam_method': config.get('cam_method', 'gradcam++'),
        }

        # Set device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model based on configuration
        if self.config['use_convnext']:
            self._init_convnext_model(num_classes, model_path)
        else:
            self._init_resnet_model(num_classes, model_path)

        # Class names matching Paddy Doctor dataset (10 classes)
        self.class_names = [
            "Bacterial Leaf Blight",
            "Brown Spot",
            "Leaf Blast",
            "Bacterial Leaf Streak",
            "Bacterial Sheath Rot",
            "Healthy",
            "Tungro",
            "Hispa",
            "Dead Heart",
            "Downy Mildew"
        ]

        # Initialize visual explainer based on configuration
        self._init_visual_explainer()

        # Initialize NLG generator based on configuration
        self._init_nlg_generator()

        # Initialize shared components
        self.uncertainty_estimator = MCDropout(self.model, num_samples=30)
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

    def _init_convnext_model(self, num_classes: int, model_path: Optional[str]) -> None:
        """Initialize ConvNeXtV2 model."""
        try:
            self.model = ConvNeXtV2Classifier(
                num_classes=num_classes,
                pretrained=True if model_path is None else False
            )

            # Load model weights if provided
            if model_path and os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                # Handle different checkpoint formats
                if 'model_state_dict' in state_dict:
                    self.model.load_state_dict(state_dict['model_state_dict'])
                elif 'state_dict' in state_dict:
                    self.model.load_state_dict(state_dict['state_dict'])
                else:
                    self.model.load_state_dict(state_dict)

            self.model = self.model.to(self.device)
            self.model.eval()
            self.model_type = 'convnext'

        except ImportError as e:
            print(f"Warning: Could not load ConvNeXtV2 ({e}). Falling back to ResNet50.")
            self._init_resnet_model(num_classes, model_path)

    def _init_resnet_model(self, num_classes: int, model_path: Optional[str]) -> None:
        """Initialize legacy ResNet50 model."""
        self.model = PaddyDiseaseClassifier(num_classes=num_classes)

        # Load model weights if provided
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model = self.model.to(self.device)
        self.model.eval()
        self.model_type = 'resnet'
        self.config['use_convnext'] = False

    def _init_visual_explainer(self) -> None:
        """Initialize the visual explainer based on configuration."""
        cam_method = self.config['cam_method'].lower()

        if cam_method == 'gradcam++' or (self.config['use_gradcam_plus'] and cam_method != 'layercam'):
            self.visual_explainer = GradCAMPlusPlus(self.model)
            self.explainer_type = 'gradcam++'
        elif cam_method == 'layercam':
            self.visual_explainer = LayerCAM(self.model)
            self.explainer_type = 'layercam'
        else:
            self.visual_explainer = GradCAM(self.model)
            self.explainer_type = 'gradcam'

    def _init_nlg_generator(self) -> None:
        """Initialize the NLG generator based on configuration."""
        if self.config['use_llm']:
            try:
                self.llm_generator = LLMExplanationGenerator(
                    model_name=self.config['llm_model'],
                    device=str(self.device),
                    load_in_4bit=self.config['llm_load_in_4bit']
                )

                self.nlg_generator = ExplanationWithLLM(
                    llm_generator=self.llm_generator,
                    preload_model=self.config['llm_preload']
                )
                self.nlg_type = 'llm'
            except Exception as e:
                print(f"Warning: Could not initialize LLM generator ({e}). Using template-based fallback.")
                self._init_template_nlg()
        else:
            self._init_template_nlg()

    def _init_template_nlg(self) -> None:
        """Initialize template-based NLG generator (fallback)."""
        self.nlg_generator = ExplanationWithVisualEvidence(
            NLExplanationGenerator()
        )
        self.nlg_type = 'template'
        self.llm_generator = None
    
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
    
    def predict(
        self,
        image_path: str,
        with_explanation: bool = True,
        audience: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Predict disease and generate explanations.

        Args:
            image_path (str): Path to image file
            with_explanation (bool): Whether to generate explanations
            audience (str): Target audience for explanations ('farmer', 'expert', 'researcher')
                           Defaults to config setting

        Returns:
            dict: Prediction results and explanations
        """
        # Preprocess image
        input_tensor, original_image = self.preprocess_image(image_path)

        # Base prediction - handle both model output formats
        with torch.no_grad():
            output = self.model(input_tensor)

            # Handle different model output formats
            if isinstance(output, tuple):
                logits = output[0]
                additional_outputs = output[1] if len(output) > 1 else {}
            else:
                logits = output
                additional_outputs = {}

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
            "probabilities": probabilities[0].cpu().numpy(),
            "model_type": self.model_type
        }

        # Add features from ConvNeXtV2 if available
        if 'features' in additional_outputs:
            prediction_info["features"] = additional_outputs['features'].cpu().numpy()

        # Return just prediction if explanation not requested
        if not with_explanation:
            return prediction_info

        # Generate visual explanation (GradCAM/GradCAM++/LayerCAM)
        visual_explanation = self.visual_explainer.generate(input_tensor, class_idx=predicted_class)

        # Add explainer type info to visual explanation
        visual_explanation['explainer_type'] = self.explainer_type

        # Estimate uncertainty
        uncertainty_info = self.uncertainty_estimator.predict(input_tensor)
        adjusted_confidence = uncertainty_to_confidence(uncertainty_info)

        # Update prediction with uncertainty-adjusted confidence
        prediction_info["raw_confidence"] = prediction_info["confidence"]
        prediction_info["confidence"] = adjusted_confidence

        # Estimate affected area
        affected_area = self._estimate_affected_area(visual_explanation)
        prediction_info["affected_area_percentage"] = affected_area

        # Detect visual features from explanation
        visual_features = self._extract_visual_features(visual_explanation, prediction_info["class_name"])

        # Generate natural language explanation
        disease_info = self.recommendation_engine.kb.get_disease_info(prediction_info["class_name"])

        # Determine target audience
        target_audience = audience if audience else self.config.get('audience', 'farmer')

        # Generate explanation based on NLG type
        if self.nlg_type == 'llm':
            # Use LLM-based generator with uncertainty info
            nl_explanation = self.nlg_generator.generate_explanation(
                prediction_info=prediction_info,
                visual_features=visual_features,
                uncertainty_info=uncertainty_info,
                disease_info=disease_info,
                audience=target_audience
            )
        else:
            # Use template-based generator (legacy interface)
            nl_explanation = self.nlg_generator.generate_explanation(
                prediction_info=prediction_info,
                visual_features=visual_features,
                disease_info=disease_info
            )

        # Generate recommendations
        metadata = {
            "affected_area_percentage": affected_area
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
            "visual_features": visual_features,
            "config": {
                "model_type": self.model_type,
                "explainer_type": self.explainer_type,
                "nlg_type": self.nlg_type,
                "audience": target_audience
            }
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
        
        # 2. Grad-CAM/Grad-CAM++/LayerCAM visualization
        ax2 = plt.subplot2grid((3, 6), (0, 2), colspan=2, rowspan=1)
        ax2.imshow(visual_explanation["overlay"])
        explainer_name = visual_explanation.get('explainer_type', visual_explanation.get('method', 'CAM'))
        ax2.set_title(f"Visual Explanation ({explainer_name})")
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
    import argparse

    parser = argparse.ArgumentParser(description='ExplainableCropAI - Paddy Disease Detection with Explanations')
    parser.add_argument('--image', type=str, default='../data/sample_image.jpg',
                       help='Path to input image')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to model weights')
    parser.add_argument('--legacy', action='store_true',
                       help='Use legacy mode (ResNet50 + GradCAM + Templates)')
    parser.add_argument('--cam-method', type=str, default='gradcam++',
                       choices=['gradcam', 'gradcam++', 'layercam'],
                       help='CAM visualization method')
    parser.add_argument('--audience', type=str, default='farmer',
                       choices=['farmer', 'expert', 'researcher'],
                       help='Target audience for explanations')
    parser.add_argument('--no-llm', action='store_true',
                       help='Disable LLM-based explanation generation')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for visualization')

    args = parser.parse_args()

    # Build configuration based on arguments
    config = {
        'use_convnext': not args.legacy,
        'use_gradcam_plus': args.cam_method != 'gradcam',
        'use_llm': not args.no_llm and not args.legacy,
        'cam_method': args.cam_method,
        'audience': args.audience,
        'llm_preload': False,  # Don't preload LLM to save memory
    }

    print("=" * 70)
    print("ExplainableCropAI - Interpretable Paddy Disease Detection")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {'ConvNeXtV2' if config['use_convnext'] else 'ResNet50 (legacy)'}")
    print(f"  Visual Explainer: {args.cam_method.upper()}")
    print(f"  NLG Generator: {'LLM (Phi-3)' if config['use_llm'] else 'Template-based'}")
    print(f"  Target Audience: {args.audience}")

    # Initialize system
    print("\nInitializing system...")
    system = ExplainableCropAI(
        model_path=args.model_path,
        config=config
    )

    print(f"  Actual model type: {system.model_type}")
    print(f"  Actual explainer: {system.explainer_type}")
    print(f"  Actual NLG: {system.nlg_type}")

    # Sample image path
    sample_image_path = args.image

    # Check if the file exists
    if not os.path.exists(sample_image_path):
        print(f"\nSample image not found at {sample_image_path}")
        print("This is just an example. Please provide a valid image path with --image")
        print("\nExample usage:")
        print("  python explainable_system.py --image path/to/image.jpg")
        print("  python explainable_system.py --legacy --image path/to/image.jpg")
        print("  python explainable_system.py --cam-method layercam --audience expert")
    else:
        print(f"\nProcessing image: {sample_image_path}")

        # Predict with explanations
        result = system.predict(sample_image_path, audience=args.audience)

        # Display results
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        print(f"\nPredicted Disease: {result['prediction']['class_name']}")
        print(f"Confidence: {result['prediction']['confidence']:.4f}")
        print(f"Raw Confidence: {result['prediction'].get('raw_confidence', 'N/A'):.4f}")
        print(f"Model Type: {result['prediction']['model_type']}")

        print(f"\nUncertainty Metrics:")
        print(f"  Entropy: {result['uncertainty']['entropy']:.4f}")
        print(f"  Mutual Information: {result['uncertainty']['mutual_information']:.4f}")
        print(f"  Class Agreement: {result['uncertainty']['class_agreement']:.4f}")

        print(f"\nVisual Features Detected: {', '.join(result['visual_features'])}")

        print(f"\nExplanation (Source: {result['nl_explanation'].get('source', 'unknown')}):")
        print("-" * 50)
        print(result['nl_explanation']['full_explanation'])
        print("-" * 50)

        print(f"\nRecommendations:")
        print(f"  Severity: {result['recommendations']['severity']}")
        print(f"  Risk Level: {result['recommendations']['risk_level']}")
        print(f"  Treatment: {result['recommendations']['recommendations']['treatment']}")

        # Visualize results
        output_path = args.output or "../results/sample_explanation.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        system.visualize_results(result, output_path=output_path)
        print(f"\nVisualization saved to: {output_path}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
