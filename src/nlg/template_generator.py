#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Template-based natural language explanation generator for ExplainableCropAI.

This module generates natural language explanations for model predictions
using templates filled with disease-specific information and visual evidence.
"""

import json
import os
import random
from string import Template


class NLExplanationGenerator:
    """
    Natural language explanation generator for paddy disease predictions.
    
    Generates human-readable explanations by combining templates with
    model prediction information, visual evidence, and domain knowledge.
    """
    
    def __init__(self, templates_file=None):
        """
        Initialize the explanation generator.
        
        Args:
            templates_file (str): Path to JSON file with explanation templates.
                                 If None, uses default templates.
        """
        self.templates = {}
        
        if templates_file and os.path.exists(templates_file):
            with open(templates_file, 'r') as f:
                self.templates = json.load(f)
        else:
            # Default templates
            self.templates = {
                "disease_detection": [
                    "This image shows a paddy plant with $disease_name. The key visual indicators include $visual_evidence.",
                    "I've identified $disease_name in this paddy plant. This is evident from the $visual_evidence visible in the image.",
                    "The paddy plant in this image is affected by $disease_name, characterized by $visual_evidence."
                ],
                "confidence_high": [
                    "The model is highly confident in this diagnosis (confidence: $confidence%).",
                    "With $confidence% confidence, this diagnosis is very reliable.",
                    "The visual evidence strongly supports this diagnosis (confidence: $confidence%)."
                ],
                "confidence_medium": [
                    "The model has moderate confidence in this diagnosis (confidence: $confidence%).",
                    "This diagnosis is supported with moderate confidence ($confidence%).",
                    "The visual evidence provides reasonable support for this diagnosis (confidence: $confidence%)."
                ],
                "confidence_low": [
                    "The model has low confidence in this diagnosis (confidence: $confidence%). Consider consulting an expert.",
                    "This diagnosis is uncertain (confidence: $confidence%). Additional inspection may be needed.",
                    "The visual evidence provides limited support for this diagnosis (confidence: $confidence%)."
                ],
                "treatment": [
                    "Recommended treatment: $treatment.",
                    "To address this issue, $treatment.",
                    "The appropriate management strategy is to $treatment."
                ],
                "prevention": [
                    "To prevent future occurrences, $prevention.",
                    "Preventive measures include $prevention.",
                    "For prevention, it's advisable to $prevention."
                ],
                "impact": [
                    "If left untreated, this disease can $impact.",
                    "The potential impact of this disease includes $impact.",
                    "This disease typically affects the crop by $impact."
                ],
                "comparison": [
                    "This could also be confused with $similar_disease, but the $distinguishing_feature indicates it's $disease_name.",
                    "While similar to $similar_disease, the presence of $distinguishing_feature confirms this as $disease_name.",
                    "$disease_name is distinguished from $similar_disease by $distinguishing_feature."
                ]
            }
    
    def _get_template(self, template_type):
        """
        Get a random template of the specified type.
        
        Args:
            template_type (str): Type of template to retrieve.
            
        Returns:
            str: Template string.
        """
        if template_type in self.templates:
            return random.choice(self.templates[template_type])
        return ""
    
    def _format_confidence(self, confidence_score):
        """
        Format confidence score and determine confidence level.
        
        Args:
            confidence_score (float): Model confidence score (0-1).
            
        Returns:
            tuple: Formatted confidence percentage and confidence level.
        """
        confidence_pct = round(confidence_score * 100)
        
        if confidence_score >= 0.8:
            confidence_level = "high"
        elif confidence_score >= 0.5:
            confidence_level = "medium"
        else:
            confidence_level = "low"
            
        return confidence_pct, confidence_level
    
    def generate_explanation(self, prediction_info, disease_info=None, visual_evidence=None):
        """
        Generate a natural language explanation for a disease prediction.
        
        Args:
            prediction_info (dict): Information about the model prediction including:
                                   - class_name: Predicted disease name
                                   - confidence: Confidence score (0-1)
            disease_info (dict): Domain knowledge about the predicted disease including:
                                - symptoms: Description of symptoms
                                - treatment: Recommended treatment
                                - prevention: Prevention measures
                                - impact: Potential impact if untreated
                                - similar_diseases: Similar diseases
                                - distinguishing_features: Features distinguishing from similar diseases
            visual_evidence (str): Description of visual evidence found in the image.
                                  If None, uses generic description based on disease_info.
                                  
        Returns:
            dict: Generated explanation including overall explanation and components.
        """
        # Extract prediction information
        disease_name = prediction_info.get('class_name', 'Unknown disease')
        confidence_score = prediction_info.get('confidence', 0.0)
        confidence_pct, confidence_level = self._format_confidence(confidence_score)
        
        # Use provided disease info or create placeholder
        if disease_info is None:
            disease_info = {
                'symptoms': 'characteristic symptoms',
                'treatment': 'apply appropriate fungicide and ensure proper drainage',
                'prevention': 'maintain proper water management and use resistant varieties',
                'impact': 'reduce yield by affecting photosynthesis and nutrient uptake',
                'similar_diseases': 'other common paddy diseases',
                'distinguishing_features': 'specific visual patterns'
            }
        
        # Use provided visual evidence or generate from symptoms
        if visual_evidence is None:
            visual_evidence = disease_info.get('symptoms', 'visible symptoms')
        
        # Prepare template variables
        template_vars = {
            'disease_name': disease_name,
            'visual_evidence': visual_evidence,
            'confidence': confidence_pct,
            'treatment': disease_info.get('treatment', ''),
            'prevention': disease_info.get('prevention', ''),
            'impact': disease_info.get('impact', ''),
            'similar_disease': disease_info.get('similar_diseases', ''),
            'distinguishing_feature': disease_info.get('distinguishing_features', '')
        }
        
        # Generate explanation components
        detection_explanation = Template(self._get_template('disease_detection')).substitute(template_vars)
        confidence_explanation = Template(self._get_template(f'confidence_{confidence_level}')).substitute(template_vars)
        treatment_explanation = Template(self._get_template('treatment')).substitute(template_vars)
        prevention_explanation = Template(self._get_template('prevention')).substitute(template_vars)
        impact_explanation = Template(self._get_template('impact')).substitute(template_vars)
        
        # Add comparison only if similar disease info is available
        comparison_explanation = ""
        if disease_info.get('similar_diseases') and disease_info.get('distinguishing_features'):
            comparison_explanation = Template(self._get_template('comparison')).substitute(template_vars)
        
        # Combine explanation components
        full_explanation = f"{detection_explanation} {confidence_explanation} {treatment_explanation} {impact_explanation}"
        
        if comparison_explanation:
            full_explanation += f" {comparison_explanation}"
            
        full_explanation += f" {prevention_explanation}"
        
        # Return structured explanation
        return {
            'full_explanation': full_explanation,
            'components': {
                'detection': detection_explanation,
                'confidence': confidence_explanation,
                'treatment': treatment_explanation,
                'prevention': prevention_explanation,
                'impact': impact_explanation,
                'comparison': comparison_explanation
            },
            'confidence_level': confidence_level,
            'confidence_pct': confidence_pct
        }


class ExplanationWithVisualEvidence:
    """
    Combines visual explanations with natural language explanations.
    
    Links specific regions highlighted by visual explanation techniques
    (like Grad-CAM) to natural language descriptions.
    """
    
    def __init__(self, nl_generator=None):
        """
        Initialize the explanation generator.
        
        Args:
            nl_generator (NLExplanationGenerator): Natural language explanation generator.
                                                  If None, creates a new instance.
        """
        if nl_generator is None:
            self.nl_generator = NLExplanationGenerator()
        else:
            self.nl_generator = nl_generator
            
        # Visual feature to textual description mapping
        self.visual_feature_mapping = {
            'leaf_spots': 'brown or yellow spots on leaves',
            'leaf_discoloration': 'yellowing or browning of leaves',
            'leaf_lesions': 'elongated lesions on leaves',
            'stem_lesions': 'dark lesions on stems',
            'white_powder': 'white powdery substance on leaves',
            'stunted_growth': 'abnormally small or stunted growth',
            'wilting': 'drooping or wilting of leaves'
        }
    
    def generate_visual_evidence_description(self, visual_features):
        """
        Generate textual description of visual evidence.
        
        Args:
            visual_features (list): List of identified visual features.
            
        Returns:
            str: Textual description of visual evidence.
        """
        if not visual_features:
            return "visible symptoms in the affected areas"
            
        descriptions = [self.visual_feature_mapping.get(feature, feature) for feature in visual_features]
        
        if len(descriptions) == 1:
            return descriptions[0]
        elif len(descriptions) == 2:
            return f"{descriptions[0]} and {descriptions[1]}"
        else:
            return ", ".join(descriptions[:-1]) + f", and {descriptions[-1]}"
    
    def generate_explanation(self, prediction_info, visual_features, disease_info=None):
        """
        Generate explanation combining visual evidence and natural language.
        
        Args:
            prediction_info (dict): Prediction information.
            visual_features (list): List of identified visual features.
            disease_info (dict): Disease information.
            
        Returns:
            dict: Combined explanation.
        """
        visual_evidence = self.generate_visual_evidence_description(visual_features)
        
        explanation = self.nl_generator.generate_explanation(
            prediction_info=prediction_info,
            disease_info=disease_info,
            visual_evidence=visual_evidence
        )
        
        # Add visual feature information
        explanation['visual_evidence'] = visual_evidence
        explanation['visual_features'] = visual_features
        
        return explanation


if __name__ == "__main__":
    # Example usage
    generator = NLExplanationGenerator()
    
    # Sample prediction
    prediction = {
        'class_name': 'Bacterial Leaf Blight',
        'confidence': 0.92
    }
    
    # Sample disease information
    disease_info = {
        'symptoms': 'water-soaked lesions that turn yellow-orange',
        'treatment': 'apply copper-based bactericides and ensure good field drainage',
        'prevention': 'use resistant varieties and practice crop rotation',
        'impact': 'cause yield losses of up to 70% in severely affected fields',
        'similar_diseases': 'Bacterial Leaf Streak',
        'distinguishing_features': 'wavy margin of lesions instead of straight edges'
    }
    
    # Generate explanation
    explanation = generator.generate_explanation(prediction, disease_info)
    
    print("Generated Explanation:")
    print(explanation['full_explanation'])
    print("\nComponents:")
    for component, text in explanation['components'].items():
        if text:
            print(f"- {component.capitalize()}: {text}")
    
    # Example with visual evidence
    visual_explainer = ExplanationWithVisualEvidence()
    visual_features = ['leaf_lesions', 'leaf_discoloration']
    
    visual_explanation = visual_explainer.generate_explanation(
        prediction_info=prediction,
        visual_features=visual_features,
        disease_info=disease_info
    )
    
    print("\nExplanation with Visual Evidence:")
    print(visual_explanation['full_explanation'])
