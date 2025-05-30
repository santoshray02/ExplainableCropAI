#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Recommendation engine for ExplainableCropAI.

This module generates context-aware treatment and management recommendations
based on disease predictions, uncertainty estimates, and domain knowledge.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from recommendations.treatment_db import PaddyDiseaseKnowledgeBase


class RecommendationEngine:
    """
    Generates context-aware recommendations for paddy disease management.
    
    Combines model predictions, uncertainty estimates, and agricultural domain
    knowledge to provide actionable recommendations.
    """
    
    def __init__(self, knowledge_base=None):
        """
        Initialize the recommendation engine.
        
        Args:
            knowledge_base (PaddyDiseaseKnowledgeBase): Knowledge base of disease information.
                                                       If None, creates a new instance.
        """
        if knowledge_base is None:
            self.kb = PaddyDiseaseKnowledgeBase()
        else:
            self.kb = knowledge_base
    
    def determine_severity(self, confidence, affected_area_percentage=None):
        """
        Determine disease severity based on model confidence and affected area.
        
        Args:
            confidence (float): Model confidence score (0-1)
            affected_area_percentage (float): Percentage of the plant affected (0-100)
                                            If None, relies solely on confidence
                                            
        Returns:
            str: Severity level ('low', 'medium', or 'high')
        """
        # Default to using just confidence if affected area is not provided
        if affected_area_percentage is None:
            if confidence < 0.5:
                return "low"
            elif confidence < 0.8:
                return "medium"
            else:
                return "high"
        
        # When both confidence and affected area are available
        # Convert confidence to 0-100 scale
        confidence_score = confidence * 100
        
        # Calculate combined severity score (equal weighting)
        combined_score = (confidence_score + affected_area_percentage) / 2
        
        if combined_score < 40:
            return "low"
        elif combined_score < 70:
            return "medium"
        else:
            return "high"
    
    def assess_risk_factors(self, metadata=None):
        """
        Assess environmental and contextual risk factors.
        
        Args:
            metadata (dict): Additional context information that may include:
                           - weather_conditions: Current weather
                           - forecast: Weather forecast
                           - crop_stage: Growth stage of the crop
                           - field_history: Previous disease occurrences
                           - soil_conditions: Soil pH, moisture, etc.
                           
        Returns:
            dict: Risk assessment and relevant factors
        """
        risk_level = "medium"  # Default risk level
        risk_factors = []
        mitigating_factors = []
        
        if metadata is None:
            return {
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "mitigating_factors": mitigating_factors,
                "notes": "Limited contextual information available for risk assessment."
            }
        
        # Assess weather conditions
        if "weather_conditions" in metadata:
            weather = metadata["weather_conditions"]
            if "humidity" in weather and weather["humidity"] > 85:
                risk_level = "high"
                risk_factors.append("High humidity (>85%) favors disease development")
            
            if "temperature" in weather:
                temp = weather["temperature"]
                if 25 <= temp <= 30:  # Ideal temperature range for many paddy diseases
                    risk_factors.append(f"Temperature ({temp}°C) is optimal for disease development")
                elif temp < 20 or temp > 35:
                    mitigating_factors.append(f"Temperature ({temp}°C) is less favorable for disease spread")
        
        # Assess crop stage
        if "crop_stage" in metadata:
            stage = metadata["crop_stage"]
            if stage in ["flowering", "heading"]:
                risk_factors.append(f"Crop is in {stage} stage, which is highly susceptible to disease")
            elif stage in ["maturity", "harvesting"]:
                mitigating_factors.append(f"Crop is in {stage} stage, approaching harvest")
        
        # Assess field history
        if "field_history" in metadata and metadata["field_history"].get("previous_occurrence", False):
            risk_factors.append("Field has history of this disease, increasing risk of recurrence")
        
        # Determine overall risk based on factors
        if len(risk_factors) > len(mitigating_factors) + 1:
            risk_level = "high"
        elif len(mitigating_factors) > len(risk_factors):
            risk_level = "low"
        
        return {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "mitigating_factors": mitigating_factors,
            "notes": "Assessment based on provided contextual information."
        }
    
    def generate_recommendations(self, prediction, uncertainty=None, metadata=None):
        """
        Generate treatment and management recommendations.
        
        Args:
            prediction (dict): Disease prediction including:
                             - class_name: Predicted disease
                             - confidence: Prediction confidence
            uncertainty (dict): Uncertainty information (optional)
            metadata (dict): Additional context information (optional)
            
        Returns:
            dict: Recommendations and supporting information
        """
        disease_name = prediction.get("class_name", "Unknown")
        confidence = prediction.get("confidence", 0.5)
        
        # Get basic disease information
        disease_info = self.kb.get_disease_info(disease_name)
        
        if disease_info is None:
            return {
                "status": "error",
                "message": f"No information available for disease: {disease_name}"
            }
        
        # Calculate severity based on confidence and affected area (if available)
        affected_area = metadata.get("affected_area_percentage") if metadata else None
        severity = self.determine_severity(confidence, affected_area)
        
        # Get severity-specific information
        severity_info = self.kb.get_treatment_for_severity(disease_name, severity)
        
        # Assess risk factors
        risk_assessment = self.assess_risk_factors(metadata)
        
        # Generate basic recommendations
        treatment_rec = severity_info.get("treatment") if severity_info else disease_info.get("treatment")
        prevention_rec = disease_info.get("prevention", "")
        
        # Adjust recommendations based on uncertainty if available
        if uncertainty and "entropy" in uncertainty:
            uncertainty_level = "high" if uncertainty["entropy"] > 0.5 else "medium" if uncertainty["entropy"] > 0.2 else "low"
            
            if uncertainty_level == "high":
                verification_note = "Due to high uncertainty in the diagnosis, consider getting expert verification."
                conservative_treatment = "Start with conservative treatment measures while seeking expert confirmation."
                treatment_rec = f"{conservative_treatment} {treatment_rec}"
            else:
                verification_note = ""
        else:
            uncertainty_level = "unknown"
            verification_note = ""
        
        # Adjust recommendations based on risk assessment
        if risk_assessment["risk_level"] == "high":
            treatment_rec = f"Due to high risk factors, consider more aggressive treatment: {treatment_rec}"
        elif risk_assessment["risk_level"] == "low":
            treatment_rec = f"Given favorable conditions, a moderate approach may be sufficient: {treatment_rec}"
        
        return {
            "status": "success",
            "disease": disease_name,
            "confidence": confidence,
            "severity": severity,
            "uncertainty_level": uncertainty_level,
            "risk_level": risk_assessment["risk_level"],
            "recommendations": {
                "treatment": treatment_rec,
                "prevention": prevention_rec,
                "verification": verification_note
            },
            "supporting_info": {
                "disease_info": disease_info,
                "severity_info": severity_info,
                "risk_factors": risk_assessment["risk_factors"],
                "mitigating_factors": risk_assessment["mitigating_factors"]
            }
        }


if __name__ == "__main__":
    # Example usage
    engine = RecommendationEngine()
    
    # Sample prediction
    prediction = {
        "class_name": "Bacterial Leaf Blight",
        "confidence": 0.85
    }
    
    # Sample uncertainty
    uncertainty = {
        "entropy": 0.3,
        "mutual_information": 0.15
    }
    
    # Sample metadata
    metadata = {
        "weather_conditions": {
            "temperature": 28,
            "humidity": 90,
            "rainfall_recent": True
        },
        "crop_stage": "flowering",
        "affected_area_percentage": 60,
        "field_history": {
            "previous_occurrence": True,
            "last_occurrence": "previous season"
        }
    }
    
    # Generate recommendations
    recommendations = engine.generate_recommendations(prediction, uncertainty, metadata)
    
    # Print recommendations
    print(f"Disease: {recommendations['disease']}")
    print(f"Confidence: {recommendations['confidence']}")
    print(f"Severity: {recommendations['severity']}")
    print(f"Uncertainty level: {recommendations['uncertainty_level']}")
    print(f"Risk level: {recommendations['risk_level']}")
    print("\nRecommendations:")
    print(f"Treatment: {recommendations['recommendations']['treatment']}")
    print(f"Prevention: {recommendations['recommendations']['prevention']}")
    
    if recommendations['recommendations']['verification']:
        print(f"Verification note: {recommendations['recommendations']['verification']}")
    
    print("\nRisk factors:")
    for factor in recommendations['supporting_info']['risk_factors']:
        print(f"- {factor}")
    
    print("\nMitigating factors:")
    for factor in recommendations['supporting_info']['mitigating_factors']:
        print(f"- {factor}")
