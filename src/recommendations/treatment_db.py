#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Database of treatments for paddy diseases in ExplainableCropAI.

This module provides a knowledge base of treatments, prevention methods,
and impact information for various paddy diseases.
"""

class PaddyDiseaseKnowledgeBase:
    """
    Knowledge base for paddy diseases, treatments, and prevention methods.
    """
    
    def __init__(self):
        """Initialize the knowledge base with disease information."""
        self.disease_info = {
            "Bacterial Leaf Blight": {
                "symptoms": "water-soaked lesions that turn yellow-orange with wavy margins",
                "treatment": "apply copper-based bactericides and ensure good field drainage",
                "prevention": "use resistant varieties, practice crop rotation, and avoid overhead irrigation",
                "impact": "cause yield losses of up to 70% in severely affected fields",
                "similar_diseases": "Bacterial Leaf Streak",
                "distinguishing_features": "wavy margins of lesions instead of straight edges",
                "severity_levels": {
                    "low": {
                        "description": "Early stage with few lesions confined to leaf tips",
                        "treatment": "improve field drainage and monitor crop carefully",
                        "expected_yield_impact": "minimal impact if addressed promptly"
                    },
                    "medium": {
                        "description": "Multiple leaves affected with lesions spreading from margins",
                        "treatment": "apply copper-based bactericides and improve field drainage",
                        "expected_yield_impact": "potential yield reduction of 15-30% if untreated"
                    },
                    "high": {
                        "description": "Widespread infection with lesions covering large portions of leaves",
                        "treatment": "aggressive application of copper-based bactericides, isolation of affected areas",
                        "expected_yield_impact": "severe yield losses of 40-70% if untreated"
                    }
                }
            },
            "Brown Spot": {
                "symptoms": "oval or circular brown lesions with yellow halos",
                "treatment": "apply fungicides containing propiconazole or hexaconazole",
                "prevention": "maintain balanced crop nutrition, especially potassium",
                "impact": "reduce grain yield and quality, especially in stressed plants",
                "similar_diseases": "Narrow Brown Spot",
                "distinguishing_features": "circular spots with distinct yellow halos",
                "severity_levels": {
                    "low": {
                        "description": "Few isolated spots on lower leaves",
                        "treatment": "correct soil nutrient deficiencies, especially potassium",
                        "expected_yield_impact": "minimal impact if soil nutrients are balanced"
                    },
                    "medium": {
                        "description": "Multiple leaves with numerous spots, beginning to affect middle canopy",
                        "treatment": "apply recommended fungicides and correct nutrient deficiencies",
                        "expected_yield_impact": "potential yield reduction of 10-25% if untreated"
                    },
                    "high": {
                        "description": "Extensive spotting throughout canopy, with leaf drying and senescence",
                        "treatment": "immediate fungicide application, possibly repeated treatments",
                        "expected_yield_impact": "yield losses of 30-45% or more if untreated"
                    }
                }
            },
            "Leaf Blast": {
                "symptoms": "diamond-shaped lesions with gray centers and brown margins",
                "treatment": "apply triazole or strobilurin fungicides at recommended rates",
                "prevention": "use resistant varieties and avoid excessive nitrogen fertilization",
                "impact": "cause severe defoliation and yield losses up to 80%",
                "similar_diseases": "Brown Spot",
                "distinguishing_features": "diamond-shaped lesions with pointed ends",
                "severity_levels": {
                    "low": {
                        "description": "Small, isolated lesions on a few leaves",
                        "treatment": "reduce nitrogen application and improve water management",
                        "expected_yield_impact": "minimal if environmental conditions don't favor disease spread"
                    },
                    "medium": {
                        "description": "Multiple diamond-shaped lesions on several leaves, beginning to coalesce",
                        "treatment": "apply recommended fungicides and adjust nitrogen management",
                        "expected_yield_impact": "potential yield reduction of 20-40% if untreated"
                    },
                    "high": {
                        "description": "Widespread lesions causing leaf death, potential neck blast symptoms",
                        "treatment": "immediate application of systemic fungicides, possibly multiple treatments",
                        "expected_yield_impact": "severe yield losses of 50-80% if untreated"
                    }
                }
            },
            "Bacterial Leaf Streak": {
                "symptoms": "narrow, linear lesions with straight edges between leaf veins",
                "treatment": "apply copper-based bactericides and avoid overhead irrigation",
                "prevention": "use disease-free seeds and resistant varieties",
                "impact": "reduce photosynthetic area and yield potential by 10-30%",
                "similar_diseases": "Bacterial Leaf Blight",
                "distinguishing_features": "straight-edged lesions strictly between veins",
                "severity_levels": {
                    "low": {
                        "description": "Few narrow streaks confined to lower leaves",
                        "treatment": "improve field sanitation and avoid overhead irrigation",
                        "expected_yield_impact": "minimal impact if contained to lower leaves"
                    },
                    "medium": {
                        "description": "Multiple leaves with streaks, beginning to affect upper canopy",
                        "treatment": "apply copper-based bactericides and improve sanitation",
                        "expected_yield_impact": "potential yield reduction of 10-20% if untreated"
                    },
                    "high": {
                        "description": "Extensive streaking throughout the canopy, with leaf yellowing and drying",
                        "treatment": "copper bactericides, sanitation, and consider early harvest if severe",
                        "expected_yield_impact": "yield losses of 20-30% or more if untreated"
                    }
                }
            },
            "Sheath Blight": {
                "symptoms": "irregularly shaped lesions on leaf sheaths with gray-white centers and brown margins",
                "treatment": "apply fungicides containing azoxystrobin or propiconazole",
                "prevention": "use wider plant spacing and moderate nitrogen fertilization",
                "impact": "reduce yield by 20-50% and affect grain quality",
                "similar_diseases": "Sheath Rot",
                "distinguishing_features": "lesions that often form rings or bands around the sheath",
                "severity_levels": {
                    "low": {
                        "description": "Few lesions on lower leaf sheaths, limited vertical spread",
                        "treatment": "adjust plant density and nitrogen management",
                        "expected_yield_impact": "5-10% if disease progression is halted"
                    },
                    "medium": {
                        "description": "Lesions on multiple sheaths, beginning to reach upper parts of plants",
                        "treatment": "apply recommended fungicides and improve air circulation",
                        "expected_yield_impact": "potential yield reduction of 15-30% if untreated"
                    },
                    "high": {
                        "description": "Extensive lesions reaching flag leaf sheath and potentially panicles",
                        "treatment": "immediate fungicide application, possibly multiple treatments",
                        "expected_yield_impact": "severe yield losses of 30-50% if untreated"
                    }
                }
            },
            "Healthy": {
                "symptoms": "normal green coloration, no visible lesions or abnormalities",
                "treatment": "maintain current management practices",
                "prevention": "continue balanced nutrition and regular monitoring",
                "impact": "not applicable",
                "similar_diseases": "early stages of nutrient deficiency may appear similar",
                "distinguishing_features": "uniform green color, absence of lesions or discoloration",
                "severity_levels": {
                    "optimal": {
                        "description": "Dark green leaves with no signs of stress or disease",
                        "treatment": "maintain current management practices",
                        "expected_yield_impact": "maximum yield potential for variety and conditions"
                    }
                }
            }
        }
    
    def get_disease_info(self, disease_name):
        """
        Get information for a specific disease.
        
        Args:
            disease_name (str): Name of the disease
            
        Returns:
            dict: Disease information or None if not found
        """
        # Try exact match first
        if disease_name in self.disease_info:
            return self.disease_info[disease_name]
        
        # Try case-insensitive match
        for name in self.disease_info:
            if name.lower() == disease_name.lower():
                return self.disease_info[name]
        
        # Try partial match
        for name in self.disease_info:
            if disease_name.lower() in name.lower() or name.lower() in disease_name.lower():
                return self.disease_info[name]
        
        return None
    
    def get_treatment_for_severity(self, disease_name, severity):
        """
        Get treatment recommendations based on disease severity.
        
        Args:
            disease_name (str): Name of the disease
            severity (str): Severity level ('low', 'medium', 'high')
            
        Returns:
            dict: Severity-specific information or None if not found
        """
        disease_info = self.get_disease_info(disease_name)
        
        if disease_info and 'severity_levels' in disease_info:
            severity_levels = disease_info['severity_levels']
            
            # Try exact match first
            if severity in severity_levels:
                return severity_levels[severity]
            
            # Default to medium severity if not found
            if 'medium' in severity_levels:
                return severity_levels['medium']
            
            # Return first severity level as fallback
            return next(iter(severity_levels.values()))
        
        return None
    
    def get_all_diseases(self):
        """
        Get a list of all diseases in the knowledge base.
        
        Returns:
            list: List of disease names
        """
        return list(self.disease_info.keys())


if __name__ == "__main__":
    # Example usage
    kb = PaddyDiseaseKnowledgeBase()
    
    # Get all diseases
    diseases = kb.get_all_diseases()
    print(f"Available diseases: {diseases}\n")
    
    # Get info for a specific disease
    disease = "Bacterial Leaf Blight"
    info = kb.get_disease_info(disease)
    
    if info:
        print(f"Disease: {disease}")
        print(f"Symptoms: {info['symptoms']}")
        print(f"Treatment: {info['treatment']}")
        print(f"Prevention: {info['prevention']}")
        print(f"Impact: {info['impact']}")
        print(f"Similar to: {info['similar_diseases']}")
        print(f"Distinguishing features: {info['distinguishing_features']}\n")
        
        # Get severity-specific treatment
        severity = "medium"
        severity_info = kb.get_treatment_for_severity(disease, severity)
        
        if severity_info:
            print(f"Severity: {severity}")
            print(f"Description: {severity_info['description']}")
            print(f"Treatment: {severity_info['treatment']}")
            print(f"Expected impact: {severity_info['expected_yield_impact']}")
    else:
        print(f"Disease '{disease}' not found in knowledge base.")
