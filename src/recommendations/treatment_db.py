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
            # NOTE: Bacterial Panicle Blight (Burkholderia glumae/gladioli). There is no
            # fully effective curative chemical control; guidance below is conservative and
            # referral-based. Verify region-specific options with a plant pathologist /
            # agricultural extension service before recommending any chemical treatment.
            "Bacterial Panicle Blight": {
                "symptoms": "discoloration of florets and grains on the panicle (light to dark brown), grain sterility and unfilled grains; affected panicles often remain upright, and seedling rot can occur",
                "treatment": "no fully effective curative chemical control; use certified disease-free seed, avoid excessive nitrogen, and consult local agricultural extension for region-approved management",
                "prevention": "use certified disease-free seed and tolerant varieties where available, apply balanced nitrogen (avoid excess), and adjust planting time to avoid high temperatures during flowering",
                "impact": "grain sterility and unfilled grains reduce yield and grain quality; losses are most severe under hot, humid conditions",
                "similar_diseases": "other grain-discoloration disorders",
                "distinguishing_features": "discoloration concentrated on grains and panicles with sterility, rather than on leaf sheaths or leaves",
                "severity_levels": {
                    "low": {
                        "description": "Few panicles with limited grain discoloration",
                        "treatment": "monitor the crop and consult extension; ensure balanced nitrogen and clean seed",
                        "expected_yield_impact": "minor if conditions do not favour spread"
                    },
                    "medium": {
                        "description": "Several panicles affected with noticeable grain sterility",
                        "treatment": "consult local agricultural extension for region-approved management; avoid excess nitrogen",
                        "expected_yield_impact": "variable yield reduction; depends on weather and variety"
                    },
                    "high": {
                        "description": "Widespread panicle discoloration and grain sterility across the field",
                        "treatment": "seek expert/extension guidance; prioritise clean seed and tolerant varieties for the next season",
                        "expected_yield_impact": "substantial losses under prolonged hot, humid conditions"
                    }
                }
            },
            "Tungro": {
                "symptoms": "yellow to orange discoloration of leaves, stunted growth, reduced tillering",
                "treatment": "no direct treatment available; control leafhopper vectors with insecticides",
                "prevention": "use resistant varieties, synchronous planting, control green leafhoppers",
                "impact": "cause severe yield losses up to 100% in susceptible varieties",
                "similar_diseases": "nutrient deficiency, other viral diseases",
                "distinguishing_features": "interveinal yellowing with stunted growth pattern",
                "severity_levels": {
                    "low": {
                        "description": "Few plants showing mild yellowing symptoms",
                        "treatment": "control leafhopper population, remove infected plants",
                        "expected_yield_impact": "10-20% if vector population is controlled"
                    },
                    "medium": {
                        "description": "Multiple plants affected with clear stunting",
                        "treatment": "intensive vector control, consider replanting if early in season",
                        "expected_yield_impact": "potential yield reduction of 30-50%"
                    },
                    "high": {
                        "description": "Widespread infection across field with severe stunting",
                        "treatment": "focus on vector control for next season, salvage what possible",
                        "expected_yield_impact": "severe yield losses of 50-100%"
                    }
                }
            },
            "Hispa": {
                "symptoms": "white parallel streaks on leaves, scraping damage, tunneling in leaves",
                "treatment": "apply contact insecticides like chlorpyrifos or lambda-cyhalothrin",
                "prevention": "avoid close planting, remove grassy weeds, use balanced fertilization",
                "impact": "reduce photosynthetic area leading to 10-30% yield loss",
                "similar_diseases": "leaf miner damage",
                "distinguishing_features": "characteristic parallel white streaks with visible beetles",
                "severity_levels": {
                    "low": {
                        "description": "Few leaves with scraping marks, low beetle population",
                        "treatment": "handpick beetles, maintain field hygiene",
                        "expected_yield_impact": "minimal if controlled early"
                    },
                    "medium": {
                        "description": "Multiple plants affected with visible beetle population",
                        "treatment": "apply recommended insecticides at proper intervals",
                        "expected_yield_impact": "potential yield reduction of 10-20%"
                    },
                    "high": {
                        "description": "Severe infestation with extensive leaf damage",
                        "treatment": "immediate insecticide application, may need repeated treatments",
                        "expected_yield_impact": "yield losses of 20-30% or more"
                    }
                }
            },
            "Dead Heart": {
                "symptoms": "central shoot dies and turns brown, can be easily pulled out",
                "treatment": "apply systemic insecticides targeting stem borers",
                "prevention": "destroy stubbles, use pheromone traps, timely planting",
                "impact": "direct tiller loss leading to 20-70% yield reduction",
                "similar_diseases": "stem rot, drought stress",
                "distinguishing_features": "dead central whorl that pulls out easily, bore holes visible",
                "severity_levels": {
                    "low": {
                        "description": "Less than 5% tillers affected",
                        "treatment": "apply granular insecticides, set up pheromone traps",
                        "expected_yield_impact": "5-10% with timely intervention"
                    },
                    "medium": {
                        "description": "5-15% tillers showing dead heart symptoms",
                        "treatment": "spray systemic insecticides, increase trap density",
                        "expected_yield_impact": "potential yield reduction of 15-30%"
                    },
                    "high": {
                        "description": "More than 15% tillers affected",
                        "treatment": "intensive insecticide program, consider economic threshold",
                        "expected_yield_impact": "severe yield losses of 30-70%"
                    }
                }
            },
            "Downy Mildew": {
                "symptoms": "white to gray downy growth on leaf undersides, yellow patches on upper surface",
                "treatment": "apply metalaxyl-based fungicides or copper-based products",
                "prevention": "use resistant varieties, improve air circulation, avoid overhead irrigation",
                "impact": "reduce photosynthesis and cause 15-40% yield loss",
                "similar_diseases": "powdery mildew, other foliar diseases",
                "distinguishing_features": "downy growth on leaf underside visible in humid conditions",
                "severity_levels": {
                    "low": {
                        "description": "Few leaves with light downy patches",
                        "treatment": "improve air circulation, apply preventive fungicides",
                        "expected_yield_impact": "minimal if environmental conditions change"
                    },
                    "medium": {
                        "description": "Multiple leaves affected with spreading patches",
                        "treatment": "apply systemic fungicides promptly",
                        "expected_yield_impact": "potential yield reduction of 15-25%"
                    },
                    "high": {
                        "description": "Widespread infection with leaf drying",
                        "treatment": "aggressive fungicide application with repeated treatments",
                        "expected_yield_impact": "yield losses of 25-40%"
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
