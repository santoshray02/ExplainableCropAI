#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM-based natural language explanation generator for ExplainableCropAI.

This module generates context-aware natural language explanations using
Microsoft's Phi-3-mini-4k-instruct model from HuggingFace Transformers.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

import torch

# Template-based fallback
try:
    from nlg.template_generator import NLExplanationGenerator, ExplanationWithVisualEvidence
except ImportError:
    from .template_generator import NLExplanationGenerator, ExplanationWithVisualEvidence

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudienceLevel(Enum):
    """Target audience levels for explanations."""
    FARMER = "farmer"
    EXPERT = "expert"
    RESEARCHER = "researcher"


@dataclass
class ExplanationContext:
    """Context information for generating explanations."""
    disease_name: str
    confidence: float
    visual_features: List[str]
    uncertainty_entropy: float
    uncertainty_mutual_info: float
    class_agreement: float
    treatment: str
    prevention: str
    impact: str
    severity: str
    similar_diseases: str
    distinguishing_features: str
    affected_area_percentage: float = 0.0


class LLMExplanationGenerator:
    """
    LLM-based explanation generator using Phi-3-mini-4k-instruct.

    Generates context-aware, natural language explanations for paddy disease
    predictions by leveraging the reasoning capabilities of large language models.
    """

    # System prompts for different audience levels
    SYSTEM_PROMPTS = {
        AudienceLevel.FARMER: """You are an agricultural AI assistant helping farmers understand crop disease diagnoses.
Your explanations should be:
- Simple and easy to understand without technical jargon
- Actionable with clear treatment recommendations
- Reassuring but honest about the situation
- Focused on practical steps the farmer can take immediately
Use language that a person without scientific background can understand.""",

        AudienceLevel.EXPERT: """You are an agricultural AI assistant providing disease analysis to agricultural experts and extension officers.
Your explanations should be:
- Technically accurate with proper disease terminology
- Include confidence levels and uncertainty metrics interpretation
- Reference visual diagnostic features precisely
- Provide differential diagnosis considerations
- Include evidence-based treatment protocols
Balance technical precision with clear communication.""",

        AudienceLevel.RESEARCHER: """You are an agricultural AI assistant providing detailed analysis for researchers and scientists.
Your explanations should be:
- Highly technical with full statistical details
- Include all uncertainty quantification metrics
- Discuss model confidence and potential limitations
- Reference visual feature correlations with diagnostic criteria
- Provide comprehensive differential diagnosis
Include all relevant technical details for scientific documentation."""
    }

    # Agricultural domain-specific prompt templates
    EXPLANATION_TEMPLATE = """Based on the AI model's analysis of a paddy plant image, generate a comprehensive explanation.

## Diagnosis Information:
- **Predicted Disease**: {disease_name}
- **Confidence Level**: {confidence:.1%}
- **Model Certainty**: {certainty_level}

## Visual Evidence Detected:
{visual_features_text}

## Uncertainty Metrics:
- Prediction Entropy: {uncertainty_entropy:.4f} (lower is more certain)
- Epistemic Uncertainty: {uncertainty_mutual_info:.4f}
- Prediction Agreement: {class_agreement:.1%} across multiple evaluations

## Disease Context:
- **Typical Symptoms**: {symptoms}
- **Similar Conditions**: {similar_diseases}
- **Key Distinguishing Features**: {distinguishing_features}

## Treatment Information:
- **Recommended Treatment**: {treatment}
- **Prevention Measures**: {prevention}
- **Potential Impact if Untreated**: {impact}
- **Estimated Severity**: {severity}

## Task:
Generate a clear, informative explanation of this diagnosis for the target audience. Include:
1. A summary of the diagnosis and confidence level
2. Explanation of the visual evidence supporting this diagnosis
3. Assessment of the model's certainty and what it means
4. Specific treatment recommendations
5. Prevention advice for the future

Be concise but thorough. The explanation should help the user understand both WHAT the diagnosis is and WHY the model made this prediction."""

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_memory: Optional[Dict[int, str]] = None
    ):
        """
        Initialize the LLM explanation generator.

        Args:
            model_name: HuggingFace model identifier for Phi-3-mini
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
            load_in_8bit: Whether to load model in 8-bit quantization (saves memory)
            load_in_4bit: Whether to load model in 4-bit quantization (saves more memory)
            max_memory: Maximum memory to use per device (for multi-GPU setups)
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.is_loaded = False
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.max_memory = max_memory

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Fallback generator for when LLM fails
        self.fallback_generator = NLExplanationGenerator()

        logger.info(f"LLMExplanationGenerator initialized with model: {model_name}")
        logger.info(f"Device: {self.device}")

    def load_model(self) -> bool:
        """
        Load the LLM model and tokenizer.

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if self.is_loaded:
            logger.info("Model already loaded")
            return True

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            logger.info(f"Loading model: {self.model_name}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Configure model loading options
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }

            # Add quantization options if specified
            if self.load_in_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    model_kwargs["device_map"] = "auto"
                except ImportError:
                    logger.warning("bitsandbytes not available, loading without 4-bit quantization")
                    model_kwargs["device_map"] = "auto" if self.device == "cuda" else None
            elif self.load_in_8bit:
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    model_kwargs["quantization_config"] = quantization_config
                    model_kwargs["device_map"] = "auto"
                except ImportError:
                    logger.warning("bitsandbytes not available, loading without 8-bit quantization")
                    model_kwargs["device_map"] = "auto" if self.device == "cuda" else None
            else:
                model_kwargs["device_map"] = "auto" if self.device == "cuda" else None

            if self.max_memory:
                model_kwargs["max_memory"] = self.max_memory

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )

            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
            )

            self.is_loaded = True
            logger.info("Model loaded successfully")
            return True

        except ImportError as e:
            logger.error(f"Required library not installed: {e}")
            logger.error("Install with: pip install transformers torch accelerate")
            return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _build_prompt(
        self,
        context: ExplanationContext,
        audience: AudienceLevel = AudienceLevel.FARMER
    ) -> str:
        """
        Build the prompt for the LLM based on context and audience.

        Args:
            context: ExplanationContext with all relevant information
            audience: Target audience level for the explanation

        Returns:
            str: Formatted prompt for the LLM
        """
        # Format visual features as bullet points
        if context.visual_features:
            visual_features_text = "\n".join(
                f"- {self._format_visual_feature(f)}"
                for f in context.visual_features
            )
        else:
            visual_features_text = "- General disease symptoms visible in affected areas"

        # Determine certainty level description
        if context.confidence >= 0.85 and context.class_agreement >= 0.8:
            certainty_level = "HIGH - The model is very confident in this diagnosis"
        elif context.confidence >= 0.6 and context.class_agreement >= 0.5:
            certainty_level = "MODERATE - The model has reasonable confidence, but expert verification is recommended"
        else:
            certainty_level = "LOW - The model is uncertain; please consult an agricultural expert"

        # Get symptoms from treatment context
        symptoms = context.treatment.replace("apply", "").strip() if context.treatment else "visible disease symptoms"

        # Build the user prompt from template
        user_prompt = self.EXPLANATION_TEMPLATE.format(
            disease_name=context.disease_name,
            confidence=context.confidence,
            certainty_level=certainty_level,
            visual_features_text=visual_features_text,
            uncertainty_entropy=context.uncertainty_entropy,
            uncertainty_mutual_info=context.uncertainty_mutual_info,
            class_agreement=context.class_agreement,
            symptoms=symptoms,
            similar_diseases=context.similar_diseases or "None identified",
            distinguishing_features=context.distinguishing_features or "Standard disease presentation",
            treatment=context.treatment or "Consult local agricultural extension office",
            prevention=context.prevention or "Regular monitoring and good agricultural practices",
            impact=context.impact or "Potential yield reduction if untreated",
            severity=context.severity or "To be determined by expert assessment"
        )

        # Get system prompt for audience level
        system_prompt = self.SYSTEM_PROMPTS[audience]

        # Format for Phi-3 chat template
        # Phi-3 uses <|system|>, <|user|>, <|assistant|> tokens
        formatted_prompt = f"""<|system|>
{system_prompt}<|end|>
<|user|>
{user_prompt}<|end|>
<|assistant|>
"""

        return formatted_prompt

    def _format_visual_feature(self, feature: str) -> str:
        """
        Convert internal feature names to human-readable descriptions.

        Args:
            feature: Internal feature name

        Returns:
            str: Human-readable description
        """
        feature_descriptions = {
            'leaf_spots': 'Brown or yellow spots visible on leaves',
            'leaf_discoloration': 'Abnormal yellowing or browning of leaf tissue',
            'leaf_lesions': 'Elongated lesions or wounds on leaves',
            'stem_lesions': 'Dark lesions visible on stems',
            'white_powder': 'White powdery substance on leaf surfaces',
            'stunted_growth': 'Abnormally small or stunted plant growth',
            'wilting': 'Drooping or wilting of leaves and stems',
            'water_soaked': 'Water-soaked appearance in leaf tissue',
            'necrosis': 'Dead (necrotic) tissue areas',
            'chlorosis': 'Loss of green color (chlorosis)',
            'diamond_shaped': 'Diamond-shaped lesion pattern characteristic of blast',
            'wavy_margins': 'Lesions with wavy or irregular margins'
        }
        return feature_descriptions.get(feature, feature.replace('_', ' ').title())

    def _generate_response(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> Optional[str]:
        """
        Generate a response from the LLM.

        Args:
            prompt: The formatted prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling (False = greedy decoding)

        Returns:
            str: Generated response or None if generation failed
        """
        if not self.is_loaded:
            if not self.load_model():
                logger.error("Failed to load model for generation")
                return None

        try:
            # Generate response using pipeline
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )

            # Extract generated text
            generated_text = outputs[0]['generated_text']

            # Clean up the response
            generated_text = generated_text.strip()

            # Remove any trailing incomplete sentences
            if generated_text and not generated_text[-1] in '.!?':
                last_period = generated_text.rfind('.')
                if last_period > len(generated_text) // 2:
                    generated_text = generated_text[:last_period + 1]

            return generated_text

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None

    def generate_explanation(
        self,
        prediction_info: Dict[str, Any],
        visual_features: List[str],
        uncertainty_info: Dict[str, Any],
        disease_info: Optional[Dict[str, Any]] = None,
        audience: Union[AudienceLevel, str] = AudienceLevel.FARMER,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """
        Generate a natural language explanation for the disease prediction.

        Args:
            prediction_info: Dictionary containing:
                - class_name: Predicted disease name
                - confidence: Confidence score (0-1)
                - class_id: Predicted class index
            visual_features: List of detected visual features
            uncertainty_info: Dictionary containing:
                - entropy: Prediction entropy
                - mutual_information: Epistemic uncertainty
                - class_agreement: Agreement across MC samples
            disease_info: Optional dictionary with disease-specific information
            audience: Target audience level (AudienceLevel or string)
            max_tokens: Maximum tokens for LLM generation

        Returns:
            dict: Generated explanation with components
        """
        # Convert string audience to enum if necessary
        if isinstance(audience, str):
            try:
                audience = AudienceLevel(audience.lower())
            except ValueError:
                audience = AudienceLevel.FARMER

        # Extract information
        disease_name = prediction_info.get('class_name', 'Unknown Disease')
        confidence = prediction_info.get('confidence', 0.0)

        # Get uncertainty metrics with defaults
        entropy = uncertainty_info.get('entropy', 0.5)
        mutual_info = uncertainty_info.get('mutual_information', 0.3)
        class_agreement = uncertainty_info.get('class_agreement', 0.7)

        # Get disease info with defaults
        if disease_info is None:
            disease_info = {}

        # Build context
        context = ExplanationContext(
            disease_name=disease_name,
            confidence=confidence,
            visual_features=visual_features if visual_features else [],
            uncertainty_entropy=entropy,
            uncertainty_mutual_info=mutual_info,
            class_agreement=class_agreement,
            treatment=disease_info.get('treatment', ''),
            prevention=disease_info.get('prevention', ''),
            impact=disease_info.get('impact', ''),
            severity=self._determine_severity(confidence, entropy),
            similar_diseases=disease_info.get('similar_diseases', ''),
            distinguishing_features=disease_info.get('distinguishing_features', ''),
            affected_area_percentage=prediction_info.get('affected_area_percentage', 0.0)
        )

        # Build prompt
        prompt = self._build_prompt(context, audience)

        # Try LLM generation
        llm_explanation = self._generate_response(prompt, max_tokens=max_tokens)

        if llm_explanation:
            # Successful LLM generation
            return {
                'full_explanation': llm_explanation,
                'source': 'llm',
                'model': self.model_name,
                'audience': audience.value,
                'confidence_level': self._get_confidence_level(confidence),
                'certainty_assessment': self._assess_certainty(confidence, class_agreement),
                'components': {
                    'diagnosis': self._extract_diagnosis_section(llm_explanation),
                    'evidence': self._extract_evidence_section(llm_explanation),
                    'treatment': disease_info.get('treatment', ''),
                    'prevention': disease_info.get('prevention', '')
                },
                'metadata': {
                    'entropy': entropy,
                    'mutual_information': mutual_info,
                    'class_agreement': class_agreement
                }
            }
        else:
            # Fallback to template-based generation
            logger.warning("LLM generation failed, using template fallback")
            return self._fallback_generation(
                prediction_info, visual_features, disease_info
            )

    def _determine_severity(self, confidence: float, entropy: float) -> str:
        """Determine severity level based on confidence and uncertainty."""
        if confidence >= 0.8 and entropy < 0.5:
            return "High confidence detection - immediate attention recommended"
        elif confidence >= 0.5:
            return "Moderate confidence - verification advised"
        else:
            return "Low confidence - expert consultation recommended"

    def _get_confidence_level(self, confidence: float) -> str:
        """Get human-readable confidence level."""
        if confidence >= 0.85:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        else:
            return "low"

    def _assess_certainty(self, confidence: float, class_agreement: float) -> str:
        """Assess overall certainty of the prediction."""
        if confidence >= 0.85 and class_agreement >= 0.8:
            return "The model is highly certain about this diagnosis."
        elif confidence >= 0.6 and class_agreement >= 0.5:
            return "The model has reasonable certainty, but expert verification is recommended."
        else:
            return "The model has significant uncertainty. Please consult an agricultural expert."

    def _extract_diagnosis_section(self, text: str) -> str:
        """Extract the diagnosis section from generated text."""
        # Simple extraction - in practice, could use more sophisticated parsing
        lines = text.split('\n')
        diagnosis_lines = []
        for line in lines[:5]:  # Usually diagnosis is in first few lines
            if line.strip():
                diagnosis_lines.append(line.strip())
        return ' '.join(diagnosis_lines[:2]) if diagnosis_lines else text[:200]

    def _extract_evidence_section(self, text: str) -> str:
        """Extract the evidence section from generated text."""
        # Look for visual evidence keywords
        keywords = ['visual', 'evidence', 'shows', 'displays', 'exhibits', 'features']
        lines = text.split('.')
        evidence_lines = []
        for line in lines:
            if any(kw in line.lower() for kw in keywords):
                evidence_lines.append(line.strip())
        return '. '.join(evidence_lines[:2]) if evidence_lines else ""

    def _fallback_generation(
        self,
        prediction_info: Dict[str, Any],
        visual_features: List[str],
        disease_info: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Fallback to template-based generation when LLM fails.

        Args:
            prediction_info: Prediction information
            visual_features: Visual features detected
            disease_info: Disease information

        Returns:
            dict: Template-generated explanation
        """
        # Use the existing template generator
        visual_explainer = ExplanationWithVisualEvidence(self.fallback_generator)

        explanation = visual_explainer.generate_explanation(
            prediction_info=prediction_info,
            visual_features=visual_features,
            disease_info=disease_info
        )

        # Add metadata indicating fallback
        explanation['source'] = 'template'
        explanation['model'] = 'template-based'
        explanation['llm_fallback_reason'] = 'LLM generation failed or unavailable'

        return explanation


class ExplanationWithLLM:
    """
    Wrapper class combining visual explanations with LLM-generated text.

    This class integrates the visual explanation pipeline with the LLM
    generator to produce comprehensive, context-aware explanations.
    """

    def __init__(
        self,
        llm_generator: Optional[LLMExplanationGenerator] = None,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        device: Optional[str] = None,
        preload_model: bool = False
    ):
        """
        Initialize the combined explanation generator.

        Args:
            llm_generator: Optional pre-configured LLM generator
            model_name: HuggingFace model name if creating new generator
            device: Device for model inference
            preload_model: Whether to load the model immediately
        """
        if llm_generator is not None:
            self.llm_generator = llm_generator
        else:
            self.llm_generator = LLMExplanationGenerator(
                model_name=model_name,
                device=device
            )

        # Visual feature to textual description mapping (extended)
        self.visual_feature_mapping = {
            'leaf_spots': 'brown or yellow spots on leaves',
            'leaf_discoloration': 'yellowing or browning of leaves',
            'leaf_lesions': 'elongated lesions on leaves',
            'stem_lesions': 'dark lesions on stems',
            'white_powder': 'white powdery substance on leaves',
            'stunted_growth': 'abnormally small or stunted growth',
            'wilting': 'drooping or wilting of leaves',
            'water_soaked': 'water-soaked lesion appearance',
            'necrosis': 'dead tissue areas',
            'chlorosis': 'yellowing due to chlorophyll loss',
            'diamond_shaped': 'diamond-shaped lesions typical of blast',
            'wavy_margins': 'lesions with wavy margins'
        }

        if preload_model:
            self.llm_generator.load_model()

    def generate_visual_evidence_description(self, visual_features: List[str]) -> str:
        """
        Generate textual description of visual evidence.

        Args:
            visual_features: List of identified visual features

        Returns:
            str: Human-readable description of visual evidence
        """
        if not visual_features:
            return "visible symptoms in the affected areas"

        descriptions = [
            self.visual_feature_mapping.get(feature, feature.replace('_', ' '))
            for feature in visual_features
        ]

        if len(descriptions) == 1:
            return descriptions[0]
        elif len(descriptions) == 2:
            return f"{descriptions[0]} and {descriptions[1]}"
        else:
            return ", ".join(descriptions[:-1]) + f", and {descriptions[-1]}"

    def generate_explanation(
        self,
        prediction_info: Dict[str, Any],
        visual_features: List[str],
        uncertainty_info: Dict[str, Any],
        disease_info: Optional[Dict[str, Any]] = None,
        audience: Union[AudienceLevel, str] = AudienceLevel.FARMER,
        include_visual_description: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive explanation combining visual and LLM components.

        Args:
            prediction_info: Prediction information from the model
            visual_features: List of detected visual features
            uncertainty_info: Uncertainty quantification results
            disease_info: Disease-specific knowledge
            audience: Target audience for the explanation
            include_visual_description: Whether to include visual evidence description

        Returns:
            dict: Comprehensive explanation combining all components
        """
        # Generate visual evidence description
        visual_evidence = self.generate_visual_evidence_description(visual_features)

        # Generate LLM explanation
        llm_result = self.llm_generator.generate_explanation(
            prediction_info=prediction_info,
            visual_features=visual_features,
            uncertainty_info=uncertainty_info,
            disease_info=disease_info,
            audience=audience
        )

        # Combine results
        combined_result = {
            'full_explanation': llm_result['full_explanation'],
            'visual_evidence': visual_evidence,
            'visual_features': visual_features,
            'source': llm_result.get('source', 'unknown'),
            'model': llm_result.get('model', 'unknown'),
            'audience': audience.value if isinstance(audience, AudienceLevel) else audience,
            'components': llm_result.get('components', {}),
            'confidence_level': llm_result.get('confidence_level', 'unknown'),
            'certainty_assessment': llm_result.get('certainty_assessment', ''),
            'metadata': llm_result.get('metadata', {})
        }

        # Add visual evidence to components
        combined_result['components']['visual_evidence'] = visual_evidence

        return combined_result

    def generate_multi_audience_explanation(
        self,
        prediction_info: Dict[str, Any],
        visual_features: List[str],
        uncertainty_info: Dict[str, Any],
        disease_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate explanations for all audience levels.

        Args:
            prediction_info: Prediction information
            visual_features: Detected visual features
            uncertainty_info: Uncertainty metrics
            disease_info: Disease knowledge

        Returns:
            dict: Explanations keyed by audience level
        """
        explanations = {}

        for audience in AudienceLevel:
            explanations[audience.value] = self.generate_explanation(
                prediction_info=prediction_info,
                visual_features=visual_features,
                uncertainty_info=uncertainty_info,
                disease_info=disease_info,
                audience=audience
            )

        return explanations


# Example usage and testing
if __name__ == "__main__":
    # Sample prediction data
    prediction_info = {
        'class_name': 'Bacterial Leaf Blight',
        'confidence': 0.87,
        'class_id': 0
    }

    # Sample visual features
    visual_features = ['leaf_lesions', 'leaf_discoloration', 'wavy_margins']

    # Sample uncertainty info
    uncertainty_info = {
        'entropy': 0.32,
        'mutual_information': 0.15,
        'class_agreement': 0.83,
        'predictive_variance': 0.08
    }

    # Sample disease info
    disease_info = {
        'symptoms': 'water-soaked lesions that turn yellow-orange with wavy margins',
        'treatment': 'apply copper-based bactericides and ensure good field drainage',
        'prevention': 'use resistant varieties, practice crop rotation, and avoid overhead irrigation',
        'impact': 'cause yield losses of up to 70% in severely affected fields',
        'similar_diseases': 'Bacterial Leaf Streak',
        'distinguishing_features': 'wavy margins of lesions instead of straight edges'
    }

    print("=" * 80)
    print("LLM-based Explanation Generator Test")
    print("=" * 80)

    # Test without loading model (will use fallback)
    print("\n--- Testing Fallback Mode ---")
    generator = LLMExplanationGenerator()

    # Generate explanation (will use fallback since model is not loaded)
    explanation = generator.generate_explanation(
        prediction_info=prediction_info,
        visual_features=visual_features,
        uncertainty_info=uncertainty_info,
        disease_info=disease_info,
        audience=AudienceLevel.FARMER
    )

    print(f"\nSource: {explanation['source']}")
    print(f"\nExplanation:\n{explanation['full_explanation']}")

    # Test the wrapper class
    print("\n--- Testing ExplanationWithLLM Wrapper ---")
    wrapper = ExplanationWithLLM(preload_model=False)

    combined = wrapper.generate_explanation(
        prediction_info=prediction_info,
        visual_features=visual_features,
        uncertainty_info=uncertainty_info,
        disease_info=disease_info,
        audience='farmer'
    )

    print(f"\nVisual Evidence: {combined['visual_evidence']}")
    print(f"Source: {combined['source']}")
    print(f"Confidence Level: {combined['confidence_level']}")

    print("\n--- Test Complete ---")
