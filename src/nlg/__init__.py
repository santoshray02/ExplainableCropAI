#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Natural Language Generation (NLG) module for ExplainableCropAI.

This module provides natural language explanation generators:
- NLExplanationGenerator: Template-based explanation generator
- ExplanationWithVisualEvidence: Combines visual features with NL explanations
- LLMExplanationGenerator: LLM-based explanation generator using Phi-3
- ExplanationWithLLM: Wrapper combining visual and LLM explanations
"""

from .template_generator import NLExplanationGenerator, ExplanationWithVisualEvidence

# Import LLM generator (optional, may require additional dependencies)
try:
    from .llm_generator import (
        LLMExplanationGenerator,
        ExplanationWithLLM,
        AudienceLevel,
        ExplanationContext
    )
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

__all__ = [
    'NLExplanationGenerator',
    'ExplanationWithVisualEvidence',
    'LLMExplanationGenerator',
    'ExplanationWithLLM',
    'AudienceLevel',
    'LLM_AVAILABLE'
]
