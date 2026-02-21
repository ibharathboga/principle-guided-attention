"""
Qwen-based Metrics for PGA.

This module implements metrics to evaluate the quality of text using Qwen.
It extracts a "Clarity Score" and checks for "Hallucination" (consistency).
"""

from __future__ import annotations

import logging
import torch
from llm_client import QwenLLMClient

logger = logging.getLogger("pga.metrics")

class QwenMetric:
    """
    Evaluates text quality using Qwen.
    """

    def __init__(self, client: QwenLLMClient):
        self.client = client

    def evaluate_clarity(self, text: str) -> float:
        """
        Score the clarity of the text from 0.0 to 1.0.
        
        Asks the model to rate the text's clarity, precision, and lack of ambiguity.
        """
        prompt = f"""<|im_start|>system
You are a strict editor. valid output is a single float number between 0.0 and 1.0.
Rate the CLARITY of the following text. 
1.0 = Perfectly clear, precise, unambiguous.
0.0 = Incoherent, vague, confusing.
Output ONLY the number.
<|im_end|>
<|im_start|>user
Text: "{text}"
<|im_end|>
<|im_start|>assistant
"""
        response = self.client._generate(prompt)
        return self._extract_float(response)

    def evaluate_factuality(self, claim: str, context: str) -> float:
        """
        Score the factuality/consistency of the claim given the context.
        0.0 to 1.0.
        """
        prompt = f"""<|im_start|>system
You are a fact-checker. valid output is a single float number between 0.0 and 1.0.
Rate the CONSISTENCY of the claim with the provided context.
1.0 = Fully supported by context.
0.0 = Contradicts context or unsupported.
Output ONLY the number.
<|im_end|>
<|im_start|>user
Context: "{context}"
Claim: "{claim}"
<|im_end|>
<|im_start|>assistant
"""
        response = self.client._generate(prompt)
        return self._extract_float(response)

    def _extract_float(self, text: str) -> float:
        """Helper to extract the first float from a string."""
        import re
        match = re.search(r"[-+]?\d*\.\d+|\d+", text)
        if match:
            try:
                val = float(match.group())
                return max(0.0, min(1.0, val))
            except ValueError:
                pass
        return 0.5  # Fallback
