"""
Qwen LLM Client for PGA.

This module provides a concrete implementation of the LLM client using Qwen 0.5B (or similar)
via the transformers library. It handles model loading, prompt engineering for structured
extraction, and parsing of the model's output.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Import from the local copied files
from models import Observation

logger = logging.getLogger("pga.qwen_client")

class QwenLLMClient:
    """
    LLM client using Qwen-0.5B-Chat (or similar) for observation extraction.
    """

    def __init__(self, model_name: str = "Qwen/Qwen1.5-0.5B-Chat", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        
        logger.info(f"Loading model {model_name} on {device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                trust_remote_code=True,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self.model.eval()
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    async def extract_observations(self, text: str) -> list[Observation]:
        """
        Extract structured observations from text using Qwen.
        """
        prompt = self._build_extraction_prompt(text)
        
        try:
            response_text = self._generate(prompt)
            observations = self._parse_response(response_text)
            return observations
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return []

    def _build_extraction_prompt(self, text: str) -> str:
        """Construct the prompt for the model."""
        return f"""<|im_start|>system
You are a scientific observation extractor. Your task is to extract "Pure Observations" from the given text.
An observation has:
- name: (str) canonical label (e.g., 'mass', 'temperature')
- value: (float) numeric value
- unit: (str) SI unit or 'dimensionless'
- certainty: (float) 0.0 to 1.0 (1.0 = fact, 0.0 = speculation)

Output valid JSON only. Return a list of objects.
Example:
Input: "The apple weighs 150 grams and is red."
Output: [
    {{"name": "mass", "value": 0.15, "unit": "kg", "certainty": 1.0}},
    {{"name": "color_wavelength", "value": 650, "unit": "nm", "certainty": 0.8}}
]
<|im_end|>
<|im_start|>user
Text: "{text}"
<|im_end|>
<|im_start|>assistant
"""

    def _generate(self, prompt: str) -> str:
        """Run inference."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Simple generation config
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,  # Deterministic for extraction
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()

    def _parse_response(self, response: str) -> list[Observation]:
        """Parse JSON from the response."""
        # Find the first '[' and last ']'
        try:
            start_idx = response.find('[')
            end_idx = response.rfind(']')
            
            if start_idx == -1 or end_idx == -1:
                logger.warning(f"No JSON list found in response: {response}")
                return []
                
            json_str = response[start_idx : end_idx + 1]
            data = json.loads(json_str)
            
            observations = []
            for item in data:
                try:
                    obs = Observation(
                        name=item.get("name", "unknown"),
                        value=float(item.get("value", 0.0)),
                        unit=item.get("unit", "dimensionless"),
                        certainty=float(item.get("certainty", 0.5)),
                        source="qwen_extracted"
                    )
                    observations.append(obs)
                except Exception as e:
                    logger.warning(f"Skipping invalid item {item}: {e}")
                    
            return observations
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}. Response: {response}")
            return []
