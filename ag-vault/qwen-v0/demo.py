"""
Demo: Qwen Integration for PGA (Proof of Concept).

This script demonstrates:
1. Loading the text generation model.
2. Extracting structured observations from unstructured text.
3. Evaluating the clarity and factuality of a claim using the QwenMetric.
"""

import asyncio
import logging
import sys

# Ensure parent directory is in path
sys.path.insert(0, ".")

from llm_client import QwenLLMClient
from metrics import QwenMetric

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("demo")

async def main():
    print("=" * 60)
    print("  PGA Qwen Integration Demo")
    print("=" * 60)

    # 1. Initialize Client
    try:
        # Using a smaller model for demo if possible, or the one specified
        # Note: This requires a GPU or decent CPU RAM.
        client = QwenLLMClient(model_name="Qwen/Qwen1.5-0.5B-Chat", device="cpu")
        metric = QwenMetric(client)
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return

    # 2. Observation Extraction
    text = "The new battery has a capacity of 5000 mAh and weighs 180 grams."
    print(f"\n[Observation Extraction]\nInput: '{text}'")
    
    observations = client.extract_observations(text) # Note: extract_observations is async in my previous code? No, I made it sync in the helper but async wrapper
    # Wait, in llm_client.py I defined it as `async def extract_observations`.
    # But `_generate` is sync.
    # Let's check llm_client.py content from my previous turn.
    # Yes: `async def extract_observations(self, text: str) -> list[Observation]:`
    
    observations = await client.extract_observations(text)
    
    for obs in observations:
        print(f"  - {obs.name}: {obs.value} {obs.unit} (certainty={obs.certainty})")

    # 3. Metric Evaluation
    print(f"\n[Metric Evaluation]")
    
    # Clarity
    unclear_text = "The thing does the stuff with the energy."
    clear_text = "The photovoltaic cell converts solar energy into electricity with 20% efficiency."
    
    score_unclear = metric.evaluate_clarity(unclear_text)
    score_clear = metric.evaluate_clarity(clear_text)
    
    print(f"  Clarity ('{unclear_text}'): {score_unclear}")
    print(f"  Clarity ('{clear_text}'): {score_clear}")

    # Factuality
    context = "The speed of light in vacuum is approximately 299,792,458 meters per second."
    claim_true = "Light travels at about 300,000 km/s."
    claim_false = "Light travels at 100 meters per hour."
    
    score_true = metric.evaluate_factuality(claim_true, context)
    score_false = metric.evaluate_factuality(claim_false, context)
    
    print(f"  Factuality ('{claim_true}'): {score_true}")
    print(f"  Factuality ('{claim_false}'): {score_false}")

    print("\nDone.")

if __name__ == "__main__":
    asyncio.run(main())
