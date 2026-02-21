import sys
import os

# Ensure src can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pga_model import PGAModel

def run_demo():
    print("==============================================")
    print("   Principle-Guided Attention (PGA) Demo      ")
    print("==============================================")
    print("Objective: Demonstrate dynamic weight modification based on discovered principles.\n")
    
    # Initialize Model
    model = PGAModel(embedding_dim=8)
    
    # Test Case 1: The "Bridge" Scenario
    # Expectation: "Beauty" (Dim 0) suppressed, "Physics" (Dim 1) amplified.
    input_text_1 = "Design a detailed bridge for heavy traffic."
    E1 = model.run(input_text_1)

    # Test Case 2: A Control Scenario
    # Expectation: Standard weights (Identity Principle)
    input_text_2 = "Write a poem about the sunrise."
    E2 = model.run(input_text_2)
    
    print("=== Demo Complete ===")

if __name__ == "__main__":
    run_demo()
