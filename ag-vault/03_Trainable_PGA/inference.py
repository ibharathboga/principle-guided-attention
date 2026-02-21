import torch
import torch.nn as nn
from src.model import PGAModel

# Re-use vocabulary from training
vocab = {
    "PAD": 0, "design": 1, "bridge": 2, "heavy": 3, "load": 4, 
    "write": 5, "poem": 6, "love": 7, "beauty": 8, "structure": 9
}
inv_vocab = {v: k for k, v in vocab.items()}

def run_inference():
    print("--- Loading PGA Model for Inference ---")
    
    # 1. Initialize Model Architecture matches training
    vocab_size = len(vocab)
    d_model = 16 
    model = PGAModel(vocab_size, d_model)
    
    # 2. Load Weights
    try:
        model.load_state_dict(torch.load("pga_model.pth"))
        model.eval() # Set to evaluation mode
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print("Error: 'pga_model.pth' not found. Run 'train.py' first.")
        return

    # 3. Define Test Input
    # Context: "design bridge structure" -> Expect "load" (ID 4)
    input_text = "design bridge structure"
    print(f"\n--- Input: '{input_text}' ---")
    
    tokens = [vocab[word] for word in input_text.split() if word in vocab]
    input_tensor = torch.tensor([tokens], dtype=torch.long) # (1, Seq)
    
    # 4. Run Forward Pass
    with torch.no_grad():
        logits, P, heatmap = model(input_tensor)
        
    # 5. Decode Output
    last_token_logits = logits[:, -1, :]
    predicted_id = torch.argmax(last_token_logits, dim=-1).item()
    predicted_word = inv_vocab.get(predicted_id, "<UNK>")
    
    print(f"Predicted Next Word: '{predicted_word}'")
    
    # 6. Analyze Principle Matrix
    print("\n--- Discovered Principle Matrix (P) ---")
    print("Analysis: The Trace represents the magnitude of transformation applied.")
    trace = torch.trace(P[0])
    print(f"Matrix Trace: {trace:.4f}")
    
    # Interpret P
    # If P is close to Identity, trace should be close to d_model (16)
    # If P amplifies concepts strongly, trace might be higher.
    if abs(trace - 16.0) > 1.0:
        print("Insight: The model applied a significant transformation to the attention space.")
    else:
        print("Insight: The model applied minimal transformation (near Identity).")

    print("\nUse this script to test the model with different inputs or analyze the P matrix further.")

if __name__ == "__main__":
    run_inference()
