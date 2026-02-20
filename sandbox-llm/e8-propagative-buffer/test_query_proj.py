
import torch
import sys
import os

# Add local directory to path
sys.path.insert(0, os.path.dirname(__file__))

from models import PropagativePGABuffer, N_EMBD, BLOCK_SIZE

def test_pga_buffer_projection():
    print("Initializing PropagativePGABuffer...")
    vocab_size = 65
    model = PropagativePGABuffer(vocab_size)
    
    # Check if query_proj exists
    if hasattr(model, 'query_proj'):
        print("✅ model.query_proj exists")
        print(f"   Shape: {model.query_proj.weight.shape}")
    else:
        print("❌ model.query_proj MISSING")
        return

    # Create dummy input
    batch_size = 2
    idx = torch.randint(0, vocab_size, (batch_size, BLOCK_SIZE))
    
    print("Running forward pass...")
    try:
        logits, loss = model(idx)
        print("✅ Forward pass successful")
        print(f"   Logits shape: {logits.shape}")
    except Exception as e:
        print(f"❌ Forward pass FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pga_buffer_projection()
