import torch
import torch.nn as nn
import random
import math
import numpy as np
import matplotlib.pyplot as plt

# Import models
import microgpt_pytorch
import microgpt_original

# Setup
random.seed(42)
torch.manual_seed(42)

def copy_weights_pytorch_to_microgpt(pt_model, mg_state_dict):
    """
    Copies weights from PyTorch model to MicroGPT Value objects.
    """
    # 1. Embeddings
    # wte: (vocab, n_embd)
    for i in range(microgpt_original.vocab_size):
        for j in range(microgpt_original.n_embd):
            mg_state_dict["wte"][i][j].data = pt_model.wte.weight[i, j].item()
            
    # wpe: (block_size, n_embd)
    for i in range(microgpt_original.block_size):
        for j in range(microgpt_original.n_embd):
            mg_state_dict["wpe"][i][j].data = pt_model.wpe.weight[i, j].item()
            
    # lm_head: (vocab, n_embd) -> Linear.weight is (out_features, in_features) -> (vocab, n_embd)
    for i in range(microgpt_original.vocab_size):
        for j in range(microgpt_original.n_embd):
            mg_state_dict["lm_head"][i][j].data = pt_model.lm_head.weight[i, j].item()
            
    # Layers
    for l in range(microgpt_original.n_layer):
        prefix = f"layer{l}"
        pt_layer = pt_model.layers[l]
        
        # Linear layers: PyTorch weights are (out, in). MicroGPT weights are (out, in) lists.
        # Check microgpt init: matrix(nout, nin) -> list of 'nout' lists, each length 'nin'.
        # So structure matches.
        
        targets = [
            ("attn_wq", pt_layer["attn_wq"]),
            ("attn_wk", pt_layer["attn_wk"]),
            ("attn_wv", pt_layer["attn_wv"]),
            ("attn_wo", pt_layer["attn_wo"]),
            ("mlp_fc1", pt_layer["mlp_fc1"]),
            ("mlp_fc2", pt_layer["mlp_fc2"]),
        ]
        
        for name, pt_module in targets:
            mg_matrix = mg_state_dict[f"{prefix}.{name}"]
            pt_weight = pt_module.weight # (out, in)
            rows, cols = pt_weight.shape
            
            for r in range(rows):
                for c in range(cols):
                    mg_matrix[r][c].data = pt_weight[r, c].item()

def run_comparison(steps=50):
    print("Initializing models...")
    
    # PyTorch Model
    pt_model = microgpt_pytorch.MicroGPT()
    pt_optimizer = torch.optim.Adam(pt_model.parameters(), lr=0.01, betas=(0.85, 0.99), eps=1e-8)
    
    # MicroGPT Model
    mg_state_dict = microgpt_original.init_state() # Random init
    
    # SYNC WEIGHTS (PyTorch -> MicroGPT)
    print("Syncing weights...")
    copy_weights_pytorch_to_microgpt(pt_model, mg_state_dict)
    
    # Optimizer State for MicroGPT
    mg_params = [p for mat in mg_state_dict.values() for row in mat for p in row]
    mg_m = [0.0] * len(mg_params)
    mg_v = [0.0] * len(mg_params)
    
    # Dummy Data Generator (Deterministic)
    # create a fixed sequence of tokens
    vocab = list(range(microgpt_original.vocab_size))
    data = []
    for i in range(steps * 20): # enough data
        data.append(vocab[i % len(vocab)])
        
    losses_pt = []
    losses_mg = []
    
    print(f"Running comparison for {steps} steps...")
    
    for step in range(steps):
        # Prepare Batch
        # MicroGPT takes 1 document. Let's create a "document" of length block_size + 1
        # for a single training step on one sequence.
        
        # To match exactly, we run MicroGPT on *one sequence* (length n).
        # And PyTorch on *batch size 1*, same sequence.
        
        start_idx = step % (len(data) - microgpt_original.block_size - 1)
        chunk = data[start_idx : start_idx + microgpt_original.block_size + 1]
        
        x_chunk = chunk[:-1]
        y_chunk = chunk[1:]
        
        # --- PyTorch Forward ---
        pt_model.train()
        pt_optimizer.zero_grad()
        
        x_pt = torch.tensor([x_chunk], dtype=torch.long) # (1, T)
        y_pt = torch.tensor([y_chunk], dtype=torch.long) # (1, T)
        
        logits_pt, loss_pt = pt_model(x_pt, y_pt)
        loss_pt.backward()
        pt_optimizer.step()
        
        losses_pt.append(loss_pt.item())
        
        # --- MicroGPT Forward ---
        # MicroGPT processes token by token in a loop and sums loss
        mg_loss = microgpt_original.Value(0)
        n = len(x_chunk)
        
        # Zero Grads
        for p in mg_params:
            p.grad = 0
            
        keys, values = [[] for _ in range(microgpt_original.n_layer)], [[] for _ in range(microgpt_original.n_layer)]
        batch_losses = []
        
        for pos_id in range(n):
            token_id = x_chunk[pos_id]
            target_id = y_chunk[pos_id]
            
            logits = microgpt_original.gpt(token_id, pos_id, keys, values, mg_state_dict)
            probs = microgpt_original.softmax(logits)
            loss_t = -probs[target_id].log()
            batch_losses.append(loss_t)
            
        if batch_losses:
            mg_loss_obj = sum(batch_losses, microgpt_original.Value(0)) / n
            mg_loss_val = mg_loss_obj.data
            
            mg_loss_obj.backward()
            
            # Optimizer Update
            lr_t = 0.01 # Fixed LR for simplicity or match decay?
            # PyTorch Adam default has no decay unless scheduled.
            # microgpt.py has linear decay: lr_t = learning_rate * (1 - step / num_steps)
            # Let's DISABLE DECAY in both for strict test, or match it.
            # pt_optimizer has constant LR 0.01.
            # So we set MicroGPT LR to 0.01 constant.
            
            for i, p in enumerate(mg_params):
                mg_m[i] = 0.85 * mg_m[i] + (1 - 0.85) * p.grad
                mg_v[i] = 0.99 * mg_v[i] + (1 - 0.99) * p.grad**2
                
                # Bias correction
                m_hat = mg_m[i] / (1 - 0.85 ** (step + 1))
                v_hat = mg_v[i] / (1 - 0.99 ** (step + 1))
                
                p.data -= 0.01 * m_hat / (v_hat**0.5 + 1e-8)
                p.grad = 0 # manually zero again just in case
                
            losses_mg.append(mg_loss_val)
        else:
            losses_mg.append(0)

        if step % 10 == 0:
            print(f"Step {step}: PT={loss_pt.item():.4f}, MG={mg_loss_val:.4f}, Diff={abs(loss_pt.item() - mg_loss_val):.6f}")

    return losses_pt, losses_mg

if __name__ == "__main__":
    pt_hist, mg_hist = run_comparison(100)
    
    plt.figure(figsize=(10, 6))
    plt.plot(pt_hist, label='PyTorch', linestyle='--')
    plt.plot(mg_hist, label='MicroGPT (Original)', alpha=0.7)
    plt.title('Strict Equivalence Check: PyTorch vs MicroGPT')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('equivalence_plot.png')
    
    final_diff = abs(pt_hist[-1] - mg_hist[-1])
    print(f"Final Difference: {final_diff:.6f}")
    
    report = f"""# Equivalence Test Report

## Comparison
- Steps: 100
- Initial Weights: Synced (PyTorch -> MicroGPT)
- Optimizer: Adam (betas=(0.85, 0.99), eps=1e-8, lr=0.01 constant)
- Data: Identical deterministic sequence

## Results
- Final PyTorch Loss: {pt_hist[-1]:.6f}
- Final MicroGPT Loss: {mg_hist[-1]:.6f}
- Difference: {final_diff:.6f}

## Conclusion
The implementation is {"STRONGLY EQUIVALENT" if final_diff < 0.01 else "DIVERGENT"}.
    """
    with open("equivalence_report.md", "w") as f:
        f.write(report)
