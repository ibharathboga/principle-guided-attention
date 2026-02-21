import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PGALayer(nn.Module):
    """
    Principle-Guided Attention Layer.
    Implements the core mechanic where Attention Weights (Q, K, V) are dynamically 
    modified by a Principle Matrix (P).
    """
    def __init__(self, d_model, n_heads):
        super(PGALayer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Standard QKV Projections (Learnable Base Weights)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Final output projection
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, principle_matrix):
        """
        Arguments:
            x: Input tensor (Batch, Seq_Len, D_Model)
            principle_matrix: The Transformation Matrix P (Batch, D_Model, D_Model)
                              This P is predicted/generated for the *entire context* or per-token.
                              Here we assume one global P per sequence for simplicity.
        """
        batch_size, seq_len, _ = x.size()
        
        # 1. Dynamic Weight Modification (Simulated)
        # Instead of explicitly computing W' = W @ P and then storing it,
        # we can compute the projection X @ W and then apply P.
        # Rationale: (X @ W) @ P is equivalent to X @ (W @ P).
        # This is more efficient for backprop.
        
        # Standard Projections
        Q = self.W_q(x) # (Batch, Seq, Dim)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Apply Principle Matrix P to the Projected vectors
        # Logic: We are rotating the latent space of Q, K, V by P.
        # Check dimensions:
        # Q: (Batch, Seq, Dim)
        # P: (Batch, Dim, Dim)
        # Result: (Batch, Seq, Dim)
        
        Q_prime = torch.bmm(Q, principle_matrix)
        K_prime = torch.bmm(K, principle_matrix)
        V_prime = torch.bmm(V, principle_matrix)
        
        # 2. Split into Heads
        Q_prime = Q_prime.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K_prime = K_prime.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V_prime = V_prime.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 3. Scaled Dot-Product Attention
        scores = torch.matmul(Q_prime, K_prime.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        
        # 4. Integrate Value
        context = torch.matmul(attn_weights, V_prime)
        
        # 5. Concatenate Heads and Output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        
        return output, attn_weights
