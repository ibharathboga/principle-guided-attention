import torch
import torch.nn as nn

class PrincipleDiscoveryNetwork(nn.Module):
    """
    The 'Principle' is not just retrieved but *discovered* (learned) from the context.
    This network takes the input context and predicts the Transformation Matrix P.
    """
    def __init__(self, d_model):
        super(PrincipleDiscoveryNetwork, self).__init__()
        self.d_model = d_model
        
        # Bottleneck architecture to capture high-level "Principle" features
        self.extractor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model * d_model) # Output full matrix
        )
        
    def forward(self, x):
        """
        x: Input context. Usually the [CLS] token or Mean Pooled vector of the sequence.
           Shape: (Batch, D_Model)
        """
        batch_size = x.size(0)
        
        # Predict flat matrix params
        p_flat = self.extractor(x)
        
        # Reshape to Matrix P
        P = p_flat.view(batch_size, self.d_model, self.d_model)
        
        # Optional: Initialize near Identity for stability at start of training?
        # For now, we let it learn from scratch.
        # A residual connection P = I + Delta might be better for convergence.
        # Let's implement P = Identity + Extracted_Features for stability.
        
        I = torch.eye(self.d_model, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        P_final = I + (P * 0.1) # Start with small perturbations
        
        return P_final
