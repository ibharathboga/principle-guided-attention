import torch
import torch.nn as nn
from src.pga_layer import PGALayer
from src.principle_network import PrincipleDiscoveryNetwork

class PGAModel(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2):
        super(PGAModel, self).__init__()
        self.d_model = d_model
        
        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Principle Discovery Network
        # Learns to generate P from context
        self.principle_net = PrincipleDiscoveryNetwork(d_model)
        
        # 3. Transformer Encoder Layers (Modified with PGA)
        self.layers = nn.ModuleList([
            PGALayer(d_model, n_heads) for _ in range(n_layers)
        ])
        
        # 4. Output Head
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        x: Input tokens (Batch, Seq_Len)
        """
        # Embed and Positional Encoding (omitted for brevity, assume simple embedding)
        x_embed = self.embedding(x) # (Batch, Seq, D_Model)
        
        # Extract Global Context Implementation: Mean Pooling
        context_vector = torch.mean(x_embed, dim=1) # (Batch, D_Model)
        
        # Discover Principle Matrix P
        P = self.principle_net(context_vector) # (Batch, D_Model, D_Model)
        
        # Pass through modified layers
        x = x_embed
        heatmap = None
        
        for layer in self.layers:
            # Each layer receives the Principle Matrix P
            # A more advanced version might refine P at each layer
            x, attn_weights = layer(x, principle_matrix=P)
            heatmap = attn_weights # Keep last layer attention for viz
            
            # Add Residual Connection & Layer Norm (Simplified here)
            # In a full model: x = LayerNorm(x + sublayer(x))
            # We omit for clarity of the core PGA mechanism demo.
            
        # Decode
        logits = self.fc_out(x)
        
        return logits, P, heatmap
