import numpy as np
from principle_engine import PrincipleEngine

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

class PGAModel:
    def __init__(self, embedding_dim=8):
        self.embedding_dim = embedding_dim
        self.principle_engine = PrincipleEngine(embedding_dim)
        
        # Initialize standard "pretrained" weights
        # In a real model, these are learned parameters.
        self.W_Q = np.random.randn(embedding_dim, embedding_dim)
        self.W_K = np.random.randn(embedding_dim, embedding_dim)
        self.W_V = np.random.randn(embedding_dim, embedding_dim)

    def encode(self, text):
        """
        Mock encoding of text into a state vector.
        In reality, this would be a Transformer embedding.
        """
        # Create a random vector first
        vec = np.random.randn(self.embedding_dim)
        
        # Inject specific values to simulate semantic meaning for the demo
        # Dim 0: Beauty, Dim 1: Physics, Dim 2: Cost
        if "bridge" in text.lower():
            vec[0] = 0.9 # High value for "Beauty" (simulating training bias)
            vec[1] = 0.5 # Moderate value for "Physics"
            vec[2] = 0.3 # Low value for "Cost"
            
        return vec

    def attention_cycle(self, input_vector, principle_matrix):
        """
        Performs the Modified QKV process:
        W' = W . P
        Arguments:
            input_vector: The encoded input state (V_Q).
            principle_matrix: The logic filter (P).
        """
        
        # 1. Dynamic Weight Modification (Change of Basis)
        # W' = W . P
        W_Q_prime = self.W_Q @ principle_matrix
        W_K_prime = self.W_K @ principle_matrix
        W_V_prime = self.W_V @ principle_matrix
        
        # 2. QKV Projections
        Q = input_vector @ W_Q_prime
        K = input_vector @ W_K_prime # Simplified self-attention (K=V_Q in this logic)
        V = input_vector @ W_V_prime
        
        # 3. Attention Score (Logical Relevance)
        d_k = self.embedding_dim
        scores = (Q @ K.T) / np.sqrt(d_k)
        
        # Note: In a full transformer, this is done across a sequence.
        # Here we verify the filtering effect on the Value vector V.
        
        # The Essence Vector (E) is the weighted sum of values.
        # For this single-vector demo, it's just the projected V.
        # But we want to see the *content* of V to see if "Beauty" was suppressed.
        
        return V

    def run(self, input_text):
        print(f"--- Processing: '{input_text}' ---")
        
        # 1. Input & Encoding
        V_Q = self.encode(input_text)
        print(f"  [Input] Original Vector (first 3 dims): {V_Q[0:3]}")
        
        # 2. Principle Extraction
        P = self.principle_engine.discover_principle(input_text)
        
        # 3. Modified QKV
        E = self.attention_cycle(V_Q, P)
        
        print(f"  [Output] Synthesized Essence (first 3 dims): {E[0:3]}")
        print("-------------------------------------------\n")
        return E
