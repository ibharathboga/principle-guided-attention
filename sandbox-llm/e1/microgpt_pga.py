"""
MicroGPT with Principle-Guided Attention (PGA).
"""

import os
import math
import random
import numpy as np

# Seeding
random.seed(42)
np.random.seed(42)

# Dataset setup (same as baseline)
if not os.path.exists("input.txt"):
    import urllib.request
    try:
        names_url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
        urllib.request.urlretrieve(names_url, "input.txt")
    except:
        pass # Expect baseline to have handled this or use dummy

try:
    docs = [line.strip() for line in open("input.txt") if line.strip()]
except FileNotFoundError:
     # Fallback dummy data if download fails
    with open("input.txt", "w") as f:
        f.write("anna\nemma\nolivia\nava\nisabella\nsophia\ncharlotte\nmia\namelia\nharper\nevelyn\nabigail\nemily\nella\nelizabeth\ncamila\nluna\nsofia\navery\nmila\naria\nscarlett\npenelope\nlayla\nchloe\nvictoria\nmadison\neleanor\ngrace\nnora\nriley\nzoey\nhannah\nhazel\nlily\nellie\nviolet\nlillian\nzoe\nstella\naurora\nnatalie\nemilia\neverly\nleah\naubrey\nwillow\naddison\nlucy\naudrey\n")
    docs = [line.strip() for line in open("input.txt") if line.strip()]

random.shuffle(docs)

uchars = sorted(set("".join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1

# Autograd Value class (same as baseline)
class Value:
    __slots__ = ("data", "grad", "_children", "_local_grads")

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data ** (other - 1),))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# Hyperparameters
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head

def init_weights():
    matrix = lambda nout, nin, std=0.08: [
        [Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)
    ]
    state_dict = {
        "wte": matrix(vocab_size, n_embd),
        "wpe": matrix(block_size, n_embd),
        "lm_head": matrix(vocab_size, n_embd),
    }
    for i in range(n_layer):
        state_dict[f"layer{i}.attn_wq"] = matrix(n_embd, n_embd)
        state_dict[f"layer{i}.attn_wk"] = matrix(n_embd, n_embd)
        state_dict[f"layer{i}.attn_wv"] = matrix(n_embd, n_embd)
        state_dict[f"layer{i}.attn_wo"] = matrix(n_embd, n_embd)
        state_dict[f"layer{i}.mlp_fc1"] = matrix(4 * n_embd, n_embd)
        state_dict[f"layer{i}.mlp_fc2"] = matrix(n_embd, 4 * n_embd)
    return state_dict

state_dict = init_weights()
params = [p for mat in state_dict.values() for row in mat for p in row]

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

# --- PGA SPECIFIC COMPONENTS ---

def compute_principle_projection(context_vectors, k=8):
    """
    Computes the projection matrix P using SVD on the context vectors.
    context_vectors: list of list of Value objects (the stack)
    k: rank of the subspace (number of top singular vectors to keep)
    """
    # 1. Extract data from Value objects (Stop Gradient essentially)
    # We do NOT backprop through SVD in this version.
    # Shape: (num_vectors, n_embd)
    X = np.array([[v.data for v in vec] for vec in context_vectors])
    
    # 2. SVD
    # X = U S V^T
    # We want V (right singular vectors). 
    # numpy svd returns vt as V^T. 
    # Shape of vt: (n_embd, n_embd)
    try:
        u, s, vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        # Fallback if SVD fails (empty or singular)
        return np.eye(n_embd)

    # 3. Select top k components
    # The rows of vt are the eigenvectors. We want the top k rows.
    # v_top shape: (k, n_embd)
    # If we have fewer samples than k, we take min(k, len(s))
    actual_k = min(k, len(s))
    v_top = vt[:actual_k, :]
    
    # 4. Construct Projection Matrix P = V_top^T @ V_top
    # Shape: (n_embd, n_embd)
    P = np.dot(v_top.T, v_top)
    
    return P

def apply_projection(vectors, P_numpy):
    """
    Projects a list of Value vectors using the numpy matrix P.
    vectors: list of Value objects (size n_embd)
    P_numpy: numpy array (n_embd, n_embd)
    Returns: list of Value objects
    """
    # Output = vectors * P
    # But since P is symmetric (projection), order is flexible, but usually:
    # v_proj = P @ v 
    # Here `vectors` is a row vector x. So x_proj = x @ P.
    
    # We treat P as constant (stop gradient).
    # So we are just doing a linear combination of the input Values.
    
    n = len(vectors) # n_embd
    out_vectors = []
    
    for j in range(n):
        # Result j is dot product of vector x and column j of P
        # col j of P is P[:, j]
        val = sum(vectors[i] * P_numpy[i, j] for i in range(n))
        out_vectors.append(val)
        
    return out_vectors

# -------------------------------

def gpt_pga(token_id, pos_id, keys, values, observation_buffer):
    tok_emb = state_dict["wte"][token_id]
    pos_emb = state_dict["wpe"][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    # Add current observation to buffer (for *next* step potentially, or this step?)
    # In PGA, we use "recent and historical" embeddings. 
    # For this micro-implementation, we'll use the 'keys' (past k vectors) 
    # or we can explicitly maintain a buffer of 'x' before layers.
    # Let's add 'x' to the buffer. 
    # Note: This is computationally expensive if we do SVD every single token.
    # To keep it runnable, we might want to do it per layer or just once.
    # Let's do it per layer for validity.
    
    # Current context: The observation buffer passed in
    # (In training loop we will manage this buffer)
    
    # Compute P based on observation buffer + current x
    # We assume observation_buffer contains previous x's of this sequence
    
    current_context = observation_buffer + [x]
    # Limit context size for performance (e.g., last 16)
    current_context = current_context[-16:] 
    
    P = compute_principle_projection(current_context, k=8)

    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)
        
        # Standard projections
        q = linear(x, state_dict[f"layer{li}.attn_wq"])
        k = linear(x, state_dict[f"layer{li}.attn_wk"])
        v = linear(x, state_dict[f"layer{li}.attn_wv"])
        
        # --- PGA INTERVENTION ---
        # Project Q, K, V onto Principle Subspace
        q = apply_projection(q, P)
        k = apply_projection(k, P)
        v = apply_projection(v, P)
        # ------------------------

        keys[li].append(k)
        values[li].append(v)
        
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs : hs + head_dim]
            k_h = [ki[hs : hs + head_dim] for ki in keys[li]]
            v_h = [vi[hs : hs + head_dim] for vi in values[li]]
            
            # Simple dot product attention
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                for t in range(len(k_h))
            ]
            attn_weights = softmax(attn_logits)
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]
            x_attn.extend(head_out)
            
        x = linear(x_attn, state_dict[f"layer{li}.attn_wo"])
        x = [a + b for a, b in zip(x, x_residual)]
        
        # MLP
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f"layer{li}.mlp_fc1"])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f"layer{li}.mlp_fc2"])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict["lm_head"])
    return logits, x # Return x to update buffer

# Optimizer
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)
v_opt = [0.0] * len(params)

def train(steps=100):
    losses = []
    
    # For PGA, we need to maintain the observation buffer per document/sequence
    # In this simple training loop, we reset per document.
    
    for step in range(steps):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        observation_buffer = [] # Reset for new sequence
        
        batch_losses = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            
            # Pass buffer and get updated x back
            logits, x_final = gpt_pga(token_id, pos_id, keys, values, observation_buffer)
            
            # Update buffer with the PRE-ATTENTION embedding (or post? usually pre-layer inputs are 'observations')
            # In gpt_pga we added the embedding 'x' (after pos_emb) to the context used for P.
            # We should probably persist that decision. 
            # In gpt_pga I did: current_context = observation_buffer + [x]
            # But observation_buffer assumes it stores past 'x's.
            # So we should append 'x' (the one used inside gpt_pga) to observation_buffer for the NEXT token.
            # But we need 'x' from inside gpt_pga.
            # Let's assume gpt_pga handles the "current" buffer correctly, 
            # but we need to store it for the next token's history.
            
            # To fix the scope, let's just cheat and regenerate x_emb here to store it? 
            # Or better, return it from gpt_pga. I updated return to include x (it's actually the final x, not the initial x).
            # The 'Observation' is typically the raw input embedding or the state. 
            # Let's capture the 'x' right after pos_emb.
            
            # Re-calculating x for buffer consistency to avoid passing too much around
            tok_emb = state_dict["wte"][token_id]
            pos_emb = state_dict["wpe"][pos_id]
            x_input = [t + p for t, p in zip(tok_emb, pos_emb)]
            x_input = rmsnorm(x_input)
            observation_buffer.append(x_input)
            
            probs = softmax(logits)
            loss_t = -probs[target_id].log()
            batch_losses.append(loss_t)
            
        if not batch_losses: continue
        
        loss = sum(batch_losses, Value(0)) / n
        losses.append(loss.data)

        loss.backward()

        lr_t = learning_rate * (1 - step / steps)
        for i, p in enumerate(params):
            m[i] = beta1 * m[i] + (1 - beta1) * p.grad
            v_opt[i] = beta2 * v_opt[i] + (1 - beta2) * p.grad**2
            m_hat = m[i] / (1 - beta1 ** (step + 1))
            v_hat = v_opt[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat**0.5 + eps_adam)
            p.grad = 0
            
        if step % 10 == 0:
            print(f"PGA Step {step}/{steps} | Loss: {loss.data:.4f}")
            
    return losses
