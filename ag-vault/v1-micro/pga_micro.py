"""
Principle-Guided Attention (PGA) Implementation of MicroGPT
Based on the architecture by @antigravity
Original MicroGPT by @karpathy
"""

import os
import math
import random
import sys

random.seed(42)

# --- Data Preparation ---
if not os.path.exists("input.txt"):
    import urllib.request
    names_url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
    urllib.request.urlretrieve(names_url, "input.txt")

all_docs = [line.strip() for line in open("input.txt") if line.strip()]
random.shuffle(all_docs)

# Train/Validation Split (90/10)
split_idx = int(len(all_docs) * 0.9)
train_docs = all_docs[:split_idx]
val_docs = all_docs[split_idx:]

print(f"Total docs: {len(all_docs)}")
print(f"Train docs: {len(train_docs)} | Val docs: {len(val_docs)}")

uchars = sorted(set("".join(all_docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# --- Autograd Engine ---
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
        # clamp to avoid log(0)
        x = self.data if self.data > 1e-6 else 1e-6
        return Value(math.log(x), (self,), (1 / x,))

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

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

# --- PGA Components ---

class ObservationBuffer:
    """Buffers recent observations (Principles) to find invariants."""
    def __init__(self, capacity=5):
        self.capacity = capacity
        # Stores 'structure' of past successful inferences. 
        # Ideally stores tensors, here we store raw data matrices for efficiency
        self.buffer = [] 

    def push(self, observation):
        """observation: list of list of floats (embedding state)"""
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(observation)
    
    def get_principle(self, current_obs):
        """
        Derives a Principle Matrix 'P' based on correlation between 
        current observation and buffered 'truths'.
        Simplified for MicroGPT: valid matches increase attention weight.
        """
        if not self.buffer:
            # Identity principle (neutral)
            size = len(current_obs)
            return [[0.0] * size for _ in range(size)]
        
        # Simplified Principle: Average Similarity to past observations
        # If current state resembles past valid states, reinforce those connections.
        # This is a dummy implementation of 'Invariant Discovery'.
        T = len(current_obs)
        P = [[0.0] * T for _ in range(T)]
        
        # We just return a zero matrix in this micro-demo to avoid 
        # excessive computation overhead in pure python, 
        # OR we can simulate a 'Logical Constraint' by inhibiting diagonal.
        # Let's return a mask that slightly penalizes long-distance dependencies 
        # (simulating a "Locality Principle").
        
        for i in range(T):
            for j in range(T):
                dist = abs(i - j)
                if dist > 2:
                    P[i][j] = -0.1 # Inhibit far connections (Logic: "Local Coherence")
        
        return P

def entropy_check(logits):
    """Calculates Shannon entropy of the logits (prob distribution)."""
    # logits is list of Value
    probs = softmax(logits) # returns list of Value
    entropy = sum([-p * p.log() for p in probs], Value(0))
    return entropy

# --- Model Parameters ---
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

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

params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")

obs_buffer = ObservationBuffer(capacity=10)

# --- Helper Functions ---
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

# --- PGA Modified GPT Forward Pass ---
def gpt(token_id, pos_id, keys, values, principle_matrix=None):
    tok_emb = state_dict["wte"][token_id]
    pos_emb = state_dict["wpe"][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    
    # Store observation for PGA (in a real system, we'd store the whole sequence 'x')
    # Here we are processing token-by-token, so 'x' is just one vector. 
    # The 'Principle' usually applies to the whole sequence relation.
    
    x = rmsnorm(x)

    for li in range(n_layer):
        # 1) Multi-head Attention
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f"layer{li}.attn_wq"])
        k = linear(x, state_dict[f"layer{li}.attn_wk"])
        v = linear(x, state_dict[f"layer{li}.attn_wv"])
        keys[li].append(k)
        values[li].append(v)
        
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs : hs + head_dim]
            k_h = [ki[hs : hs + head_dim] for ki in keys[li]]
            v_h = [vi[hs : hs + head_dim] for vi in values[li]]
            
            # Attention Scores
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                for t in range(len(k_h))
            ]
            
            # --- PGA: Apply Principle Matrix ---
            # principle_matrix is T x T. We are at step 'pos_id' (current T).
            # attn_logits has length pos_id+1.
            # We add the corresponding row of P to the logits.
            if principle_matrix:
                # Retrieve row 'pos_id' of P, sliced to current length
                p_row = principle_matrix[pos_id][:len(attn_logits)]
                # Apply Principle "Mask" (P is additive logit bias here)
                attn_logits = [l + p for l, p in zip(attn_logits, p_row)]

            attn_weights = softmax(attn_logits)
            
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]
            x_attn.extend(head_out)
            
        x = linear(x_attn, state_dict[f"layer{li}.attn_wo"])
        x = [a + b for a, b in zip(x, x_residual)]
        
        # 2) MLP
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f"layer{li}.mlp_fc1"])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f"layer{li}.mlp_fc2"])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict["lm_head"])
    return logits, x  # Return x to buffer it

# --- Optimizer ---
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)
v = [0.0] * len(params)

def run_epoch(docs, steps, training=True):
    total_loss = 0
    
    for step in range(steps):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)
        
        # --- PGA: Principle Discovery ---
        # In a full system, we'd pre-scan the doc to find the Principle.
        # Here, we generate a 'Locality Principle' matrix based on sequence length.
        # This simulates the "Invariant Discovery" phase.
        P = obs_buffer.get_principle([0]*n) # Dummy input to get size-based P
        
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        losses = []
        
        # Document Forward Pass
        doc_embedding_sequence = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            
            # Pass P to gpt
            logits, final_x = gpt(token_id, pos_id, keys, values, principle_matrix=P)
            
            # Buffer the observation (detached data)
            doc_embedding_sequence.append([v.data for v in final_x])
            
            # Entropy Check (Logging only)
            if step % 50 == 0 and pos_id == 0:
                e = entropy_check(logits)
                # print(f"Entropy: {e.data:.2f}")

            probs = softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)
            
        loss = sum(losses, Value(0)) / n
        total_loss += loss.data
        
        # Update Observation Buffer with this document's structural "shape"
        # We only buffer if we are fairly confident (not implemented), or just FIFO.
        if training:
            obs_buffer.push(doc_embedding_sequence)
        
        if training:
            loss.backward()
            
            # Adam Update
            lr_t = learning_rate * (1 - step / steps) 
            for i, p in enumerate(params):
                m[i] = beta1 * m[i] + (1 - beta1) * p.grad
                v[i] = beta2 * v[i] + (1 - beta2) * p.grad**2
                m_hat = m[i] / (1 - beta1 ** (step + 1))
                v_hat = v[i] / (1 - beta2 ** (step + 1))
                p.data -= lr_t * m_hat / (v_hat**0.5 + eps_adam)
                p.grad = 0
                
        # Logging
        if training:
            print(f"Step {step+1:4d}/{steps} | Train Loss: {loss.data:.4f}", end="\r")
            
            # Validation Check
            if (step + 1) % 100 == 0:
                val_loss = run_epoch(val_docs, steps=5, training=False)
                print(f"\n[VALIDATION] Step {step+1} | Val Loss: {val_loss:.4f}")

    return total_loss / steps if steps > 0 else 0

# --- Execution ---
print("Starting Training with PGA...")
steps = 300
if len(sys.argv) > 1:
    steps = int(sys.argv[1])
run_epoch(train_docs, steps=steps, training=True)

# --- Inference ---
print("\n--- Inference (PGA Guided) ---")
temperature = 0.5
for sample_idx in range(5):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    # Use the same 'Locality Principle' for inference
    P_inf = obs_buffer.get_principle([0]*block_size)
    
    for pos_id in range(block_size):
        logits, _ = gpt(token_id, pos_id, keys, values, principle_matrix=P_inf)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"Sample {sample_idx+1}: {''.join(sample)}")
