"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
Modified to include Principle-Guided Attention (PGA).

Original: @karpathy
PGA Mod: Antigravity
"""

import os
import math
import random
import numpy as np  # Added for efficient SVD and vector operations by user permission

random.seed(42)
np.random.seed(42)

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
import argparse
import sys

# Parse args early
parser = argparse.ArgumentParser(description="MicroGPT PGA")
parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
parser.add_argument(
    "--data", type=str, default="input.txt", help="Path to input data file"
)
args, _ = parser.parse_known_args()

if not os.path.exists(args.data):
    if args.data == "input.txt":
        # Check parent or create dummy
        if os.path.exists("../input.txt"):
            input_path = "../input.txt"
        else:
            input_path = "input.txt"
    else:
        print(f"Error: {args.data} not found")
        sys.exit(1)
else:
    input_path = args.data

try:
    with open(input_path, "r", encoding="utf-8") as f:
        data = f.read()
    docs = [line.strip() for line in data.split("\n") if line.strip()]
except FileNotFoundError:
    # Fallback to names if local file missing and it was default
    if input_path == "input.txt":
        import urllib.request

        names_url = (
            "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
        )
        urllib.request.urlretrieve(names_url, "input.txt")
        docs = [line.strip() for line in open("input.txt") if line.strip()]
    else:
        raise

random.shuffle(docs)
print(f"num docs: {len(docs)}")

# -----------------------------------------------------------------------------
# Tokenizer
# -----------------------------------------------------------------------------
uchars = sorted(set("".join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")


# -----------------------------------------------------------------------------
# Autograd Engine (Value)
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# PGA Components
# -----------------------------------------------------------------------------


class ObservationBuffer:
    def __init__(self, dim, capacity=1000):
        self.dim = dim
        self.capacity = capacity
        # Stores vectors as numpy arrays for efficient retrieval
        self.buffer = np.zeros((capacity, dim), dtype=np.float32)
        self.pointer = 0
        self.count = 0

    def add(self, vector):
        # vector: list of float or np.array
        if isinstance(vector, list):
            vector = np.array(vector)
        self.buffer[self.pointer] = vector
        self.pointer = (self.pointer + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def retrieve_similar(self, query_vector, k=5):
        if self.count == 0:
            return []

        # Simple cosine similarity
        # query: (dim,)
        # buffer: (count, dim)
        current_buffer = self.buffer[: self.count]

        # Normalize
        q_norm = np.linalg.norm(query_vector) + 1e-8
        b_norm = np.linalg.norm(current_buffer, axis=1) + 1e-8

        sims = np.dot(current_buffer, query_vector) / (q_norm * b_norm)

        # Top k
        k = min(k, self.count)
        indices = np.argsort(sims)[-k:]
        return current_buffer[indices]

    def get_recent(self, k=10):
        if self.count == 0:
            return []
        # Get last k inserted elements
        # Handle wrap-around or just take simplistic view for this linear buffer implementation
        # Since we just fill it linearly up to capacity and overwrite:
        # If we wrapped, the "recent" are at pointer-1, pointer-2...
        # If not wrapped, same.

        indices = []
        for i in range(k):
            idx = (self.pointer - 1 - i) % self.capacity
            # Check if this index actually has data (if not full)
            # But self.count tracks how many valid items we have.
            # If we haven't wrapped, we shouldn't go negative beyond 0 effectively.
            if self.count < self.capacity and idx > self.pointer:
                # determining valid range is tricky with circular buffer if not full
                # simpler: if count < capacity, we only have data up to pointer.
                continue
            indices.append(idx)

        if not indices:
            return []

        return self.buffer[indices]


class PrincipleEngine:
    def __init__(self, dim):
        self.dim = dim

    def discover_principle(self, vectors):
        """
        Input: list of vectors (np.arrays)
        Output: Principle Matrix W_p (np.array of shape (dim, dim))
        """
        if len(vectors) == 0:
            # Identity if no context
            return np.eye(self.dim)

        # Stack vectors: (N, dim)
        X = np.stack(vectors)

        # Center the data
        X_mean = np.mean(X, axis=0)
        X_centered = X - X_mean

        # SVD: X = U S V^T
        # V^T rows are eigenvectors (principal components)
        try:
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

            # Construct Projection Matrix P using top components
            # We want to project INTO the subspace defined by the principle (or filter noise)
            # Let's keep top components that explain majority of variance
            # For simplicity, keep top 50% or fixed rank
            rank = max(1, min(self.dim // 2, len(vectors)))

            V_k = Vt[:rank, :]  # (rank, dim)

            # Projection P = V_k^T . V_k
            # V_k is (rank, dim).
            # We want to project x onto the subspace spanned by V_k.
            # The projection matrix P is V_k.T @ V_k if V_k rows are orthonormal (which they are from SVD)
            P = np.dot(V_k.T, V_k)

            return P

        except np.linalg.LinAlgError:
            return np.eye(self.dim)


# -----------------------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------------------
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head

# Init components
# We need a global buffer for this demo
obs_buffer = ObservationBuffer(n_embd, capacity=2000)
principle_engine = PrincipleEngine(n_embd)

# Init params
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

params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")


# -----------------------------------------------------------------------------
# Architecture
# -----------------------------------------------------------------------------
def linear(x, w):
    # x: list of K vectors, each dim Din. w: Dout x Din. Result: K vectors of Dout.
    # OR x: vector of Din. w: Dout x Din. Result: vector of Dout.
    if isinstance(x[0], list) or isinstance(
        x[0], Value
    ):  # It's a vector since Value is scalar
        pass

    # Check if x is a single vector or batch (sequence)
    # in microgpt, x is usually [Value()...] (a single vector) or [[Value()...]...] not really support batched linear easily without loops
    # actually, look at usage: q = linear(x, ...) where x is [t1_emb, t2_emb...] ?
    # In microgpt:
    # x = [t+p...] -> list of lists? No.
    # state_dict["wte"][token_id] returns a list of Values (one embedding vector).
    # x = [t+p] is a list of lists of Values?
    # token_id is scalar. state_dict["wte"] is matrix (list of lists). state_dict["wte"][token_id] is a ROW (list of Values).
    # So x in gpt main loop:
    # x starts as one vector (embedding of current token) + pos_emb.
    # Wait, microgpt gpt function:
    # x = [t+p ...] -> checks if list of lists?
    # zip(tok_emb, pos_emb) returns tuples of (Value, Value).
    # x becomes [Value, Value...] -> Single Vector (dim n_embd).

    # BUT, the attention block needs the sequence.
    # gpt() in microgpt.py takes `pos_id` and `token_id`. It processes ONE token at a time?
    # YES. "logits = gpt(token_id, pos_id, keys, values)"
    # It seems microgpt is an RNN-like processing loop or auto-regressive stepwise.
    # It builds keys/values cache passed in.

    # linear(x, w): x is a vector [v1, v2...]. w is list of lists.
    # returns [sum(...) for wo in w]. A new vector.
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def matmul_val_np(val_vec, np_mat):
    """
    Multiply a list of Values (vector) by a numpy matrix (constant).
    Result is a list of Values.
    val_vec: (D,)
    np_mat: (D, D')
    """
    # result = val_vec . np_mat
    D_in = len(val_vec)
    D_out = np_mat.shape[1]

    out = []
    for j in range(D_out):
        # dot product of val_vec and j-th column of np_mat
        acc = Value(0)  # start with zero
        # Optim: use sum
        acc = sum((val_vec[i] * np_mat[i, j] for i in range(D_in)), Value(0))
        out.append(acc)
    return out


def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def gpt(token_id, pos_id, keys, values, training=True):
    # 1. Embedding
    tok_emb = state_dict["wte"][token_id]
    pos_emb = state_dict["wpe"][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    # --- PGA: Principle Discvoery ---
    # We use x (current embedding) to query the buffer
    # We must treat x as data for the query (detach gradient)
    x_data = np.array([v.data for v in x])

    # 1. Retrieve
    retrieved_obs = obs_buffer.retrieve_similar(x_data, k=5)

    # 2. Discover Principle
    # Stack = Retrieved (Long-term) + Recent (Short-term) + Current (Immediate)
    recent_obs = obs_buffer.get_recent(k=10)

    # If buffer empty, Wp is identity.
    # Combine retrieved + recent + current
    context_vectors = []
    if len(retrieved_obs) > 0:
        context_vectors.extend(list(retrieved_obs))
    if len(recent_obs) > 0:
        context_vectors.extend(list(recent_obs))

    context_vectors.append(x_data)

    if len(context_vectors) > 0:
        Wp_np = principle_engine.discover_principle(context_vectors)
    else:
        Wp_np = np.eye(n_embd)

    # 3. Store current observation in buffer (if training)
    if training:
        obs_buffer.add(x_data)

    # --------------------------------

    for li in range(n_layer):
        # 1) Modified Attention
        x_residual = x
        x = rmsnorm(x)

        # Apply Wp to weights via the inputs?
        # "q = X . Wq . Wp"
        # Standard: q = Wq . x  (note: in microgpt linear(x, w) does W . x treating x as col vector?
        # linear def: [sum(wi*xi...) for wo in w]. So yes, W . x.
        # So q = Wq . x.
        # Modified: q = Wp^T . (Wq . x) ?
        # Wait, usually X is row vector in formulas (N, D). q = X Wq.
        # Here x is 1D list.
        # Effectively we want to rotate Q, K, V space.

        # Calculate standard q, k, v
        q_std = linear(x, state_dict[f"layer{li}.attn_wq"])
        k_std = linear(x, state_dict[f"layer{li}.attn_wk"])
        v_std = linear(x, state_dict[f"layer{li}.attn_wv"])

        # APPLY PGA: Transform q, k, v by Wp
        # Wp is (D, D). q_std is (D,).
        # We want q_pga = Wp . q_std (if Wp is rotation/projection)
        # Note: Wp_np is symmetric if P = V^T V.
        # Let's apply it.
        q = matmul_val_np(q_std, Wp_np)
        k = matmul_val_np(k_std, Wp_np)
        v = matmul_val_np(v_std, Wp_np)  # Also transform V? Usually yes.

        keys[li].append(k)
        values[li].append(v)

        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs : hs + head_dim]
            k_h = [ki[hs : hs + head_dim] for ki in keys[li]]
            v_h = [vi[hs : hs + head_dim] for vi in values[li]]

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

        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f"layer{li}.mlp_fc1"])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f"layer{li}.mlp_fc2"])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict["lm_head"])
    return logits


# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)
v = [0.0] * len(params)


def step_optimizer(step_num):
    lr_t = learning_rate * (1 - step_num / 1000)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad**2
        m_hat = m[i] / (1 - beta1 ** (step_num + 1))
        v_hat = v[i] / (1 - beta2 ** (step_num + 1))
        p.data -= lr_t * m_hat / (v_hat**0.5 + eps_adam)
        p.grad = 0


# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Training Loop & Execution
# -----------------------------------------------------------------------------


def main():
    # Training
    num_steps = args.steps
    print(f"Training for {num_steps} steps on {args.data}...")

    # Split data
    split_idx = int(len(docs) * 0.9)
    train_docs = docs[:split_idx]
    val_docs = docs[split_idx:]
    print(f"train docs: {len(train_docs)} | val docs: {len(val_docs)}")

    def estimate_loss(split_docs, n_steps=50):
        losses = []
        # Evaluate on a random subset of split_docs
        for _ in range(n_steps):
            doc = random.choice(split_docs)
            tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
            n = min(block_size, len(tokens) - 1)

            keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
            for pos_id in range(n):
                token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
                # training=False prevents buffer updates
                logits = gpt(token_id, pos_id, keys, values, training=False)
                probs = softmax(logits)
                # Use data to avoid building graph
                loss_t = -probs[target_id].log()
                losses.append(loss_t.data)
        return sum(losses) / len(losses)

    for step in range(num_steps):
        doc = train_docs[step % len(train_docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        # Forward
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        losses = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(token_id, pos_id, keys, values, training=True)
            probs = softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)

        loss = sum(losses, Value(0)) / n
        loss.backward()

        # Optimizer step
        lr_t = learning_rate * (1 - step / num_steps)
        for i, p in enumerate(params):
            m[i] = beta1 * m[i] + (1 - beta1) * p.grad
            v[i] = beta2 * v[i] + (1 - beta2) * p.grad**2
            m_hat = m[i] / (1 - beta1 ** (step + 1))
            v_hat = v[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat**0.5 + eps_adam)
            p.grad = 0

        if step % 100 == 0:
            val_loss = estimate_loss(val_docs, n_steps=20)
            print(
                f"Step {step+1:4d}/{num_steps} | Train Loss: {loss.data:.4f} | Val Loss: {val_loss:.4f}"
            )
        elif (step + 1) % 10 == 0:
            print(
                f"Step {step+1:4d}/{num_steps} | Train Loss: {loss.data:.4f}"
            )

    print("\n--- inference (PGA, hallucinated names) ---")
    temperature = 0.5
    for sample_idx in range(20):
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        token_id = BOS
        sample = []
        # Clear buffer for inference or keep?
        # Ideally we keep the buffer from training as "long term memory"

        for pos_id in range(block_size):
            logits = gpt(token_id, pos_id, keys, values, training=False)
            probs = softmax([l / temperature for l in logits])
            token_id = random.choices(
                range(vocab_size), weights=[p.data for p in probs]
            )[0]
            if token_id == BOS:
                break
            sample.append(uchars[token_id])
        print(f"sample {sample_idx+1:2d}: {''.join(sample)}")


if __name__ == "__main__":
    main()
