"""
PGA Training Script — with Observation Buffer feedback.

Demonstrates the full lifecycle:
  1. Model receives input tokens.
  2. It retrieves relevant past observations from the buffer.
  3. It discovers a Principle Matrix P from (query + retrieved).
  4. It runs modified attention with P.
  5. The output essence E is written BACK to the buffer.
  6. Over time, the buffer accumulates experience and the model can
     retrieve richer context for principle discovery.

Training dataset is intentionally small (proof of concept).
Two domains: Physics (bridge/load) and Art (poem/beauty).
The model should learn to produce different P matrices for each domain,
AND the buffer should accumulate domain-specific observations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from src.model import PGAModel

# ── Vocabulary ──────────────────────────────────────────────────────
vocab = {
    "PAD": 0, "design": 1, "bridge": 2, "heavy": 3, "load": 4,
    "write": 5, "poem": 6, "love": 7, "beauty": 8, "structure": 9,
    "tension": 10, "mass": 11, "emotion": 12, "rhythm": 13,
    "steel": 14, "concrete": 15,
}
inv_vocab = {v: k for k, v in vocab.items()}

# ── Training Data ───────────────────────────────────────────────────
# (context_tokens, target_token)
# Physics domain: bridge-related → physics answers
# Art domain: poem-related → art answers
data = [
    # Physics
    ([1, 2, 3],   4),   # design bridge heavy       → load
    ([1, 2, 9],   10),  # design bridge structure    → tension
    ([1, 2, 11],  4),   # design bridge mass         → load
    ([1, 14, 15], 9),   # design steel concrete      → structure
    ([2, 3, 11],  10),  # bridge heavy mass          → tension
    # Art
    ([5, 6, 7],   8),   # write poem love            → beauty
    ([5, 6, 12],  13),  # write poem emotion         → rhythm
    ([5, 6, 0],   7),   # write poem PAD             → love
    ([5, 12, 13], 8),   # write emotion rhythm       → beauty
    ([6, 7, 12],  13),  # poem love emotion          → rhythm
]


def train():
    print("=" * 60)
    print("  PGA Training — with Observation Buffer Feedback Loop")
    print("=" * 60)

    # ── Hyperparameters ─────────────────────────────────────────
    vocab_size = len(vocab)
    d_model = 32
    n_heads = 4
    n_layers = 2
    buffer_capacity = 64
    retrieval_top_k = 4
    learning_rate = 0.005
    epochs = 200

    model = PGAModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        buffer_capacity=buffer_capacity,
        retrieval_top_k=retrieval_top_k,
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Observation Buffer: {model.obs_buffer}")
    print()

    for epoch in range(epochs):
        total_loss = 0.0
        total_consistency = 0.0

        for context, target in data:
            input_tensor  = torch.tensor([context], dtype=torch.long)
            target_tensor = torch.tensor([target],  dtype=torch.long)

            optimizer.zero_grad()

            logits, P, attn, essence, retrieved, consistency = model(input_tensor)

            last_logits = logits[:, -1, :]
            loss = criterion(last_logits, target_tensor)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_consistency += consistency.item()

        if epoch % 20 == 0:
            avg_loss = total_loss / len(data)
            avg_con  = total_consistency / len(data)
            buf_count = model.obs_buffer.count.item()
            print(
                f"Epoch {epoch:>4d}  |  Loss: {avg_loss:.4f}  |  "
                f"Avg Consistency: {avg_con:.4f}  |  "
                f"Buffer Size: {buf_count}/{buffer_capacity}"
            )

    print("\n--- Training Complete ---")

    # ── Save ────────────────────────────────────────────────────
    torch.save(model.state_dict(), "pga_model_v2.pth")
    print("Model saved to pga_model_v2.pth")

    # ── Verification ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Verification: Principle Matrix & Buffer Analysis")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        # Physics context
        inp_physics = torch.tensor([[1, 2, 3]], dtype=torch.long)
        _, P1, _, E1, ret1, con1 = model(inp_physics)

        # Art context
        inp_art = torch.tensor([[5, 6, 7]], dtype=torch.long)
        _, P2, _, E2, ret2, con2 = model(inp_art)

        print(f"\nContext: 'design bridge heavy'")
        print(f"  Principle Matrix Trace:  {torch.trace(P1[0]):.4f}")
        print(f"  Consistency (E·P):       {con1.item():.4f}")
        print(f"  Retrieved {ret1.shape[1]} vectors from buffer")

        print(f"\nContext: 'write poem love'")
        print(f"  Principle Matrix Trace:  {torch.trace(P2[0]):.4f}")
        print(f"  Consistency (E·P):       {con2.item():.4f}")
        print(f"  Retrieved {ret2.shape[1]} vectors from buffer")

        p_diff = torch.norm(P1 - P2).item()
        e_diff = torch.norm(E1 - E2).item()
        print(f"\n  ‖P_physics - P_art‖ = {p_diff:.4f}")
        print(f"  ‖E_physics - E_art‖ = {e_diff:.4f}")

        if p_diff > 0.5:
            print("\n✓ SUCCESS: Different domains produce distinct Principle Matrices.")
        else:
            print("\n⚠ WARNING: Principle Matrices are too similar.")

        print(f"\n  Final Buffer state: {model.obs_buffer}")


if __name__ == "__main__":
    train()
