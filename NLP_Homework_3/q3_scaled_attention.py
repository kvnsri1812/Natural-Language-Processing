"""
=============================================================
CS5760 Natural Language Processing - Spring 2026
Homework 4 - Q3: Scaled Dot-Product Attention
Student: KOMATLAPALLI VENKATA NAGA SRI
ID     : 700773763
=============================================================

Description:
    Implements the core scaled dot-product attention mechanism:

        Attention(Q, K, V) = softmax( QK^T / sqrt(d_k) ) * V

    This is the fundamental building block of the Transformer architecture
    (Vaswani et al., "Attention Is All You Need", 2017).

    Steps demonstrated:
      1. Compute raw dot-product scores: QK^T
      2. Scale by 1/sqrt(d_k) to prevent softmax saturation.
      3. Apply optional causal mask (sets future positions to -inf).
      4. Apply softmax to get attention weights (rows sum to 1).
      5. Compute weighted sum of Value vectors.

    Output printed:
      - Raw vs. scaled score statistics (stability check)
      - Attention weight matrix (after softmax)
      - Row sum verification (each row must sum to 1.0)
      - Final output vectors
      - Optional: causal (decoder) mask demonstration
"""

import torch
import torch.nn.functional as F
import math

# ─────────────────────────────────────────────────────────
# Core Function: Scaled Dot-Product Attention
# ─────────────────────────────────────────────────────────
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.

    Formula:
        Attention(Q, K, V) = softmax( QK^T / sqrt(d_k) ) * V

    Args:
        Q    (Tensor): Query  matrix, shape (..., seq_len, d_k)
        K    (Tensor): Key    matrix, shape (..., seq_len, d_k)
        V    (Tensor): Value  matrix, shape (..., seq_len, d_v)
        mask (Tensor): Optional boolean mask; positions where mask==0
                       are set to -inf before softmax (used for causal
                       / decoder masking to prevent attending to future
                       tokens).

    Returns:
        output        (Tensor): Attended output,       shape (..., seq_len, d_v)
        attn_weights  (Tensor): Attention weight matrix, shape (..., seq_len, seq_len)
        raw_scores    (Tensor): Unscaled dot-product scores, shape (..., seq_len, seq_len)
        scaled_scores (Tensor): Scores after dividing by sqrt(d_k), same shape
    """
    d_k = Q.size(-1)   # key/query dimension

    # ── Step 1: Raw dot products QK^T ────────────────────
    # Each query vector is compared against all key vectors.
    # High dot product = high relevance between two positions.
    raw_scores = torch.matmul(Q, K.transpose(-2, -1))   # (..., seq_len, seq_len)

    # ── Step 2: Scale by 1/sqrt(d_k) ─────────────────────
    # Without scaling, large d_k causes dot products to grow with variance d_k,
    # pushing softmax into saturation (near-zero gradients).
    # Dividing by sqrt(d_k) keeps variance ~1 regardless of d_k.
    scaled_scores = raw_scores / math.sqrt(d_k)

    # ── Step 3: Optional causal mask ─────────────────────
    # For decoder self-attention, future positions are masked out so the
    # model cannot "peek ahead" during training (autoregressive property).
    if mask is not None:
        # Replace masked positions with -inf → softmax gives them weight ≈ 0
        scaled_scores = scaled_scores.masked_fill(mask == 0, float("-inf"))

    # ── Step 4: Softmax → Attention Weights ──────────────
    # Softmax is applied over the key dimension (dim=-1).
    # Each row sums to 1.0, representing a probability distribution
    # over positions in the sequence.
    attn_weights = F.softmax(scaled_scores, dim=-1)   # (..., seq_len, seq_len)

    # ── Step 5: Weighted Sum of Values ───────────────────
    # Each output position is a convex combination of value vectors,
    # weighted by how much the corresponding query attends to each key.
    output = torch.matmul(attn_weights, V)   # (..., seq_len, d_v)

    return output, attn_weights, raw_scores, scaled_scores


# ─────────────────────────────────────────────────────────
# Test 1: Basic Attention with Random Inputs
# ─────────────────────────────────────────────────────────
torch.manual_seed(42)   # fixed seed for reproducibility

SEQ_LEN = 4   # number of tokens
D_K     = 8   # key / query dimension (d_k)
D_V     = 8   # value dimension (d_v)

# Simulate Q, K, V as random projections of token embeddings
Q = torch.randn(SEQ_LEN, D_K)
K = torch.randn(SEQ_LEN, D_K)
V = torch.randn(SEQ_LEN, D_V)

output, attn_weights, raw_scores, scaled_scores = scaled_dot_product_attention(Q, K, V)

print("=" * 60)
print("TEST 1 — Basic Scaled Dot-Product Attention")
print("=" * 60)
print(f"\nInput shapes:")
print(f"  Q = {tuple(Q.shape)}  (seq_len={SEQ_LEN}, d_k={D_K})")
print(f"  K = {tuple(K.shape)}  (seq_len={SEQ_LEN}, d_k={D_K})")
print(f"  V = {tuple(V.shape)}  (seq_len={SEQ_LEN}, d_v={D_V})")
print(f"\nScale factor: 1 / sqrt({D_K}) = 1 / {math.sqrt(D_K):.4f} = {1/math.sqrt(D_K):.4f}")

# ── Softmax stability check ─────────────────────────────
# The key motivation for scaling: compare the magnitude of raw vs. scaled scores.
print("\n─── Softmax Stability Check ───────────────────────────")
print(f"Raw QK^T scores — max: {raw_scores.max().item():.4f},  "
      f"min: {raw_scores.min().item():.4f},  "
      f"std: {raw_scores.std().item():.4f}")
print(raw_scores.detach().numpy().round(4))

print(f"\nScaled QK^T/sqrt(d_k) — max: {scaled_scores.max().item():.4f},  "
      f"min: {scaled_scores.min().item():.4f},  "
      f"std: {scaled_scores.std().item():.4f}")
print(scaled_scores.detach().numpy().round(4))
print("\nObservation: scaling reduces the std by ~sqrt(d_k), keeping values")
print("in a range where softmax produces well-distributed (non-peaked) weights.")

# ── Attention weight matrix ─────────────────────────────
print("\n─── Attention Weight Matrix (after softmax) ───────────")
print(attn_weights.detach().numpy().round(4))

# Verify rows sum to 1.0 (required property of a valid attention distribution)
row_sums = attn_weights.sum(dim=-1).detach().numpy().round(6)
print(f"\nRow sums (all must equal 1.0): {row_sums}")
assert all(abs(s - 1.0) < 1e-5 for s in row_sums), "Row sums must all be 1.0!"
print("✓ All rows sum to 1.0")

# ── Output vectors ─────────────────────────────────────
print("\n─── Output Vectors (weighted sum of Values) ───────────")
print(output.detach().numpy().round(4))


# ─────────────────────────────────────────────────────────
# Test 2: Causal (Decoder) Masking
# ─────────────────────────────────────────────────────────
# A causal mask is a lower-triangular boolean matrix:
#   mask[i, j] = 1 if j <= i  (position i can attend to j)
#   mask[i, j] = 0 if j >  i  (position i CANNOT attend to future j)
# This enforces the autoregressive property in the Transformer decoder.
print("\n" + "=" * 60)
print("TEST 2 — Causal (Decoder) Masked Attention")
print("=" * 60)

# Build lower-triangular causal mask (1 = keep, 0 = mask out)
causal_mask = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN)).int()
print("\nCausal mask (lower-triangular):")
print(causal_mask.numpy())
print("(0 = future position, will be set to -inf before softmax)")

output_masked, attn_masked, _, scaled_masked = scaled_dot_product_attention(
    Q, K, V, mask=causal_mask
)

print("\nScaled scores after masking (upper triangle = -inf → 0 after softmax):")
print(scaled_masked.detach().numpy().round(4))

print("\nAttention weights with causal mask:")
print(attn_masked.detach().numpy().round(4))
print("\nObservation: upper-triangular entries are 0 — each position")
print("attends only to itself and all positions that came before it.")

row_sums_masked = attn_masked.sum(dim=-1).detach().numpy().round(6)
print(f"\nRow sums (masked): {row_sums_masked}")
print("✓ Rows still sum to 1.0 even with masking")

print("\n─── Masked Output Vectors ─────────────────────────────")
print(output_masked.detach().numpy().round(4))
