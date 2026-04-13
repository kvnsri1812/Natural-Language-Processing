"""
=============================================================
CS5760 Natural Language Processing - Spring 2026
Homework 4 - Q2: Mini Transformer Encoder for Sentences
Student: KOMATLAPALLI VENKATA NAGA SRI
ID     : 700773763
=============================================================

Description:
    Implements a single Transformer encoder block from scratch in PyTorch.

    Pipeline:
      1. Define a small dataset of 10 short sentences.
      2. Build a word-level vocabulary and tokenize + pad each sentence.
      3. Pass token IDs through an Embedding layer.
      4. Add sinusoidal positional encoding so the model knows word order.
      5. Apply Multi-Head Self-Attention (2 heads) → Add & Norm.
      6. Apply a Feed-Forward sublayer (Linear → ReLU → Linear) → Add & Norm.
      7. Print:
           - Input token sequences
           - Final contextual embeddings for sentence 1
           - A text-based attention heatmap showing which tokens attend
             to which other tokens in sentence 1.

    Key concepts demonstrated:
      - Sinusoidal PE: position information injected without learned params.
      - Multi-head attention: each head attends in a different subspace.
      - Add & Norm: residual connections + LayerNorm for stable training.
      - Feed-forward: position-wise 2-layer MLP with inner dim = 4 × d_model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ─────────────────────────────────────────────────────────
# Step 1: Dataset — 10 Short Sentences
# ─────────────────────────────────────────────────────────
# Chosen to cover a variety of NLP-related vocabulary.
sentences = [
    "i love nlp",
    "deep learning is fun",
    "transformers are powerful",
    "attention is all you need",
    "language models predict text",
    "nlp solves real problems",
    "embeddings capture meaning",
    "gradient descent trains models",
    "softmax gives probabilities",
    "words have context",
]

# ─────────────────────────────────────────────────────────
# Step 2: Vocabulary and Tokenization
# ─────────────────────────────────────────────────────────
# Index 0 is reserved for the PAD token (padding_idx).
# All other words are assigned consecutive indices starting at 1.
all_words  = sorted(set(" ".join(sentences).split()))
word2idx   = {w: i + 1 for i, w in enumerate(all_words)}   # 0 = PAD
vocab_size = len(word2idx) + 1   # +1 for PAD

# All sentences are padded to the length of the longest sentence.
MAX_LEN = max(len(s.split()) for s in sentences)

def tokenize(sentence):
    """
    Convert a sentence string to a list of integer token IDs (with padding).

    Returns:
        ids   : list of int, length MAX_LEN
        tokens: list of str (original words, no padding)
    """
    tokens = sentence.split()
    ids    = [word2idx[w] for w in tokens]
    ids   += [0] * (MAX_LEN - len(ids))   # right-pad with zeros
    return ids, tokens

tokenized = [tokenize(s) for s in sentences]
# ids_batch shape: (10, MAX_LEN)
ids_batch = torch.tensor([t[0] for t in tokenized], dtype=torch.long)

print("=" * 55)
print("Input Tokens (first 3 sentences):")
for i in range(3):
    print(f"  S{i+1}: {tokenized[i][1]}")

# ─────────────────────────────────────────────────────────
# Step 3: Token Embedding
# ─────────────────────────────────────────────────────────
# padding_idx=0 ensures the PAD embedding is always zero (no gradient).
D_MODEL   = 32   # embedding / model dimension
embedding = nn.Embedding(vocab_size, D_MODEL, padding_idx=0)

# x shape: (10, MAX_LEN, D_MODEL)
x = embedding(ids_batch)

# ─────────────────────────────────────────────────────────
# Step 4: Sinusoidal Positional Encoding
# ─────────────────────────────────────────────────────────
# PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
# This gives each position a unique, fixed encoding that the model
# can use to distinguish word order without learned position embeddings.
def sinusoidal_pe(max_len, d_model):
    """
    Construct a sinusoidal positional encoding matrix.

    Args:
        max_len : maximum sequence length
        d_model : embedding dimension

    Returns:
        pe : Tensor of shape (1, max_len, d_model)
    """
    pe       = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()   # (max_len, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)   # even dimensions → sin
    pe[:, 1::2] = torch.cos(position * div_term)   # odd  dimensions → cos
    return pe.unsqueeze(0)   # (1, max_len, d_model) — broadcastable over batch

pe = sinusoidal_pe(MAX_LEN, D_MODEL)
x  = x + pe   # broadcast adds PE to every sentence in the batch

# ─────────────────────────────────────────────────────────
# Step 5: Multi-Head Self-Attention (2 heads) + Add & Norm
# ─────────────────────────────────────────────────────────
# Self-attention: Q = K = V = x  (every token attends to all others).
# 2 heads: head 1 and head 2 each operate in a d_model/2 = 16-dim subspace,
# potentially specialising in different linguistic relationships.
N_HEADS = 2
mha     = nn.MultiheadAttention(embed_dim=D_MODEL, num_heads=N_HEADS, batch_first=True)

# Forward pass: returns attended output and per-head averaged weight matrix
attn_out, attn_weights = mha(x, x, x)
# attn_out    : (10, MAX_LEN, D_MODEL)
# attn_weights: (10, MAX_LEN, MAX_LEN) — attention scores between all token pairs

# Add & Norm: residual connection stabilises gradient flow;
# LayerNorm normalises across the feature dimension for each token.
norm1 = nn.LayerNorm(D_MODEL)
x     = norm1(x + attn_out)

# ─────────────────────────────────────────────────────────
# Step 6: Feed-Forward Sublayer + Add & Norm
# ─────────────────────────────────────────────────────────
# Position-wise MLP applied identically at every token position.
# Inner dimension is 4 × D_MODEL (following the original Transformer paper).
ff = nn.Sequential(
    nn.Linear(D_MODEL, D_MODEL * 4),   # expand to richer representation
    nn.ReLU(),                          # non-linearity
    nn.Linear(D_MODEL * 4, D_MODEL)    # project back to model dimension
)
ff_out = ff(x)

# Second residual connection + LayerNorm
norm2 = nn.LayerNorm(D_MODEL)
x     = norm2(x + ff_out)   # final contextual embeddings: (10, MAX_LEN, D_MODEL)

# ─────────────────────────────────────────────────────────
# Step 7: Display Contextual Embeddings
# ─────────────────────────────────────────────────────────
print("\nFinal Contextual Embeddings — Sentence 1 (all tokens):")
print(f"  Sentence: \"{sentences[0]}\"")
for i, word in enumerate(tokenized[0][1]):
    vec = x[0, i].detach().numpy().round(3)
    print(f"  '{word}': {vec}")

# ─────────────────────────────────────────────────────────
# Step 8: Text-Based Attention Heatmap (Sentence 1)
# ─────────────────────────────────────────────────────────
# attn_weights[0] is the (MAX_LEN × MAX_LEN) attention matrix for
# the first sentence. Each row i shows how much token i attends to
# every other token j. Row sums equal 1.0 (after softmax).
words_s1 = tokenized[0][1]                     # ['i', 'love', 'nlp']
attn_s1  = attn_weights[0].detach().numpy()    # (MAX_LEN, MAX_LEN)
L        = len(words_s1)

print(f"\nAttention Heatmap — Sentence 1: \"{sentences[0]}\"")
print("(row = query token, col = key token, values = attention weight)")
print()

header = "         " + "  ".join(f"{w[:5]:>5}" for w in words_s1)
print(header)
for i, w in enumerate(words_s1):
    row = f"{w[:7]:>7}: " + "  ".join(f"{attn_s1[i, j]:.2f}" for j in range(L))
    print(row)

print("\nRow sums (should all ≈ 1.0):")
for i, w in enumerate(words_s1):
    print(f"  {w}: {sum(attn_s1[i, j] for j in range(L)):.4f}")
