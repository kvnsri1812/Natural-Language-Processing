# Homework 4 – Neural Networks, RNNs, Transformers & Attention

## Student Information

* **Name:** Komatlapalli Venkata Naga Sri
* **ID:** 700773763
* **Course:** CS5760 – Natural Language Processing
* **Semester:** Spring 2026

---

## Homework Description

This homework covers feedforward neural networks, XOR with ReLU networks, the perceptron algorithm, and three programming implementations in PyTorch: a character-level RNN language model, a mini Transformer encoder, and scaled dot-product attention.

---

## Repository Structure

### `q1_char_rnn.py`

Trains a character-level LSTM language model on a toy corpus (`hello`, `help`, `world`, `here`).

* Builds a character vocabulary and encodes the corpus as integer indices
* Uses a sliding-window `CharDataset` with sequence length 20
* Model architecture: **Embedding → LSTM → Linear (logits over vocab)**
* Trained with **teacher forcing** using Cross-Entropy loss and Adam optimizer
* Gradient clipping (`max_norm=5.0`) prevents exploding gradients
* Generates 300 characters autoregressively at three temperatures:
  * `τ = 0.7` → confident, repetitive output
  * `τ = 1.0` → raw model distribution
  * `τ = 1.2` → creative but noisier output
* Saves training loss curve to `rnn_loss_curve.png`

### `q1_output.txt`

Console output from running `q1_char_rnn.py`:

* Vocabulary listing (9 characters)
* Per-epoch loss (Epoch 5 → 20)
* Three temperature-controlled generation samples
* Loss dropped from ~2.0 at epoch 1 to **0.1625** at epoch 20

### `rnn_loss_curve.png`

Training loss curve saved by `q1_char_rnn.py` showing steady convergence over 20 epochs.

---

### `q2_transformer_encoder.py`

Builds a single **Transformer encoder block** from scratch in PyTorch and processes a batch of 10 short sentences.

* Builds a word-level vocabulary; index 0 reserved for PAD
* Pads all sentences to `MAX_LEN` (length of longest sentence)
* Adds **sinusoidal positional encoding** (no learned position params):
  * `PE(pos, 2i)   = sin(pos / 10000^(2i/d))`
  * `PE(pos, 2i+1) = cos(pos / 10000^(2i/d))`
* Applies **Multi-Head Self-Attention** (2 heads, `d_model=32`) — Q = K = V = x
* **Add & Norm** after attention (residual connection + LayerNorm)
* **Feed-Forward sublayer**: Linear(32→128) → ReLU → Linear(128→32)
* **Add & Norm** after feed-forward
* Prints final 32-dimensional contextual embeddings for each word in sentence 1
* Prints a text-based **attention heatmap** showing pairwise attention weights

### `q2_output.txt`

Console output from running `q2_transformer_encoder.py`:

* Input token sequences for the first 3 sentences
* 32-dimensional contextual embedding vectors for `'i'`, `'love'`, `'nlp'`
* Attention heatmap with row/column labels for sentence 1

---

### `q3_scaled_attention.py`

Implements the **Scaled Dot-Product Attention** mechanism:

```
Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V
```

**Two test cases are run:**

* **Test 1 – Basic attention** (no mask):
  * Random Q, K, V of shape `(4, 8)`
  * Prints raw vs. scaled score statistics (softmax stability check)
  * Prints attention weight matrix and verifies all rows sum to 1.0
  * Prints output vectors (weighted sum of values)

* **Test 2 – Causal (decoder) masked attention**:
  * Applies a lower-triangular causal mask (future positions set to `-inf`)
  * Demonstrates the autoregressive property: token `i` attends only to positions `≤ i`
  * Verifies row sums still equal 1.0 after masking

### `q3_output.txt`

Console output from running `q3_scaled_attention.py`:

* Raw score std = **3.07** → scaled score std = **1.09** (≈ 3.07 / √8), confirming scaling stabilises softmax
* Attention weight matrix with all row sums = 1.0
* Causal mask matrix and masked attention weights
* Both sets of output vectors

---

## Mathematical Formulations

**Hidden pre-activation (Q6):**
```
zⱼ = Σᵢ wᵢⱼ · xᵢ + bⱼ
```

**Sigmoid activation:**
```
σ(z) = 1 / (1 + e^(−z))
```

**Binary cross-entropy loss:**
```
L = −[t · log(ŷ) + (1 − t) · log(1 − ŷ)]
```

**Perceptron update rule:**
```
w ← w + η · t · x,    b ← b + η · t
```

**Scaled dot-product attention:**
```
Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V
```

---

## Files Included

| File | Description |
|------|-------------|
| `q1_char_rnn.py` | Character-level RNN language model |
| `q1_output.txt` | Output from Q1 |
| `rnn_loss_curve.png` | Training loss plot from Q1 |
| `q2_transformer_encoder.py` | Mini Transformer encoder |
| `q2_output.txt` | Output from Q2 |
| `q3_scaled_attention.py` | Scaled dot-product attention |
| `q3_output.txt` | Output from Q3 |
| `README.md` | Project documentation |

---

## How to Run

1. Clone the repository
2. Navigate to the project directory
3. Install dependencies:

```
pip install torch matplotlib
```

4. Run each program:

```
python q1_char_rnn.py
python q2_transformer_encoder.py
python q3_scaled_attention.py
```

---

## Expected Outputs

**Q1** prints vocabulary, per-epoch loss, and three temperature-controlled text generation samples. Saves `rnn_loss_curve.png`.

**Q2** prints input token sequences for the first 3 sentences, final 32-dimensional contextual embeddings for sentence 1, and an attention heatmap.

**Q3** prints a softmax stability check (raw vs. scaled scores), the full attention weight matrix with row sum verification, output vectors, and a causal masked attention demonstration.

---

## Key Concepts Used

* Feedforward neural networks with sigmoid activations
* Binary cross-entropy loss
* ReLU networks and XOR classification
* Perceptron learning rule and decision boundaries
* Recurrent Neural Networks (RNNs) and LSTMs
* Teacher forcing and temperature-controlled sampling
* Sinusoidal positional encoding
* Multi-head self-attention
* Add & Norm (residual connections + LayerNorm)
* Scaled dot-product attention and causal masking

---

## Conclusion

This homework demonstrates the progression from simple perceptrons to deep sequence models. The character-level RNN illustrates how LSTMs capture sequential patterns through gated memory. The mini Transformer encoder shows how self-attention builds context-aware representations without recurrence. The scaled dot-product attention implementation confirms that dividing by √d_k is essential for numerical stability — without it, the softmax saturates and gradients vanish during training.
