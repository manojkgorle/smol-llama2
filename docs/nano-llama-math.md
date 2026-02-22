# Nano LLaMA -- A 220-Parameter LLM You Can Trace By Hand

## Introduction

Large language models like LLaMA 2 contain 7 billion parameters. That number is so vast that no human can look at the weights and understand what is happening. But here is the thing: the *architecture* of a 7-billion-parameter model is not fundamentally different from a model with 220 parameters. The same operations -- embedding lookups, matrix multiplications, normalizations, rotations, gated activations, softmax -- repeat at every scale. The best way to truly understand a large language model is to shrink it until every single number fits on a sheet of paper, then trace the computation by hand.

This document walks through a complete forward pass of the smallest valid LLaMA model: just **220 learnable parameters**. It uses the exact same architecture as the real thing -- RMSNorm for normalization, Rotary Positional Embeddings (RoPE) for encoding token positions, SwiGLU for the feed-forward network, and multi-head attention for gathering context. The only difference is the size: where the production model uses 384-dimensional embeddings, 8 layers, and a 4096-token vocabulary, our nano model uses 4-dimensional embeddings, 1 layer, and a vocabulary of 6 words. Every architectural choice is identical; only the config values change.

We will process the input **"the cat"** and watch the model predict the next word, step by step. At each stage you will see the exact tensor shapes, the exact matrix multiplications, and the exact element-wise operations that transform two token IDs into a probability distribution over the vocabulary $\{\texttt{<s>}, \texttt{the}, \texttt{cat}, \texttt{sat}, \texttt{on}, \texttt{mat}\}$. By the end, there will be nothing mysterious left about how a LLaMA-style transformer turns text into predictions.

This is the exact architecture from the [llama-vc](https://github.com/balu/llama-vc) project -- a from-scratch LLaMA implementation in PyTorch that trains on TinyStories and reaches 15.74M parameters at production scale. Our nano model is the same `ModelConfig`, the same `LLaMA` class, the same `Attention`, `FeedForward`, and `RMSNorm` modules. We just pass in different numbers.

---

## The Configuration

The nano model uses these hyperparameters:

```python
ModelConfig(
    vocab_size  = 6,        # tokens: <s>, the, cat, sat, on, mat
    dim         = 4,        # each token = 4-dimensional vector
    n_layers    = 1,        # single transformer block
    n_heads     = 2,        # 2 query attention heads
    n_kv_heads  = 2,        # 2 KV heads (full MHA, no GQA grouping)
    head_dim    = 2,        # dim / n_heads = 4 / 2
    hidden_dim  = 8,        # SwiGLU intermediate dimension
    max_seq_len = 8,        # max 8 tokens
)
```

Every choice here is the minimum that satisfies the architectural constraints:

- **`dim = 4`** is the smallest dimension that gives us 2 attention heads with `head_dim = 2`. RoPE requires `head_dim` to be even (it operates on pairs of dimensions), so `head_dim = 2` is the minimum, and 2 heads is the minimum for "multi-head" attention.
- **`n_heads = n_kv_heads = 2`** means every query head has its own key/value head. This is standard multi-head attention (MHA) with no GQA grouping. In the production model, `n_heads = 6` and `n_kv_heads = 2`, so each KV head serves 3 query heads. Here the ratio is 1:1 because we are already at the minimum.
- **`hidden_dim = 8`** is the SwiGLU intermediate dimension. The production model uses $\frac{8}{3} \times \text{dim}$; here $\frac{8}{3} \times 4 \approx 10.67$, but we round to 8 to keep things clean while staying close to the intended ratio.
- **`n_layers = 1`** because one transformer block is enough to demonstrate the full data flow. Every additional layer is an identical repeat.
- **`vocab_size = 6`** gives us just enough tokens for a simple sentence: `<s> the cat sat on mat`.

### Parameter Count Breakdown

Every learnable parameter in the model, accounted for:

| Component | Shape | Parameters |
|:---|:---:|---:|
| Token embedding (`tok_embeddings`) | $6 \times 4$ | 24 |
| Attention norm $\gamma$ (`attention_norm.weight`) | $4$ | 4 |
| Query projection (`Wq`) | $4 \times 4$ | 16 |
| Key projection (`Wk`) | $4 \times 4$ | 16 |
| Value projection (`Wv`) | $4 \times 4$ | 16 |
| Output projection (`Wo`) | $4 \times 4$ | 16 |
| FFN norm $\gamma$ (`ffn_norm.weight`) | $4$ | 4 |
| Gate projection (`W_gate`) | $4 \times 8$ | 32 |
| Up projection (`W_up`) | $4 \times 8$ | 32 |
| Down projection (`W_down`) | $8 \times 4$ | 32 |
| Final norm $\gamma$ (`norm.weight`) | $4$ | 4 |
| Output projection (`output`) | $4 \times 6$ | 24 |
| **Total** | | **220** |

Note how the parameter budget is distributed: the SwiGLU FFN accounts for $32 + 32 + 32 = 96$ parameters (44%), the attention projections account for $16 \times 4 = 64$ parameters (29%), and the embeddings plus output projection account for $24 + 24 = 48$ parameters (22%). The three RMSNorm layers contribute just 12 parameters (5%). This distribution mirrors the production model, where the FFN is the largest component per layer.

220 numbers. That is all it takes to build a (very bad) language model that is architecturally identical to LLaMA.

---

## Architecture Diagram

The complete data flow for our nano model, from token IDs to next-word probabilities:

```
  Input: "the cat"
  Token IDs: [1, 2]
        |
        v
  +-------------------------------+
  |   Token Embedding (6 x 4)    |    lookup: id -> row of embedding matrix
  |   tok_embeddings.weight       |
  +-------------------------------+
        |
        |  output: [2, 4]  (2 tokens, 4 dims each)
        v
  +=====================================================================+
  |                                                                     |
  |   Transformer Block  (x1)                                           |
  |                                                                     |
  |   +---------------------------------------------------------------+ |
  |   |                                                               | |
  |   |  x ---+---> RMSNorm -----> Multi-Head Attention (2 heads) --+ | |
  |   |       |     (gamma: 4)     Wq [4x4]  Wk [4x4]             | | |
  |   |       |                    Wv [4x4]  Wo [4x4]             | | |
  |   |       |                    + RoPE (head_dim=2)             | | |
  |   |       |                                                    | | |
  |   |       +-----------------------(+)--- residual add <--------+ | |
  |   |                                |                             | |
  |   +--------------------------------|-----------------------------+ |
  |                                    |                               |
  |   +--------------------------------|-----------------------------+ |
  |   |                                |                             | |
  |   |  x ---+---> RMSNorm -----> SwiGLU FFN (4 -> 8 -> 4) -----+ | |
  |   |       |     (gamma: 4)     W_gate [4x8]                   | | |
  |   |       |                    W_up   [4x8]                   | | |
  |   |       |                    W_down [8x4]                   | | |
  |   |       |                                                    | | |
  |   |       +-----------------------(+)--- residual add <--------+ | |
  |   |                                |                             | |
  |   +--------------------------------|-----------------------------+ |
  |                                    |                               |
  +====================================|===============================+
        |
        |  output: [2, 4]
        v
  +-------------------------------+
  |   Final RMSNorm               |    normalize before projection
  |   norm.weight (gamma: 4)      |
  +-------------------------------+
        |
        |  output: [2, 4]
        v
  +-------------------------------+
  |   Output Projection (4 x 6)  |    linear map to vocabulary size
  |   output.weight               |
  +-------------------------------+
        |
        |  output: [2, 6]  (2 tokens, 6 logits each)
        v
  +-------------------------------+
  |   Softmax                     |    convert logits to probabilities
  +-------------------------------+
        |
        |  output: [2, 6]
        v
  Probabilities over {<s>, the, cat, sat, on, mat}

  We take position 1 (the last token, "cat") to get
  the model's prediction for what comes next.
```

Every arrow in this diagram is one of three things: a matrix multiplication (the linear projections), an element-wise operation (RMSNorm scaling, SwiGLU gating, residual addition), or a simple table lookup (the embedding). There are no hidden operations and no black boxes. Let's trace each one.

---

## Step 1 --- Token Embedding

A language model cannot operate on words directly. It needs numbers. The
**embedding table** is the dictionary that converts every token in the
vocabulary into a fixed-length vector of numbers the model can manipulate.

For our Nano LLaMA the vocabulary has 6 tokens and the model dimension is 4,
so the embedding table $E$ is a $6 \times 4$ matrix --- six rows (one per
token), four columns (one per dimension):

```
E = ┌                              ┐
    │  0.2   -0.1    0.4    0.3    │  ← <s>  (id 0)
    │  0.5    0.8   -0.3    0.1    │  ← the  (id 1)
    │  0.9    0.2    0.7   -0.4    │  ← cat  (id 2)
    │ -0.1    0.6    0.3    0.8    │  ← sat  (id 3)
    │  0.3   -0.5    0.2    0.6    │  ← on   (id 4)
    │  0.4    0.3    0.5   -0.2    │  ← mat  (id 5)
    └                              ┘
```

Every value in this table is a **learned parameter** --- randomly initialized
before training, then adjusted by gradient descent until the model performs
well. There is nothing hand-crafted about these numbers.

### The lookup

Embedding is **not** a matrix multiplication. It is a plain table lookup: given
a token ID $i$, return row $i$ of $E$.

Our input is the string `"the cat"`. The tokenizer converts it to IDs
$[1,\; 2]$. We look up:

| Token | ID | Embedding vector (row of $E$) |
|-------|----|-------------------------------|
| `the` | 1  | $[0.5,\; 0.8,\; {-0.3},\; 0.1]$ |
| `cat` | 2  | $[0.9,\; 0.2,\; 0.7,\; {-0.4}]$ |

The result is a $(2,\; 4)$ matrix --- two tokens, each represented as a
4-dimensional vector:

$$
X^{(0)} = \begin{bmatrix} 0.5 & 0.8 & -0.3 & 0.1 \\ 0.9 & 0.2 & 0.7 & -0.4 \end{bmatrix}
$$

That is the entire operation. Two index lookups, no arithmetic.

### Why this matters

At this stage the vector for `"the"` is **always** $[0.5,\; 0.8,\; {-0.3},\;
0.1]$, no matter what surrounds it. The word `"the"` in `"the cat"` and
`"the"` in `"the mat"` start life as the exact same four numbers.
Context-dependence does not exist yet. It is the job of the **attention**
mechanism (Step 3) to mix information between positions so that each token's
vector begins to reflect its neighbours.

### Scale of the real model

In a full-size LLaMA (even our "small" 15M-parameter variant), the dimensions
are much larger: $\text{vocab} = 4096$ and $\text{dim} = 384$, giving an
embedding table of size $4096 \times 384 = 1{,}572{,}864$ parameters. The
operation is identical --- look up row $i$ --- just with a bigger table and
longer vectors.

---

## Step 2 --- RMSNorm (Normalizing the Vector)

Before attention gets to process our token vectors, they pass through a
normalization layer. This is a small but critical piece of housekeeping.

### Why normalize?

As vectors flow through successive layers of a neural network, their magnitudes
can drift --- some grow explosively large, others shrink toward zero. This makes
training unstable: gradients blow up or vanish, and the optimizer struggles to
find a good direction. Normalization reins the magnitudes in, keeping every
layer's input on a predictable scale.

LLaMA uses **RMSNorm** (Root Mean Square Normalization), a variant that is
simpler than the classic LayerNorm. RMSNorm does **not** center the values by
subtracting the mean, and it has **no bias term**. It only rescales:

$$
\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\dfrac{1}{d}\displaystyle\sum_{i=1}^{d} x_i^2 \;+\; \epsilon}} \;\cdot\; \boldsymbol{\gamma}
$$

where:

- $d = 4$ --- the model dimension,
- $\epsilon = 10^{-5}$ --- a tiny constant that prevents division by zero,
- $\boldsymbol{\gamma}$ --- a **learned** per-dimension scale vector, initialized to $[1,\; 1,\; 1,\; 1]$.

### Worked example: normalizing `"the"`

The embedding for `"the"` is $\mathbf{x} = [0.5,\; 0.8,\; {-0.3},\; 0.1]$.

**1. Square each element:**

$$
\mathbf{x}^2 = [0.25,\; 0.64,\; 0.09,\; 0.01]
$$

**2. Compute the mean of the squares:**

$$
\text{MS} = \frac{0.25 + 0.64 + 0.09 + 0.01}{4} = \frac{0.99}{4} = 0.2475
$$

**3. Take the square root (adding $\epsilon$ for safety):**

$$
\text{RMS} = \sqrt{0.2475 + 0.00001} \approx 0.4975
$$

**4. Divide each element by the RMS:**

$$
\hat{\mathbf{x}} = \frac{\mathbf{x}}{0.4975} = \left[\frac{0.5}{0.4975},\; \frac{0.8}{0.4975},\; \frac{-0.3}{0.4975},\; \frac{0.1}{0.4975}\right] \approx [1.005,\; 1.608,\; {-0.603},\; 0.201]
$$

**5. Multiply by the learned scale $\boldsymbol{\gamma} = [1,\; 1,\; 1,\; 1]$:**

$$
\text{RMSNorm}(\mathbf{x}) = \hat{\mathbf{x}} \cdot \boldsymbol{\gamma} \approx [1.005,\; 1.608,\; {-0.603},\; 0.201]
$$

Because $\boldsymbol{\gamma}$ is initialized to all ones, it has no effect at
the start. During training, the model learns to scale certain dimensions up or
down, giving it fine-grained control over which features matter most.

### What just happened, geometrically?

RMSNorm **preserves the direction** of the vector but **standardizes its
magnitude**. The original vector $[0.5,\; 0.8,\; {-0.3},\; 0.1]$ and the
normalized vector $[1.005,\; 1.608,\; {-0.603},\; 0.201]$ point in the same
direction in 4-dimensional space --- the ratios between components are
unchanged. Only the overall length has been adjusted so that the root mean
square of the components is approximately 1.

A useful mental model: "I do not care how loud this signal is, just tell me
which direction it points."

### Pre-norm vs. post-norm

LLaMA applies RMSNorm **before** each sub-layer (attention, feed-forward),
not after. This is called **pre-normalization**. The alternative ---
normalizing after the sub-layer --- was used in the original Transformer paper,
but pre-norm has become the standard in modern LLMs because it creates a
cleaner gradient path through the residual connections. The residual stream
carries un-normalized values, and each sub-layer reads a freshly normalized
copy. This means gradients can flow backward through the residual additions
without being distorted by normalization, making training more stable.

After this step our two token vectors look like:

| Token | Before RMSNorm | After RMSNorm |
|-------|---------------|--------------|
| `the` | $[0.5,\; 0.8,\; {-0.3},\; 0.1]$ | $\approx [1.005,\; 1.608,\; {-0.603},\; 0.201]$ |
| `cat` | $[0.9,\; 0.2,\; 0.7,\; {-0.4}]$ | (same procedure, different numbers) |

These normalized vectors are what the attention mechanism will receive next.

---

## Step 3 --- Attention: How Tokens Talk to Each Other

Up to now, each token has lived in isolation. "the" has a 4D vector; "cat" has a 4D vector. Neither knows the other exists. Attention is the mechanism that changes this.

Consider the word "the." In "the cat sat on the mat," the first "the" should eventually point toward *cat-ness* while the second should point toward *mat-ness*. They start with the same embedding, but after attention they diverge, because each "the" absorbs different context from its neighbors.

Every token asks a single question: **"What information from the tokens I can see is relevant to me?"** To answer it, the model learns three projections:

- **Q** (Query) -- "What am I looking for?"
- **K** (Key) -- "What do I contain that others might want?"
- **V** (Value) -- "If someone attends to me, what information do I hand over?"

These are not metaphors. They are literal learned linear transformations, and we will trace every number through them.

---

### 3a. Projecting into Q, K, V and Splitting into Heads

Our two normalized vectors entering attention are:

$$\mathbf{x}_{\text{the}} = [0.5,\; 0.8,\; -0.3,\; 0.1], \qquad \mathbf{x}_{\text{cat}} = [-0.2,\; 0.7,\; 0.4,\; 0.6]$$

The model holds three weight matrices $W_Q$, $W_K$, $W_V$, each of shape $4 \times 4$. A token's query vector is simply:

$$\mathbf{q} = \mathbf{x} \cdot W_Q$$

and likewise for $\mathbf{k}$ and $\mathbf{v}$. Let us pick a concrete $W_Q$ and trace one multiplication in full:

$$W_Q = \begin{bmatrix} 0.3 & 0.1 & -0.2 & 0.4 \\ 0.5 & -0.3 & 0.6 & 0.1 \\ -0.1 & 0.4 & 0.2 & -0.3 \\ 0.2 & 0.6 & -0.1 & 0.5 \end{bmatrix}$$

For "the" ($\mathbf{x} = [0.5, 0.8, -0.3, 0.1]$):

$$q_0 = 0.5 \times 0.3 + 0.8 \times 0.5 + (-0.3) \times (-0.1) + 0.1 \times 0.2 = 0.15 + 0.40 + 0.03 + 0.02 = 0.60$$

$$q_1 = 0.5 \times 0.1 + 0.8 \times (-0.3) + (-0.3) \times 0.4 + 0.1 \times 0.6 = 0.05 - 0.24 - 0.12 + 0.06 = -0.25$$

$$q_2 = 0.5 \times (-0.2) + 0.8 \times 0.6 + (-0.3) \times 0.2 + 0.1 \times (-0.1) = -0.10 + 0.48 - 0.06 - 0.01 = 0.31$$

$$q_3 = 0.5 \times 0.4 + 0.8 \times 0.1 + (-0.3) \times (-0.3) + 0.1 \times 0.5 = 0.20 + 0.08 + 0.09 + 0.05 = 0.42$$

$$\mathbf{q}_{\text{the}} = [0.60,\; -0.25,\; 0.31,\; 0.42]$$

This 4D query vector is then **split** across our 2 attention heads. Each head gets a 2D slice:

| | Head 0 (dims 0--1) | Head 1 (dims 2--3) |
|---|---|---|
| $\mathbf{q}_{\text{the}}$ | $[0.60, -0.25]$ | $[0.31, 0.42]$ |

The same matrix multiply and split happen for $W_K$ and $W_V$, producing per-head keys and values. After all three projections, suppose we have for **Head 0**:

| Token | $\mathbf{q}$ | $\mathbf{k}$ | $\mathbf{v}$ |
|-------|-----|-----|-----|
| "the" | $[0.8,\; 0.6]$ | $[0.7,\; 0.5]$ | $[0.2,\; 0.6]$ |
| "cat" | $[0.3,\; 0.9]$ | $[0.4,\; 0.8]$ | $[0.5,\; 0.3]$ |

These are content-only vectors. They encode *what* the token is, but not *where* it sits in the sequence. A "cat" at position 1 and a "cat" at position 47 would produce the exact same $\mathbf{q}$, $\mathbf{k}$, $\mathbf{v}$. We need to fix that.

---

### 3b. RoPE: Position Through Rotation

This is the most elegant idea in the LLaMA architecture, and with our head_dim=2 model, we can see every moving part.

#### The problem

After projection, $\mathbf{q}_{\text{the}}$ and $\mathbf{q}_{\text{cat}}$ carry no positional information whatsoever. If we computed attention scores right now, a token at position 0 and the same token at position 100 would behave identically. The model could never learn word-order-sensitive patterns like "the adjective usually comes right before the noun."

#### The solution: Rotary Position Embeddings (RoPE)

Instead of *adding* a position signal to the vector (as in the original Transformer), RoPE **rotates** it. Each position gets a unique rotation angle, and the rotation is applied to the query and key vectors before they interact.

#### The frequency

With head_dim $= 2$, there is exactly one 2D plane to rotate in, governed by a single frequency:

$$\theta_0 = \frac{1}{10000^{\,0/2}} = \frac{1}{10000^0} = 1.0$$

At sequence position $m$, the rotation angle is:

$$\phi = m \cdot \theta_0 = m \times 1.0 = m \text{ radians}$$

That's it. Position 0 rotates by 0 radians. Position 1 rotates by 1 radian ($\approx 57.3\degree$). Position 2 by 2 radians. And so on.

#### The rotation matrix

A 2D rotation by angle $\phi$ is:

$$R(\phi) = \begin{bmatrix} \cos\phi & -\sin\phi \\ \sin\phi & \cos\phi \end{bmatrix}$$

Applied to a vector $[a, b]$:

$$R(\phi) \begin{bmatrix} a \\ b \end{bmatrix} = \begin{bmatrix} a\cos\phi - b\sin\phi \\ a\sin\phi + b\cos\phi \end{bmatrix}$$

This preserves the length of the vector (rotations never stretch or shrink) while encoding position purely through angle.

#### Tracing the numbers

**Position 0** -- "the": rotation by $0$ radians.

$\cos(0) = 1.000, \quad \sin(0) = 0.000$

$$\mathbf{q}_{\text{the}}^{\text{rot}} = \begin{bmatrix} 0.8 \times 1.000 - 0.6 \times 0.000 \\ 0.8 \times 0.000 + 0.6 \times 1.000 \end{bmatrix} = \begin{bmatrix} 0.800 \\ 0.600 \end{bmatrix}$$

No change -- rotating by zero does nothing. The first token keeps its original direction.

$$\mathbf{k}_{\text{the}}^{\text{rot}} = \begin{bmatrix} 0.7 \times 1.000 - 0.5 \times 0.000 \\ 0.7 \times 0.000 + 0.5 \times 1.000 \end{bmatrix} = \begin{bmatrix} 0.700 \\ 0.500 \end{bmatrix}$$

Again unchanged.

**Position 1** -- "cat": rotation by $1$ radian ($\approx 57.3\degree$).

$\cos(1) \approx 0.540, \quad \sin(1) \approx 0.841$

$$\mathbf{q}_{\text{cat}}^{\text{rot}} = \begin{bmatrix} 0.3 \times 0.540 - 0.9 \times 0.841 \\ 0.3 \times 0.841 + 0.9 \times 0.540 \end{bmatrix} = \begin{bmatrix} 0.162 - 0.757 \\ 0.252 + 0.486 \end{bmatrix} = \begin{bmatrix} -0.595 \\ 0.738 \end{bmatrix}$$

The query vector for "cat" has been rotated substantially. It now points in a completely different direction than the unrotated $[0.3, 0.9]$, but it has the same length: $\sqrt{0.3^2 + 0.9^2} = \sqrt{0.90} \approx 0.949$ and $\sqrt{(-0.595)^2 + 0.738^2} = \sqrt{0.354 + 0.545} = \sqrt{0.899} \approx 0.949$. Length preserved, direction changed.

$$\mathbf{k}_{\text{cat}}^{\text{rot}} = \begin{bmatrix} 0.4 \times 0.540 - 0.8 \times 0.841 \\ 0.4 \times 0.841 + 0.8 \times 0.540 \end{bmatrix} = \begin{bmatrix} 0.216 - 0.673 \\ 0.336 + 0.432 \end{bmatrix} = \begin{bmatrix} -0.457 \\ 0.768 \end{bmatrix}$$

#### The key insight: rotation encodes *relative* position

Here is where the elegance reveals itself. When we compute the attention score between a query at position $m$ and a key at position $n$, we take their dot product:

$$\text{score} = \bigl(R(m)\,\mathbf{q}\bigr)^T \bigl(R(n)\,\mathbf{k}\bigr)$$

Expand the transpose:

$$= \mathbf{q}^T \, R(m)^T \, R(n) \, \mathbf{k}$$

Now the crucial property: rotation matrices are **orthogonal**, meaning $R(m)^T = R(-m)$. And composing two rotations just adds their angles: $R(-m) \, R(n) = R(n - m)$. Therefore:

$$\boxed{\text{score} = \mathbf{q}^T \, R(n - m) \, \mathbf{k}}$$

The absolute positions $m$ and $n$ have vanished. Only their difference $(n - m)$ remains. This means:

- A query at position 3 attending to a key at position 1 produces the **same geometric relationship** as a query at position 50 attending to a key at position 48. Both have a relative distance of 2.
- The model never needs to learn "attend to position 3." It learns **"attend to the token 2 steps back"** -- a pattern that generalizes across the entire sequence.
- Because rotation preserves vector lengths, the magnitude of the attention score reflects only content similarity and relative distance, never absolute position.

This is why the same LLaMA model can handle sequences of varying length without retraining -- the position encoding is inherently relative.

#### Scaling to the real model

In our nano model, head_dim $= 2$ gives us a single rotation plane with a single frequency $\theta_0 = 1.0$. In the real LLaMA with head_dim $= 64$, there are 32 independent 2D rotation planes, each with its own frequency:

$$\theta_i = \frac{1}{10000^{\,2i/64}}, \quad i = 0, 1, \ldots, 31$$

The frequencies form a geometric progression from $\theta_0 = 1.0$ (fast rotation, encoding fine-grained local position) down to $\theta_{31} = 1/10000 \approx 0.0001$ (extremely slow rotation, encoding long-range position). Together they give the model a rich, multi-scale sense of where every token sits relative to every other. Our single-frequency nano model captures the full principle in its simplest form.

---

### 3c. Computing Attention Scores

With rotated queries and keys in hand, we can compute how much each token attends to every other. The raw score between a query and key is their dot product, scaled by $\frac{1}{\sqrt{\text{head\_dim}}}$ to keep gradients well-behaved:

$$\text{score}(i \to j) = \frac{\mathbf{q}_i^{\text{rot}} \cdot \mathbf{k}_j^{\text{rot}}}{\sqrt{d_{\text{head}}}}$$

For Head 0, let us compute all scores for "cat" at position 1 (it can see both tokens):

**score(cat $\to$ the)**:

$$\frac{[-0.595,\; 0.738] \cdot [0.700,\; 0.500]}{\sqrt{2}} = \frac{(-0.595)(0.700) + (0.738)(0.500)}{1.414}$$

$$= \frac{-0.417 + 0.369}{1.414} = \frac{-0.048}{1.414} \approx -0.034$$

**score(cat $\to$ cat)**:

$$\frac{[-0.595,\; 0.738] \cdot [-0.457,\; 0.768]}{\sqrt{2}} = \frac{(-0.595)(-0.457) + (0.738)(0.768)}{1.414}$$

$$= \frac{0.272 + 0.567}{1.414} = \frac{0.839}{1.414} \approx 0.593$$

Token "cat" has a much higher raw score for itself ($0.593$) than for "the" ($-0.034$). The rotated query-key geometry is telling us that, for this head, "cat" finds itself more relevant than "the."

#### The causal mask

LLaMA is an autoregressive (left-to-right) model. When predicting the next token, a token must not peek at future tokens -- that would be cheating. We enforce this with a **causal mask**: any attention score where the key position is *after* the query position gets set to $-\infty$:

```
           the     cat
  the  [  score   -inf  ]   <-- "the" (pos 0) cannot see "cat" (pos 1)
  cat  [  score   score ]   <-- "cat" (pos 1) can see both
```

For "the" at position 0: it can only attend to itself -- it sees a single score.
For "cat" at position 1: it sees both scores as computed above.

#### Softmax: turning scores into weights

Raw scores become proper probability weights (non-negative, summing to 1) via softmax. For "cat" attending:

$$\text{softmax}([-0.034,\; 0.593]) = \frac{[\,e^{-0.034},\; e^{0.593}\,]}{e^{-0.034} + e^{0.593}}$$

$$= \frac{[0.967,\; 1.809]}{0.967 + 1.809} = \frac{[0.967,\; 1.809]}{2.776} \approx [0.348,\; 0.652]$$

The attention weights for "cat":

| Attending to | Weight |
|---|---|
| "the" | $0.348$ (34.8%) |
| "cat" | $0.652$ (65.2%) |

"cat" pays about two-thirds of its attention to itself and one-third to "the." This ratio is entirely determined by the learned $W_Q$ and $W_K$ matrices and the RoPE-induced rotation -- different weights would yield a different split.

---

### 3d. Weighted Sum of Values

The attention weights tell each token *how much* to listen to each source. The values $\mathbf{v}$ tell it *what* to copy. The output is the weighted sum:

$$\mathbf{o}_{\text{cat}} = 0.348 \times \mathbf{v}_{\text{the}} + 0.652 \times \mathbf{v}_{\text{cat}}$$

$$= 0.348 \times [0.2,\; 0.6] + 0.652 \times [0.5,\; 0.3]$$

$$= [0.070,\; 0.209] + [0.326,\; 0.196]$$

$$= [0.396,\; 0.405]$$

This $[0.396, 0.405]$ is **Head 0's output** for "cat." Meanwhile, Head 1 has been running the same computation independently with its own $W_Q$, $W_K$, $W_V$ slices and may produce a very different 2D vector -- perhaps $[0.112, -0.287]$.

The two head outputs are **concatenated** back into a single 4D vector:

$$\mathbf{o}_{\text{cat}}^{\text{concat}} = [\underbrace{0.396,\; 0.405}_{\text{Head 0}},\; \underbrace{0.112,\; -0.287}_{\text{Head 1}}]$$

This concatenated vector passes through one final linear projection $W_O$ (a $4 \times 4$ matrix) that mixes information across heads:

$$\mathbf{o}_{\text{cat}}^{\text{final}} = \mathbf{o}_{\text{cat}}^{\text{concat}} \cdot W_O$$

The result is a 4D vector -- the same dimensionality we started with -- ready to be added back to the residual stream.

---

**What just happened?** Before attention, the vector for "cat" encoded only the meaning of the word "cat" in isolation. After attention, it is a blend: roughly two-thirds cat-information and one-third the-information, mixed through learned projections. The vector for "cat" now *knows it was preceded by "the."* This is the fundamental operation of the transformer -- each token's representation becomes a context-aware summary of everything it can see, weighted by learned relevance. Stack enough of these layers, and the model builds representations rich enough to predict what word comes next.

---

## Step 4 --- The Residual Connection

After attention computes its output, we do **not** replace the original vector.
We **add** the attention output to it:

$$
\mathbf{x} = \mathbf{x}_{\text{original}} + \text{Attention}(\text{RMSNorm}(\mathbf{x}_{\text{original}}))
$$

Concretely, the embedding for `"cat"` was $\mathbf{x}_{\text{cat}} = [0.9,\; 0.2,\; 0.7,\; {-0.4}]$.
Suppose the attention sub-layer produced the output $[0.1,\; {-0.05},\; 0.15,\; 0.08]$.
The new vector is their element-wise sum:

$$
\mathbf{x}_{\text{cat}}^{\text{new}} = [0.9 + 0.1,\; 0.2 + (-0.05),\; 0.7 + 0.15,\; {-0.4} + 0.08] = [1.0,\; 0.15,\; 0.85,\; {-0.32}]
$$

This pattern --- **add, never overwrite** --- is the single most important
structural choice in the transformer. The running sum of vectors through the
network is called the **residual stream**, and it acts as a highway for
information. Each sub-layer (attention, feed-forward) reads from the stream,
computes a small correction, and writes that correction back by addition. The
original embedding is never destroyed; it is still present, summed together with
every refinement made along the way.

Why does this matter for training? The gradient of $x + y$ with respect to $x$
is always exactly 1. No matter how deep the network is, gradients flow backward
through the addition operations without shrinking or exploding. This is what
makes deep transformers trainable at all. Without residual connections, a 32-layer
model would face crippling vanishing gradients.

After a single layer, the output for each position is:

$$
\text{output} = \mathbf{x}_{\text{embed}} + \Delta_{\text{attn}} + \Delta_{\text{ffn}}
$$

three additive terms. A model with 8 layers has $1 + 2 \times 8 = 17$ additive
terms: the original embedding plus one attention delta and one FFN delta per
layer. The final vector is a sum of 17 independent contributions, each one
computed by a different sub-layer. The residual stream carries all of them.

---

## Step 5 --- SwiGLU FFN: The Thinking Step

If attention is "gathering information from context," the feed-forward network
is "thinking about what you have gathered." It processes each token's vector
**independently** --- there is no cross-token communication here. Every position
goes through the same function with the same weights, but each one gets its own
output because each one has a different input vector (different thanks to
attention).

### 5a. The Three Projections

The LLaMA feed-forward network uses **SwiGLU**, a gated activation introduced
by Noam Shazeer in 2020. It looks like this:

$$
\text{FFN}(\mathbf{x}) = \bigl(\text{SiLU}(\mathbf{x}\, W_{\text{gate}}) \odot (\mathbf{x}\, W_{\text{up}})\bigr) \cdot W_{\text{down}}
$$

There are three weight matrices:

| Matrix | Shape | Role |
|:---|:---:|:---|
| $W_{\text{gate}}$ | $4 \times 8$ | Projects to "gate" space --- decides what to keep |
| $W_{\text{up}}$ | $4 \times 8$ | Projects to "content" space --- computes the actual information |
| $W_{\text{down}}$ | $8 \times 4$ | Projects back to model dimension |

The symbol $\odot$ denotes element-wise (Hadamard) multiplication, and
$\text{SiLU}(x) = x \cdot \sigma(x)$ where $\sigma$ is the sigmoid function
$\sigma(x) = 1 / (1 + e^{-x})$.

The data flow is: start with a 4D vector, project it to **two** 8D vectors
in parallel (gate and up), combine them element-wise, then project the 8D
result back down to 4D. The expansion from 4 to 8 dimensions gives the network
more room to represent intermediate features before compressing back.

### 5b. The Gating Mechanism --- A Worked Example

This is the key insight of SwiGLU. Let us trace it with concrete numbers.

Suppose for the `"cat"` position, after the residual update, we have
$\mathbf{x} = [1.0,\; 0.15,\; 0.85,\; {-0.32}]$. After RMSNorm (applied
before the FFN, just as it was before attention) and the two projections, we get
two 8-dimensional vectors:

$$
\text{gate\_pre} = \mathbf{x}\, W_{\text{gate}} = [1.2,\; {-2.0},\; 0.5,\; 3.1,\; {-0.8},\; 0.0,\; 1.7,\; {-1.5}]
$$

$$
\text{up} = \mathbf{x}\, W_{\text{up}} = [0.4,\; 0.9,\; {-0.3},\; 0.7,\; 0.5,\; 1.2,\; {-0.8},\; 0.6]
$$

Now apply SiLU to each element of `gate_pre`, then multiply element-wise with
`up`:

| gate\_pre | SiLU(gate\_pre) | up | gated = SiLU $\times$ up | What happened |
|---:|---:|---:|---:|:---|
| 1.2 | 0.849 | 0.4 | 0.340 | Mostly passed through |
| $-2.0$ | $-0.238$ | 0.9 | $-0.214$ | Heavily attenuated + sign flipped |
| 0.5 | 0.311 | $-0.3$ | $-0.093$ | Partially passed |
| 3.1 | 2.969 | 0.7 | 2.078 | **Amplified** (gate $> 1$) |
| $-0.8$ | $-0.263$ | 0.5 | $-0.132$ | Attenuated |
| 0.0 | 0.000 | 1.2 | 0.000 | **Completely blocked** |
| 1.7 | 1.338 | $-0.8$ | $-1.070$ | Amplified |
| $-1.5$ | $-0.277$ | 0.6 | $-0.166$ | Attenuated |

Look at what the gate learned to do. Dimension 6 had a gate value of exactly
0.0, which means $\text{SiLU}(0) = 0$, completely zeroing out whatever the
content pathway computed (1.2). The model decided that particular piece of
information is irrelevant. Dimension 4 had a gate value of 3.1, and since
$\text{SiLU}(3.1) \approx 2.97$, the content (0.7) was amplified more than
threefold to 2.078. The model found that information important and boosted it.

This selective filtering --- keep this, suppress that, amplify the other ---
is what makes SwiGLU more expressive than a simple ReLU activation. The gate and
content pathways are computed from the same input but through different weight
matrices, so the network can learn independent criteria for "what to compute"
and "whether to use it."

### 5c. SiLU vs. ReLU

To see why LLaMA chose SiLU over the classic ReLU, compare them side by side:

```
Input x:  -2.0   -1.0    0.0    0.5    1.0    2.0    3.0
ReLU(x):   0.0    0.0    0.0    0.5    1.0    2.0    3.0   <-- hard cutoff at 0
SiLU(x): -0.24  -0.27   0.00   0.31   0.73   1.76   2.86  <-- smooth, allows small negatives
```

Three properties stand out:

- **Smooth everywhere.** ReLU has a sharp kink at $x = 0$ where its derivative
  jumps from 0 to 1 instantly. SiLU is smooth, so gradients never abruptly
  vanish at the kink the way they do with ReLU.
- **Slightly non-monotonic.** SiLU dips below zero for negative inputs (its
  minimum is about $-0.28$ near $x \approx -1.28$). This lets small negative
  gate values produce small negative outputs, giving the network a richer
  signal than the flat zero that ReLU would produce.
- **Can exceed its input.** For large positive values, $\text{SiLU}(x)$
  approaches $x$ from below but for moderate values it can act as a slight
  amplifier. Combined with the gating mechanism, this means the gate can
  boost content values, not just pass or block them.

### 5d. Projection Back Down

The 8-dimensional gated vector is now projected back to 4 dimensions by
$W_{\text{down}}$ (an $8 \times 4$ matrix):

$$
\Delta_{\text{ffn}} = \text{gated} \cdot W_{\text{down}}
$$

This gives us a 4D vector --- the FFN's contribution for this position.

Then comes another **residual connection**, exactly like the one after attention:

$$
\mathbf{x}_{\text{final}} = \mathbf{x}_{\text{post\text{-}attn}} + \Delta_{\text{ffn}}
$$

The post-attention vector and the FFN delta are added together. The residual
stream now carries three layers of information: the original embedding, the
attention refinement, and the FFN refinement.

---

## Step 6 --- From Vector to Word

We have traced `"the cat"` through embedding, normalization, attention, the
residual connection, and the feed-forward network. The transformer block is
complete. Now we need to convert the 4-dimensional vector at each position back
into a prediction over the vocabulary.

### 6a. Final Normalization

One more RMSNorm is applied after all transformer blocks, using the same formula
as before:

$$
\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \boldsymbol{\gamma}
$$

This final normalization has its own learned scale vector $\boldsymbol{\gamma}$
(4 parameters), separate from the ones inside the transformer block. It ensures
the vectors are on a stable scale before the output projection.

### 6b. Output Projection

The output matrix $W_{\text{output}}$ has shape $4 \times 6$: it maps from the
4D representation space to 6 scores, one per token in the vocabulary. These
scores are called **logits**:

$$
\text{logits} = \mathbf{h} \cdot W_{\text{output}}
$$

Suppose after the final RMSNorm, the vector at the `"cat"` position is
$\mathbf{h} = [1.2,\; {-0.3},\; 0.8,\; 0.5]$. The matrix multiplication
produces one number per vocabulary token:

$$
\text{logits} = \mathbf{h} \cdot W_{\text{output}} \;\to\; [{-0.4},\; 0.2,\; {-0.1},\; 1.8,\; 0.3,\; 0.9]
$$

$$
\phantom{\text{logits} = \mathbf{h} \cdot W_{\text{output}} \;\to\;}\; \underset{\texttt{<s>}}{-0.4} \quad \underset{\texttt{the}}{0.2} \quad \underset{\texttt{cat}}{-0.1} \quad \underset{\texttt{sat}}{1.8} \quad \underset{\texttt{on}}{0.3} \quad \underset{\texttt{mat}}{0.9}
$$

The logit for `"sat"` (1.8) is the highest, meaning the model is most confident
about that token --- but logits are raw scores, not probabilities. They can be
negative, and they do not sum to 1. We need one more step.

### 6c. Softmax --- Probabilities

The **softmax** function converts logits into a proper probability distribution:

$$
P(\text{token}_i) = \frac{e^{\text{logit}_i}}{\displaystyle\sum_{j} e^{\text{logit}_j}}
$$

Working through the numbers:

**1. Exponentiate each logit:**

$$
e^{-0.4} = 0.670 \quad e^{0.2} = 1.221 \quad e^{-0.1} = 0.905 \quad e^{1.8} = 6.050 \quad e^{0.3} = 1.350 \quad e^{0.9} = 2.460
$$

**2. Sum the exponentials:**

$$
0.670 + 1.221 + 0.905 + 6.050 + 1.350 + 2.460 = 12.656
$$

**3. Divide each by the sum:**

| Token | Logit | $e^{\text{logit}}$ | Probability |
|:---|---:|---:|---:|
| `<s>` | $-0.4$ | 0.670 | 5.3% |
| `the` | $0.2$ | 1.221 | 9.6% |
| `cat` | $-0.1$ | 0.905 | 7.1% |
| **`sat`** | **$1.8$** | **6.050** | **47.8%** |
| `on` | $0.3$ | 1.350 | 10.7% |
| `mat` | $0.9$ | 2.460 | 19.4% |

The model predicts **`sat`** with 47.8% confidence. Given the input `"the cat"`,
the most likely next word is `"sat"` --- the model is learning the pattern
`"the cat sat on mat"`.

### 6d. Cross-Entropy Loss

During training, we know the correct next token is `"sat"` (token ID 3). The
**cross-entropy loss** measures how surprised the model is by the correct
answer:

$$
\mathcal{L} = -\log P(\texttt{sat}) = -\log(0.478) \approx 0.738
$$

To put this in perspective:

- **Perfect prediction:** $P(\texttt{sat}) = 1.0 \;\to\; \mathcal{L} = -\log(1) = 0$ (zero surprise)
- **Uniform guessing:** $P(\texttt{sat}) = 1/6 \approx 0.167 \;\to\; \mathcal{L} = -\log(0.167) \approx 1.79$ (maximum surprise for a uniform distribution over 6 tokens)
- **Our model:** $\mathcal{L} = 0.738$ --- better than random, but plenty of room to improve

This single number, 0.738, is the signal that drives all of training. It flows
backward through every operation we have traced --- softmax, output projection,
RMSNorm, residual addition, SwiGLU, residual addition, attention, RMSNorm,
embedding --- adjusting all 220 parameters by a tiny amount to make
$P(\texttt{sat})$ a little higher next time. That process is **backpropagation**,
and it runs for thousands of iterations until the model converges.

---

## The Full Picture

Here is the complete data flow we have traced, from token IDs to prediction:

```
"the cat" --> [1, 2]
    |
    v  Embedding lookup (6x4 table)
[0.5, 0.8, -0.3, 0.1]    <-- "the" (4D)
[0.9, 0.2,  0.7, -0.4]   <-- "cat" (4D)
    |
    v  RMSNorm (normalize, scale by gamma)
    v  Q, K, V projections (three 4x4 matrices)
    v  RoPE rotation (encode positions 0 and 1)
    v  Attention scores + causal mask + softmax + weighted sum
    v  Output projection Wo (4x4)
    |
    v  + residual  (add back original embedding)
    |
    v  RMSNorm (normalize, scale by gamma)
    v  SwiGLU FFN: gate(4->8) * up(4->8) --> down(8->4)
    |
    v  + residual  (add back post-attention vector)
    |
    v  Final RMSNorm (normalize, scale by gamma)
    v  Output projection (4->6 logits)
    v  Softmax (logits -> probabilities)
    |
    v  P(sat) = 47.8%   <-- prediction for position after "cat"
```

Every arrow is one of three things: a matrix multiplication (the linear
projections), an element-wise operation (normalization, gating, residual
addition), or a table lookup (the embedding). There are no hidden operations.
The entire model is a deterministic sequence of arithmetic on 220 numbers.

---

## From Nano to Real

Our nano model is absurdly small, but the architecture is identical to
production-scale LLaMA. Here is how the configurations compare:

| | Nano LLaMA | llama-vc (15M) | LLaMA 2 7B |
|:---|---:|---:|---:|
| dim | 4 | 384 | 4,096 |
| layers | 1 | 8 | 32 |
| heads | 2 | 6 | 32 |
| KV heads | 2 | 2 | 32 |
| head\_dim | 2 | 64 | 128 |
| hidden\_dim | 8 | 1,024 | 11,008 |
| vocab | 6 | 4,096 | 32,000 |
| context | 8 | 512 | 4,096 |
| parameters | **220** | **15.7M** | **6.7B** |
| RoPE freqs | 1 pair | 32 pairs | 64 pairs |

The jump from 220 parameters to 6.7 billion is a factor of 30 million. And yet
the same code runs all three models. The `LLaMA` class, the `Attention` module,
the `FeedForward` module, the `RMSNorm` layer --- they are all identical. Only
the numbers in `ModelConfig` change. Going from nano to 15M means setting
`dim=384, n_layers=8, n_heads=6, n_kv_heads=2, hidden_dim=1024, vocab_size=4096`.
Going from 15M to 7B means setting bigger numbers still. The architecture does
not change.

Every operation we traced by hand --- the embedding lookup, the RMSNorm
rescaling, the RoPE rotation of query-key pairs, the dot-product attention with
causal masking, the SwiGLU gating, the residual additions, the softmax over
logits --- works exactly the same way at every scale. The matrices get bigger.
The vectors get longer. The rotations happen in 64 pairs instead of 1. But the
math is the same math.

You now understand every operation inside a large language model. The rest is
just scale.
