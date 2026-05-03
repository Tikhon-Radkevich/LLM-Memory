# Memory-Augmented Transformer (MPNet Attention Modification)

## Goal

Extend a pretrained Transformer (MPNet: `sentence-transformers/all-mpnet-base-v2`) with a **persistent internal memory mechanism**.

The goal is to enable the model to:

* accumulate information across multiple sequential calls
* use this accumulated memory to improve question answering
* eliminate reliance on external retrieval systems

---

## Core Idea

Introduce a set of **memory tokens** inside the last attention layer of the Transformer.

These tokens:

* interact with input tokens via self-attention
* store compressed contextual information across calls
* are updated after each forward pass
* persist across multiple model invocations

---

## Architecture Modification

Modify only the **last encoder layer attention block**.

### Input to attention

```
[input_tokens] + [memory_tokens]
```

### Memory

* Shape: `(16, 768)`
* Initialized once per trajectory using random normal distribution
* Fixed size hyperparameter

### Attention behavior

* Standard multi-head attention is preserved
* Memory tokens participate fully in attention
* No structural change to Q/K/V mechanism

---

## Memory Initialization

```python
import torch

torch.manual_seed(42)

MEMORY_SIZE = 16
HIDDEN_SIZE = 768

INITIAL_MEMORY_EMBEDDINGS = torch.randn(MEMORY_SIZE, HIDDEN_SIZE) * 0.02
```

Memory is:

* randomly initialized once for model
* not zero-initialized (to avoid symmetry collapse)

---

## Memory Update Rule

After each forward pass:

```
M_{t+1} = memory_output_from_attention
```

No EMA or residual mixing in the minimal version.

---

## Model Interface

```python
outputs, updated_memory = model(input_ids, memory)
```

Where:

* `input_ids`: tokenized question or question+answer
* `memory`: tensor of shape `(16, 768)`

---

## Training Data Structure

Each training sample is a **short trajectory**:

```(Q1, A1), (Q2, A2), (Q3, A3)```

No replay buffer in the initial version.

---

## Training Loop (Minimal Version)

For each trajectory:

```
M = INITIAL_MEMORY_EMBEDDINGS

for t in range(T):

    # 1. Memory update step (observe Q+A)
    _, M = model(Q_t + A_t, M)

    # 2. Prediction step (question only)
    pred = model(Q_t, M)

    # 3. Target representation
    target = model(Q_t + A_t).detach()

    # 4. Loss
    loss += cosine_similarity(pred, target)

backprop once per trajectory
```

---

## Loss Function

* Primary: cosine similarity between CLS embeddings
* No token-level loss in the initial version

---

## Training Strategy

* No batching (initially)
* No replay buffer
* No curriculum
* Sequence length fixed to 2–3 steps
* Full trajectory backpropagation

---

## Design Rationale

This system tests whether:

* Transformer attention can act as a memory accumulator
* memory tokens can store reusable context across calls
* CLS representation improves with persistent memory state

---

## Expected Behavior

Short-term:

* works for 2–3 step memory chains

Long-term:

* may degrade without stabilization techniques

---

## Future Extensions (not included)

* batching of trajectories
* memory normalization
* adapter layers in attention
* longer memory horizons (10–20 steps)
* replay buffer for state transitions
* contrastive objectives
* memory compression mechanisms

---

## Current Status

Minimal experimental prototype for validating internal memory in Transformer attention without external retrieval systems.
