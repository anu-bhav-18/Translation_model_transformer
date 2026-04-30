
# Transformer-Based English → Hindi Translation

## Overview

This notebook focuses on building and training a **Transformer-based machine translation model** from scratch using PyTorch. The goal is to translate English sentences into Hindi using a custom dataset and tokenizer.

The implementation follows the core ideas from the original Transformer architecture:

* Encoder–Decoder structure
* Multi-head self-attention
* Positional encoding
* Teacher forcing during training

---

## Pipeline

### 1. Data Preparation

* Dataset: English–Hindi sentence pairs (CSV format)
* Removed null values and ensured proper string formatting
* Combined both languages to train a shared tokenizer

---

### 2. Tokenization

* Used **Byte Pair Encoding (BPE)** tokenizer
* Vocabulary size: 3000
* Special tokens:

  * `[PAD]`, `[UNK]`, `[SOS]`, `[EOS]`, `[BOS]`
* Each sentence is:

  * Tokenized
  * Padded/truncated to fixed length (`max_len = 20`)

---

### 3. Model Architecture

* Embedding layer
* Positional Encoding (sinusoidal)
* Transformer (Encoder + Decoder)
* Linear output layer (projection to vocabulary)

**Key Hyperparameters:**

* `d_model = 512`
* `n_heads = 8`
* `num_layers = 6`
* `feedforward_dim = 2048`

---

### 4. Training Strategy

* Input:

  * Source: full English sentence
  * Target input: Hindi sentence shifted right
  * Target output: Hindi sentence shifted left
* Loss:

  * CrossEntropyLoss
* Optimization:

  * Adam optimizer
* Masking:

  * Causal mask applied in decoder

---

### 5. Inference (Translation)

* Greedy decoding
* Starts with `[BOS]` token
* Generates tokens step-by-step
* Stops when `[EOS]` is predicted or max length is reached

---

## Current Results

* The model is **not performing well yet**
* Output mostly predicts repetitive or `[PAD]` tokens
* This is expected due to:

  * Very **low training epochs**
  * Limited dataset size
  * No padding mask in loss
  * No proper evaluation metric

---

## Limitations

* No attention mask for padding tokens
* No validation loop
* No evaluation metrics (BLEU, ROUGE, etc.)
* Greedy decoding only (no beam search)
* Small training duration

---

## Future Work

* Train for **more epochs**
* Use **larger and cleaner dataset**
* Add **evaluation metrics**:

  * BLEU score
  * ROUGE score
* Implement:

  * Beam search decoding
  * Padding masks
  * Learning rate scheduling
* Experiment with:

  * Different tokenizers
  * Pretrained embeddings
  * Larger Transformer models
* Test model on **different real-world scenarios**

---

## Conclusion

This notebook provides a **from-scratch implementation** of a Transformer for machine translation. While current performance is limited, it establishes a strong foundation for further improvements and experimentation.

---


