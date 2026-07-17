# Shakespeare Language Model

A character-level language model trained on Shakespeare's works.
Built completely from scratch using PyTorch.

## Purpose

This project is built for deep learning education.
The goal is to understand every component of a modern
language model — from raw text to trained model.

## Hardware

- MacBook Air M2, 8GB unified memory
- Training uses Apple MPS (Metal Performance Shaders) GPU backend


## Project Structure

shakespeare_lm/
│
├── data/
│   ├── raw/                    # Raw downloaded text (gitignored)
│   └── processed/              # Encoded tensors and vocab (gitignored)
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── src/
│   ├── data/
│   │   ├── download.py         # Downloads Tiny Shakespeare dataset
│   │   ├── tokenizer.py        # Character-level tokenizer (build_vocab, encode, decode)
│   │   └── tensor_creation.py  # Encodes text, splits 90/10, saves processed tensors
│   │
│   ├── models/
│   │   └── bigram.py           # BigramLanguageModel (4,225 parameters)
│   │
│   └── training/
│       ├── data_loader.py      # load_dataset(), load_vocab(), get_batch()
│       └── train.py            # Training loop + text generation
│
├── environment.yml
└── README.md

## Status

**Phase completed: Bigram Language Model**

A character-level bigram model has been fully implemented and trained.
It predicts the next character using only the current character.

- Training loss after 10,000 steps: ~2.4
- Theoretical baseline (random guessing): 4.17
- Architecture ceiling: ~2.0–2.2 (bigram is fundamentally limited by one character of context)
- Training device: Apple MPS (M2 GPU)

Sample generated text after training:

an:
MNELI's tibowaid
HERDms t of IXf hasen d ffr h Iotosit
The model has learned character-pair statistics: correct spacing, apostrophe
placement, colons after names, and common English bigrams like 'th'. It has
not learned words, grammar, or meaning — that requires a transformer.

**Next phase: Self-Attention and Transformer Architecture**

---

---

## Dataset

**Tiny Shakespeare** — the complete works of Shakespeare as a single plain text file.

- Source: https://github.com/karpathy/char-rnn
- Size: 1,115,394 characters
- Vocabulary: 65 unique characters (letters, punctuation, newline, space)
- Split: 90% training (1,003,854 characters), 10% validation (111,540 characters)

---

## Setup

**Requirements:**
- macOS with Apple Silicon (M2 or later) — MPS GPU backend used
- Miniforge (Conda) for environment management
- Python 3.12

**Create environment:**
```bash
conda env create -f environment.yml
conda activate shakespeare_lm
pip install torch
```

**Prepare dataset:**
```bash
python -m src.data.download
python -m src.data.tensor_creation
```

**Run training:**
```bash
python -m src.training.train
```

---

## Architecture — Bigram Model

The simplest possible language model. One embedding table of shape [65, 65].

Each row corresponds to one input character. Each column in that row is the
raw score (logit) for what the next character should be. The model learns by
adjusting these scores until the correct next character has the highest score.

**Parameters:** 4,225 (65 × 65 embedding table)

**Loss function:** Cross-entropy

**Optimizer:** AdamW (learning rate 1e-3)

**Hyperparameters:**

| Parameter | Value | Meaning |
|---|---|---|
| block_size | 8 | Characters of context per training example |
| batch_size | 32 | Sequences processed in parallel per step |
| learning_rate | 1e-3 | AdamW base learning rate |
| max_steps | 10,000 | Total training iterations |

---

## Roadmap

- [x] Environment setup (Miniforge, PyTorch, MPS verified)
- [x] Data pipeline (download, tokenize, encode, split, persist)
- [x] Character-level tokenizer
- [x] Bigram language model
- [x] Training loop with AdamW
- [x] Text generation (autoregressive sampling)
- [ ] Self-attention mechanism
- [ ] Single attention head
- [ ] Multi-head attention
- [ ] Feed-forward layer
- [ ] Residual connections
- [ ] Layer normalisation
- [ ] Full GPT-style transformer
- [ ] Validation loss tracking
- [ ] Checkpointing

---

## Key Engineering Decisions

- Character IDs are 0-indexed positions in `sorted(set(text))`, not ASCII values
- Train/validation split is a fixed index cut, not a random shuffle — required for sequential text data
- Processed tensors are saved to disk (`data/processed/`) so the encode step runs only once
- Only `stoi` is persisted as JSON; `itos` is always reconstructed from it
- `load_dataset()` and `load_vocab()` are separate functions — training and inference need different things
- Device selection uses a runtime check (`mps` → `cpu` fallback), never hardcoded
- `@torch.no_grad()` and `model.eval()` are used consistently during generation