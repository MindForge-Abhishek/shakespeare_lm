# Shakespeare Language Model

A character-level transformer language model trained on the complete works of Shakespeare, built from scratch using PyTorch.

This project implements the full GPT-style transformer architecture from first principles — every component built and understood from the ground up, with no black boxes.

## Results

| Model | Parameters | Val Loss | Sample Output |
|-------|-----------|----------|---------------|
| Bigram | 4,225 | ~2.4 | `HERDms t of IXf hasen d` |
| Transformer | 209,729 | ~1.75 | `WARWICK: Now how their come, our chown` |

The transformer model learned Shakespeare character names, dialogue formatting, punctuation placement, and basic English grammar structure purely from character-level statistics.

## Architecture

A GPT-style autoregressive transformer with the following components:

- **Token embeddings** — learned character representations
- **Positional embeddings** — learned position representations
- **Multi-head self-attention** — causal (masked) attention with multiple heads in parallel
- **Feed-forward layers** — position-wise MLP with 4x expansion
- **Residual connections** — skip connections around attention and feed-forward
- **Layer normalisation** — pre-norm configuration
- **Dropout** — regularisation throughout

### Hyperparameters

vocab_size = 65 (unique characters in dataset)
block_size = 32 (context length)
d_model = 64 (embedding dimension)
num_heads = 4 (attention heads per block)
num_layers = 4 (transformer blocks)
dropout = 0.2
batch_size = 32
learning_rate = 1e-3
training_steps = 5000

## Project Structure

shakespeare_lm/
│
├── src/
│ ├── data/
│ │ ├── download.py # Downloads Tiny Shakespeare dataset
│ │ ├── tokenizer.py # Character-level tokenizer (build_vocab, encode, decode)
│ │ └── tensor_creation.py # Encodes dataset, splits 90/10, saves to disk
│ │
│ ├── models/
│ │ ├── bigram.py # Baseline bigram language model
│ │ └── transformer.py # Full transformer (AttentionHead, MultiHeadAttention,
│ │ # FeedForward, Block, TransformerLanguageModel)
│ │
│ └── training/
│ ├── data_loader.py # load_dataset, load_vocab, get_batch
│ ├── train.py # Bigram training loop
│ └── transformer_training.py # Transformer training loop with validation,
│ # gradient clipping, and checkpointing
│
├── data/
│ ├── raw/ # Raw downloaded text (gitignored)
│ └── processed/ # Encoded tensors and vocab (gitignored)
│
├── checkpoints/ # Saved model weights (gitignored)
├── notebooks/
│ └── 01_data_exploration.ipynb
| └── 01_attention.ipynb
├── environment.yml
└── README.md


## Setup

### Prerequisites

- macOS with Apple Silicon (MPS backend) or any machine with CUDA/CPU
- Miniforge (conda)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/shakespeare_lm.git
cd shakespeare_lm

# Create and activate conda environment
conda env create -f environment.yml
conda activate shakespeare_lm

# Install PyTorch
pip install torch
```

### Prepare Data

```bash
# Download the dataset
python -m src.data.download

# Create processed tensors
python -m src.data.tensor_creation
```

### Train

```bash
# Train the bigram baseline
python -m src.training.train

# Train the transformer
python -m src.training.transformer_training
```

Trained model is saved to `checkpoints/transformer.pt`.

## What Was Built And Learned

This project was built as a structured learning exercise to understand every component of modern language models from first principles.

### Concepts covered

- Character-level tokenization and vocabulary construction
- Tensor creation, train/validation splitting
- Batch sampling for sequential text data
- The bigram language model as a baseline
- Cross-entropy loss and what it measures
- The training loop — forward, loss, backward, step
- Self-attention — queries, keys, values, scaled dot-product attention
- Causal masking — why future tokens must be hidden during training
- Multi-head attention — parallel attention heads, concatenation, output projection
- Feed-forward layers — expansion, ReLU non-linearity, compression
- Residual connections — gradient flow, vanishing gradient problem
- Layer normalisation — training stability, pre-norm vs post-norm
- Positional embeddings — why attention is position-blind without them
- Dropout — regularisation, train vs eval mode
- Gradient clipping — preventing exploding gradients
- Checkpointing — saving and resuming training
- Autoregressive text generation — sampling, context cropping

## Dataset

[Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) — 1,115,394 characters, 65 unique characters, complete works of Shakespeare.

## Acknowledgements

Architecture based on the transformer introduced in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).