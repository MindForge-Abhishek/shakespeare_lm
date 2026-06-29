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

├── configs/          ← All hyperparameters and settings
├── data/
│   ├── raw/          ← Original downloaded data (not tracked by Git)
│   └── processed/    ← Preprocessed tensors (not tracked by Git)
├── src/
│   ├── data/         ← Data downloading and preprocessing code
│   ├── models/       ← Model architecture definitions
│   └── training/     ← Training loop and evaluation
├── notebooks/        ← Jupyter notebooks for exploration
├── checkpoints/      ← Saved model weights (not tracked by Git)
└── logs/             ← Training logs (not tracked by Git)

## Status

🚧 Project initialization in progress.