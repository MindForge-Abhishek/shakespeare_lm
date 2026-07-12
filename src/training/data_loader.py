import torch
import json
from pathlib import Path

processed_dir = Path(__file__).resolve().parent.parent.parent /"data" / "processed"

def load_dataset():
    train_data = torch.load(processed_dir / "train.pt" , weights_only = True)
    val_data = torch.load(processed_dir / "val.pt" ,weights_only = True)
    return train_data, val_data

def load_vocab():
    with open(processed_dir / "vocab.json", "r", encoding="utf-8") as f:
        stoi =json.load(f)
    itos = {v:k for k, v in stoi.items()}
    return stoi, itos

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data)- block_size ,(batch_size,))
    x = torch.stack([data[i: i + block_size]for i in ix])
    y = torch.stack([data[i+1 : i + block_size +1] for i in ix])
    return x,y

