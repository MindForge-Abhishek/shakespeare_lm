# Functions of this file
'''
1. load raw text from data/raw/input.txt 
2. Encode the text using tokenizer from src/data/tokenizer.py 
3. create the tensor of the encoded text and split the tensors into train and validation set
4. Then save the training set of tensors into data/processed/train.pt and validation set into data/processed/val.pt 
5. Load the tensor dataset for training '''

import torch
from pathlib import Path
from src.data.tokenizer import build_vocab, encode

raw_path = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "input.txt"
processed_dir = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

def prepare_dataset():
    text = raw_path.read_text(encoding = "utf-8")
    print(f"Loaded {len(text): ,} characters")

    stoi, itos = build_vocab(text)
    data = encode(text, stoi)
    data = torch.tensor(data, dtype = torch.long)
    print(f"Encoded the text into tensor of shape {data.shape}")

    split = int(0.9 *len(data))
    train_data = data[:split]
    val_data = data[split:]
    print(f"Data have splitted and length of training data is : {len(train_data)} and length of validataion data is : {len(val_data)}")

    processed_dir.mkdir(parents = True , exist_ok = True)
    torch.save(train_data, processed_dir / "train.pt")
    torch.save(val_data, processed_dir / "val.pt" )
    print(f"Saved to {processed_dir}")


def load_dataset ():
    train_data = torch.load(processed_dir / "train.pt")
    val_data = torch.load(processed_dir / "val.pt")
    return( train_data, val_data)


if __name__ == "__main__" :
    prepare_dataset()
