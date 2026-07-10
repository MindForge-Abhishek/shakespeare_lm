def build_vocab(text):
    unique_chars = sorted(set(text))
    stoi = {char: idx for idx, char in enumerate(unique_chars)}
    itos = {idx: char for idx, char in enumerate(unique_chars)}
    return stoi, itos


def encode(s, stoi):
    return [stoi[char] for char in s]


def decode(ids, itos):
    return "".join([itos[i] for i in ids])