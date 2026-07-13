import torch
import torch.nn as nn

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, vocab_size)

    
    def forward(self,x):
        logits = self.embedding_table(x)
        return logits