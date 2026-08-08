import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):

    def __init__(self, head_size, d_model, block_size):
        super().__init__()
        self.key   = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        scores = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size, d_model, block_size):
        super().__init__()
        self.heads = nn.ModuleList([
            AttentionHead(head_size, d_model, block_size) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, d_model, num_heads, block_size):
        super().__init__()
        head_size = d_model // num_heads
        self.attention = MultiHeadAttention(num_heads, head_size, d_model, block_size)
        self.ffwd = FeedForward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TransformerLanguageModel(nn.Module):

    def __init__(self, vocab_size, d_model, num_heads, num_layers, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[
            Block(d_model, num_heads, block_size) for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.block_size = block_size

    def forward(self, x, targets=None):
        B, T = x.shape

        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(T, device=x.device))
        x = tok_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss




if __name__ == "__main__":
    torch.manual_seed(42)
    vocab_size = 65
    d_model = 64
    num_heads = 4
    num_layers = 4
    block_size = 8
    B = 4

    x = torch.randint(0, vocab_size, (B, block_size))
    targets = torch.randint(0, vocab_size, (B, block_size))

    model = TransformerLanguageModel(vocab_size, d_model, num_heads, num_layers, block_size)
    logits, loss = model(x)

    print(f"Input shape:   {x.shape}")
    print(f"Logits shape:  {logits.shape}")
    # print(f"Loss:          {loss.item():.4f}")
    print(f"Expected loss: ~{__import__('math').log(vocab_size):.4f}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params:  {total_params:,}")