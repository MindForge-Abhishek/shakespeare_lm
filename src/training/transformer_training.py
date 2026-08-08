import torch
from src.training.data_loader import load_dataset, load_vocab, get_batch
from src.models.transformer import TransformerLanguageModel

# hyperparameters
block_size     = 32
batch_size     = 32
d_model        = 64
num_heads      = 4
num_layers     = 4
learning_rate  = 1e-3
max_steps      = 5000
eval_interval  = 500
eval_steps     = 50

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

train_data, val_data = load_dataset()
stoi, itos = load_vocab()
vocab_size = len(stoi)

model = TransformerLanguageModel(vocab_size, d_model, num_heads, num_layers, block_size)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}
    for split_name, data in [('train', train_data), ('val', val_data)]:
        split_losses = []
        for _ in range(eval_steps):
            x, y = get_batch(data, block_size, batch_size)
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            split_losses.append(loss.item())
        losses[split_name] = sum(split_losses) / len(split_losses)
    model.train()
    return losses

for step in range(max_steps):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step:4d}: train loss {losses['train']:.4f}  val loss {losses['val']:.4f}")

    x, y = get_batch(train_data, block_size, batch_size)
    x, y = x.to(device), y.to(device)
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

losses = estimate_loss()
print(f"step {max_steps:4d}: train loss {losses['train']:.4f}  val loss {losses['val']:.4f}")



@torch.no_grad()
def generate(model, itos, stoi, start_char, max_new_tokens, device):
    model.eval()
    input_ids = torch.tensor([[stoi[start_char]]], dtype=torch.long).to(device)
    result = [start_char]
    for _ in range(max_new_tokens):
        input_ids_cropped = input_ids[:, -block_size:]
        logits, _ = model(input_ids_cropped)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        result.append(itos[next_id[0, 0].item()])
        input_ids = torch.cat([input_ids, next_id], dim=1)
    return ''.join(result)

print("\n----- Generated text -----")
print(generate(model, itos, stoi, start_char='A', max_new_tokens=300, device=device))