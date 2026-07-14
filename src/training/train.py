import torch
from src.training.data_loader import load_dataset, load_vocab, get_batch
from src.models.bigram import BigramLanguageModel

block_size = 8
batch_size = 32
learning_rate = 1e-3
max_steps = 10000
eval_interval = 500

device = 'mps' if torch.backends.mps.is_available() else 'cpu'


train_data ,val_data = load_dataset()
stoi,itos = load_vocab()
vocab_size = len(stoi)

model = BigramLanguageModel(vocab_size)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for step in range(max_steps):
    x, y = get_batch(train_data, batch_size, block_size)
    x,y = x.to(device), y.to(device)

    logits, loss = model(x,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0:
        print (f"step {step}: loss {loss.item():.4f}")



@torch.no_grad()
def generate (model,itos,start_char, max_new_tokens, device):
    model.eval()
    input_ids = torch.tensor([[stoi[start_char]]], dtype = torch.long).to(device)
    result = [start_char]

    for _ in range (max_new_tokens):
        logits, _ = model(input_ids[:, -1:])
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim = -1)
        next_id = torch.multinomial(probs, num_samples=1)
        result.append(itos[next_id[0, 0].item()])
        input_ids = torch.cat([input_ids, next_id], dim = 1)

    return ''.join(result)

print ("\n-----Generated text ----")
print(repr(generate(model, itos, start_char= 'a', max_new_tokens=200, device = device)))



