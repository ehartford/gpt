import torch
import torch_directml
from time import time
from datetime import timedelta
from model import BigramLanguageModel, encode, decode

from config import config

dml = torch_directml.device()
torch.device(dml)

start = time()


def elapsed():
    return str(timedelta(seconds=(time() - start)))


with open('input.txt', 'r', encoding='utf8') as f:
    text = f.read()

data = torch.tensor(encode(text), dtype=torch.long, device=dml)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i + config.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + config.block_size + 1] for i in ix])
    x, y = x.to(dml), y.to(dml)
    return x, y


blm = BigramLanguageModel().to(dml)
optimizer = torch.optim.AdamW(blm.parameters(), lr=config.learning_rate)


@torch.no_grad()
def estimate_loss():
    out = {}
    blm.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for i in range(config.eval_iters):
            x, y = get_batch(split)
            logits, loss = blm(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    blm.train()
    return out


for iter in range(config.max_iters):
    if iter % config.eval_interval == 0:
        losses = estimate_loss()
        print(f'[{iter}] train loss: {losses["train"]:.4f}, val loss: {losses["val"]:.4f}, elapsed: {elapsed()}')

    xb, yb = get_batch('train')
    logits, loss = blm(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=dml)
print(decode(blm.generate(context, max_new_tokens=500)[0].tolist()))
torch.save(blm.state_dict(), 'model.pt')
