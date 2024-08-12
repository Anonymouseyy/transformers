import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle

from GPT import *

# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
block_size = 64
batch_size = 128
max_iters = 1000
learning_rate = 3e-4
eval_iters = 100
n_embed = 384
n_layer = 8
n_head = 8
dropout = 0.2

with open("x.txt", "r", encoding="UTF-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

str_to_int = { ch:i for i, ch in enumerate(chars) }
int_to_str = { i:ch for i, ch in enumerate(chars) }
encode = lambda str: [str_to_int[ch] for ch in str]
decode = lambda ints: "".join([int_to_str[i] for i in ints])

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.8*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

model = GPTLanguageModel(vocab_size)
m = model.to(device)

@torch.no_grad
def estimate_loss():
    model.eval() # Set model to evaluation mode
    split = "val"
    losses = torch.zeros(eval_iters)

    for i in range(eval_iters):
        X, Y = get_batch(split)
        _, loss = model(X, Y) # Automatically calls forward method
        losses[i] = loss.item()
    
    model.train() # Set model back to training mode
    return losses.mean()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

print(estimate_loss().item())

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_chars = decode(m.generate(context, 500)[0].tolist())
print(generated_chars)