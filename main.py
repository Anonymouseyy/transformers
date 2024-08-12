import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle

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


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # Ensure each position only attends to previous positions

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # B * T * T tensor where each element represents the attention score between two characters per batch scaled
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # Autoregressive (only attending to previous terms) attention
        wei = F.softmax(wei, dim=-1) # Probabilities
        wei = self.dropout(wei)

        v = self.value(x) # Scale back to B * T * C
        out = wei @ v # Compute weighted sum of value matrix and attention scores
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embed) # Projecting concatenated size back original
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # Combine all learned data
        out = self.dropout(self.proj(out)) # Project back to original size and perform dropout
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        
        self.sa = MultiHeadAttention(n_head, head_size) # Self attention
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y) # Post norm architecture

        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embed) # Final layer normalization
        self.lm_head = nn.Linear(n_embed, vocab_size) # Language model head

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Normal distribution of starting weights
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape

        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        
        # Reshape tensor to calculate loss
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, loss = self.forward(index_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1) # (B, T+1)

        return index


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
    model = pickle.load(f).to(device)

print(f"Model with loss of {estimate_loss().item()} loaded, EXIT to cancel")

while True:
    prompt = input("Prompt: ")

    if prompt == "EXIT":
        break

    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist())
    print(f'Completion:\n{generated_chars}')