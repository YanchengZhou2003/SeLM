import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.para_bugfixed import *
from src.utils import *

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
text_size = len(data)
datai = torch.tensor(range(text_size), dtype=torch.long)
n = int(0.9*text_size) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
train_datai = datai[:n]
val_datai = datai[n:]

da = (torch.arange(vocab_size) + text_size * block_size).unsqueeze(0).expand(batch_size, -1)

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    datai = train_datai if split == 'train' else val_datai
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    xi = torch.stack([datai[i:i+block_size] for i in ix])
    xi = torch.cat((xi, da), dim=1)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, xi, y = x.to(device), xi.to(device), y.to(device)
    return x, xi, y

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        # head_size = n_embd // n_head
        head_size = 8
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, idi, targets: torch.Tensor, train_cte=False,
                visualization=False):
        B, T, V = batch_size, block_size, vocab_size
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,E)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,E)
        x = tok_emb + pos_emb # (B,T,E)
        x = self.blocks(x) # (B,T,E)
        x: torch.Tensor = self.ln_f(x) # (B,T,E)
        token_embeddings = self.token_embedding_table.weight  # (V, E)
        
        logits_eu = torch.matmul(x, token_embeddings.t()) # (B, T, V)
        loss_eu = F.cross_entropy(logits_eu.view(B * T, V), targets.view(B * T))
        
        if visualization == True:
            x_norm     = torch.norm(x, dim=-1, keepdim=True)                
            x_normed   = x / x_norm  
            return loss_eu, x_normed
        
        return loss_eu, 0


def train_gpt():
    model: GPTLanguageModel = GPTLanguageModel().to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    running_loss = []
    for iter in range(max_iters):
        if iter % save_interval == 0 or iter == max_iters - 1:
            visualization = True
        else:
            visualization = False
       
        xb, xi, yb = get_batch('train')
        loss, emb_normed = model(xb, xi, targets=yb, visualization=visualization)
        optimizer.zero_grad(set_to_none=True)
        running_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        if visualization:
            print(f"current iter: {iter}, avg loss in last {save_interval} iters: {sum(running_loss) / len(running_loss)}")
            running_loss = []
            visualize_similarity(emb_normed, iter)   



def visualize_similarity(emb_eu: torch.Tensor, iter: int):
    emb_eu = emb_eu[0].detach()
    distance_eu = torch.matmul(emb_eu, emb_eu.transpose(0, 1)).cpu().numpy()
    
    Z = linkage(distance_eu, method='ward')
    cluster_order = leaves_list(Z) 
    
    reordered_sim = distance_eu[cluster_order, :][:, cluster_order]
    similarity_matrix = reordered_sim
    plt.figure(figsize=(15, 15))
    plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
    cbar = plt.colorbar()
    cbar.set_label('eu-cosine similarity', rotation=270, labelpad=20)
    plt.title('Token Embedding Similarity Matrix')
    plt.xlabel('Token Index')
    plt.ylabel('Token Index')
    plt.savefig(sim_eu_path + f"iter_{iter}.png", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Similarity matrix visualization saved to {sim_eu_path}")


if __name__ == "__main__":
    train_gpt()
