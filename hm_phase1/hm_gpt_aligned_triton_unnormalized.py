import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.nn import functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

from hm_cte_aligned_triton_unnormalized import *
from hm_para_aligned_triton_unnormalized import *

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

da = (torch.arange(vocab_size)+text_size).unsqueeze(0).expand(batch_size, -1)

# data loading
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

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses_ori = torch.zeros(eval_iters)
        losses_cts = torch.zeros(eval_iters)
        losses_gap = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, XI, Y = get_batch(split)
            _, loss_ori, loss_cts, loss_gap = model(X, XI, targets=Y, evaluation=1)
            losses_ori[k] = loss_ori.item()
            losses_cts[k] = loss_cts.item()
            losses_gap[k] = loss_gap.item()
        out[split] = [losses_ori.mean(), losses_cts.mean(), losses_gap.mean()]
    model.train()
    return out

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
        self.cte = CritiGraph(h, tp, c, eps, epoch_cte, batch_size_cte, convergence, text_size + vocab_size, block_size + vocab_size, loss_calc, division_fact)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm

        self.region_mask = torch.zeros(block_size+vocab_size, block_size+vocab_size, dtype=torch.long, device=device)
        self.region_mask[:block_size, :block_size] = 0  # 左上
        self.region_mask[:block_size, block_size:] = 1  # 右上
        self.region_mask[block_size:, :block_size] = 1  # 左下（由于对称性，使用右上归一化）
        self.region_mask[block_size:, block_size:] = 2  # 右下

        self.apply(self._init_weights)
        x = torch.rand((batch_size, block_size + vocab_size, block_size + vocab_size), device=device)
        self.register_buffer('x', x)
        xi = torch.rand((batch_size, block_size + vocab_size, block_size + vocab_size), device=device)
        self.register_buffer('xi', xi)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, idi, targets=None, evaluation=None):
        st = time.perf_counter()
        with_cte = False
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        token_embeddings = self.token_embedding_table.weight  # (vocab_size, n_embd)
    
        logits = torch.matmul(x, token_embeddings.t())

        if targets is None:
            loss_ori, loss_cts, loss_gap = None, None, None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss_ori = F.cross_entropy(logits, targets)
            
            if evaluation == 1:
                with_cte = True
                x_norm = torch.cat((x, token_embeddings.unsqueeze(0).expand(B, -1, -1)), dim=1)
                dismatrix_eu = torch.matmul(x_norm, x_norm.transpose(1, 2))
                
                td_eu = torch.abs(dismatrix_eu)
                max_vals = torch.stack([
                    td_eu[:, :T, :T].amax(dim=(1, 2), keepdim=True),  # 左上
                    td_eu[:, :T, T:].amax(dim=(1, 2), keepdim=True),  # 右上
                    td_eu[:, T:, T:].amax(dim=(1, 2), keepdim=True)   # 右下
                ], dim=1)
                selected_max = max_vals[:, self.region_mask]
                dismatrix_eu = dismatrix_eu / selected_max.squeeze(-1).squeeze(-1)

                self.x = dismatrix_eu
                dismatrix_ct = self.cte(idi, dismatrix_eu)
                self.xi = dismatrix_ct
                logits_ct = dismatrix_ct[:, :T, T:]
                logits_ct = logits_ct.view(B*T, C)
                loss_cts = F.cross_entropy(logits_ct, targets)
                delt = dismatrix_eu - dismatrix_ct
                # loss_gap = torch.abs(delt).mean()
                loss_gap = (delt * delt).mean()
        ed = time.perf_counter()
        # if with_cte: 
        #     print(f"Time with cte: {(ed - st) * 1000} ms")
        
        return logits, loss_ori, loss_cts, loss_gap

def visualize_similarity(xi):
    model = GPTLanguageModel().to(device)
    model.load_state_dict(torch.load('gpt_model_dy_unnormalized.pth'))
    model.eval()
    print("Model loaded from gpt_model_dy_unnormalized.pth")
    distance_eu = model.x[0].cpu().numpy()
    distance_ct = model.xi[0].cpu().numpy()
    
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
    plt.savefig('token_similarity_heatmap_eu_alinged_triton_unnormalized.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Similarity matrix visualization saved to token_similarity_heatmap_eu_unnormalized.png")

    reordered_sim = distance_ct[cluster_order, :][:, cluster_order]
    similarity_matrix = reordered_sim
    plt.figure(figsize=(15, 15))
    plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
    cbar = plt.colorbar()
    cbar.set_label('ct-cosine similarity', rotation=270, labelpad=20)
    plt.title('Token Embedding Similarity Matrix')
    plt.xlabel('Token Index')
    plt.ylabel('Token Index')
    plt.savefig('token_similarity_heatmap_ct_alinged_triton_unnormalized.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Similarity matrix visualization saved to token_similarity_heatmap_ct_unnormalized.png")

def train_model():
    model = GPTLanguageModel().to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {iter}: train: ori={losses['train'][0]:.4f}, cts={losses['train'][1]:.4f}, gap={losses['train'][2]:.8f}\
                | val: ori={losses['val'][0]:.4f}, cts={losses['val'][1]:.4f}, gap={losses['val'][2]:.8f}")
            torch.save(model.state_dict(), 'gpt_model_dy_unnormalized.pth')
            print("Model saved to gpt_model_dy_unnormalized.pth")
            visualize_similarity()
        xb, xi, yb = get_batch('train')
        _, loss, _, _ = model(xb, xi, iter, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    train_model()
