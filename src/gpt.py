import json
import os
import random
from datetime import datetime

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.cte import *
from src.para import *
from src.utils import *

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long, pin_memory=True)
text_size = len(data)
datai = torch.tensor([i * block_size for i in range(text_size)], dtype=torch.long, pin_memory=True)
n = int(0.9*text_size) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
train_datai = datai[:n]
val_datai = datai[n:]

da = (torch.arange(vocab_size) + text_size * block_size).unsqueeze(0).expand(batch_size, -1) # (B, vocab_size)

# Train Cache
_train_cache = []

# data loading
def get_batch(split, ix=None):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    datai = train_datai if split == 'train' else val_datai
    if ix is None:
        ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    xi = torch.stack([datai[i:i+block_size] for i in ix])
    xi_pad = torch.arange(0, block_size).unsqueeze(0)
    xi = xi + xi_pad
    
    xi = torch.cat((xi, da[0:1].repeat(xi.size(0), 1)), dim=1)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, xi, y = x.to(device), xi.to(device), y.to(device)
    return x, xi, y, ix

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
        self.cte = CritiGraph(h, tp, c, eps, epoch_cte, batch_size_cte, convergence, text_size * block_size + vocab_size, block_size + vocab_size, division_fact, loss_type)
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

    def forward(self, idx, idi, targets, 
                id_pos=None,
                train_cte=False,
                visualization=False,
                return_dyn_emb=False
    ):
        B, T, V = batch_size, block_size, vocab_size
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,E)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,E)
        x = tok_emb + pos_emb # (B,T,E)
        x = self.blocks(x) # (B,T,E)
        x: torch.Tensor = self.ln_f(x) # (B,T,E)
        if return_dyn_emb:
            return x
        
        token_embeddings = self.token_embedding_table.weight  # (V, E)
        logits_eu = torch.matmul(x, token_embeddings.t()) # (B, T, V)

        loss_ct = torch.tensor(0.0, device=x.device)
        loss_eu = F.cross_entropy(logits_eu.view(B * T, V), targets.view(B * T))
        
        if visualization == True or train_cte == True:
            ### step 1: 计算基本信息
            sta          = idi                                                  # (B, T+V)
            pos          = idi                                                  # (B, T+V)
            
            #### step 1.1: 相应的 embeddings (dynamic / static)
            emb          = torch.cat([x, token_embeddings[None, :, :].expand(B, -1, -1)], dim=1)  
                                                                                # (B, T+V, E)
            emb_norm     = torch.norm(emb, dim=-1, keepdim=True).clamp_min(1e-12)                
                                                                                # (B, T+V, 1)
            emb_normed   = emb / emb_norm                                       # (B, T+V, E)
            
            #### step 1.2: 待拟合的 Euclidean Value
            val          = emb_normed @ emb_normed.transpose(1, 2)              # (B, T+V, E) @ (B, E, T+V) -> (B, T+V, T+V)
            logits_norm  = (emb_norm[:, :T, :] *                                #  这一步是为了获得 logits
                            emb_norm.transpose(1, 2)[:, :, T:T+V])              # (B, T, 1) * (B, V, 1) -> (B, T, V)
            # val[:, :T, T:T+V] *= logits_norm                                  # (B, :T, T:T+V) * (B, T, V)
            val[:, :T, T:T+V] = to_one_hot_logits(targets, V)
            
            #### step 1.3: 直接拿来乘的 Euclidean Norm
            val_norm               = torch.ones_like(val)                       # (B, T+V, T+V)
            val_norm[:, :T, T:T+V] = logits_norm                                # 除了 logits 部分，其它都是余弦相似度
            
            
            ### step 2: 考虑有效预测
            # prob_mask    = val[:, :T, T:T+V].argmax(dim=-1) == targets          # (B, T)
            # val_mask     = torch.ones_like(val)                                 # (B, T+V, T+V)       
            # val_mask[:, :T, :] &= prob_mask[:, :, None]                         # (B, 0:T, T+V) &= (B, T, -)
            # val_mask[:, :, :T] &= prob_mask[:, None, :]                         # (B, T+V, 0:T) &= (B, -, T)
            # val_mask[:, T:, :T] = 0                                             # 左下角暂时置为无效区域 
            val_mask = torch.ones_like(val)
            
            if train_cte == True:
                ### step 3: 拟合 cos 相似度与 prob 概率分布
                logits_ct: torch.Tensor = self.cte(
                    sta, pos     ,  
                    val, val_mask, val_norm,
                    targets=targets
                )                                                                     # (B, T, V)
                
                ### step 4: 计算最终 loss
                loss_ct = F.cross_entropy(logits_ct.view(B * T, V), targets.view(B * T))

        if visualization:
            return logits_eu, loss_eu, loss_ct, 0, fetch_locals('val', 'idi', 'emb_normed')
        
        return logits_eu, loss_eu, loss_ct, 0, 0

    def TTT_cte_forward(self, 
                        x_train: torch.Tensor,    # (vB, T)
                        x_valid: torch.Tensor,    # (tB, T)
                        sta    : torch.Tensor,    # (vB, T)
                        pos    : torch.Tensor,    # (tB, T)
                        targets: torch.Tensor     # (vB, T)
    ):
        ### step.1 获取欧式空间信息
        dyn_train: torch.Tensor = self.forward(x_train, None, None, return_dyn_emb=True) # type: ignore
        dyn_train               = dyn_train.reshape(-1, dyn_train.size(2)) # (tB, T, E) -> (tB * T, E)
        dyn_valid: torch.Tensor = self.forward(x_valid, None, None, return_dyn_emb=True) # type: ignore
        dyn_valid               = dyn_valid.reshape(-1, dyn_valid.size(2)) # (vB, T, E) -> (vB * T, E)
        sta_emb                 = self.token_embedding_table.weight        # (V ,    E)
        
        ### step.2 获取范数信息以及归一化
        dyn_train_norm   = torch.norm(dyn_train, dim=-1, keepdim=True).clamp_min(1e-12) # (tB * T, 1)
        dyn_train_normed = dyn_train / dyn_train_norm
        dyn_valid_norm   = torch.norm(dyn_valid, dim=-1, keepdim=True).clamp_min(1e-12) # (vB * T, 1)
        dyn_valid_normed = dyn_valid / dyn_valid_norm
        sta_emb_norm     = torch.norm(sta_emb, dim=-1, keepdim=True).clamp_min(1e-12) # (V, 1)
        sta_emb_normed   = sta_emb / sta_emb_norm 
        
        ### step.3 获取 CTE 需要拟合的内容
        val = dyn_valid_normed @ dyn_train_normed.t()             # (vB * T, E) @ (tB * T, E) -> (vB * T, tB * T)
        val_mask = torch.ones_like(val, dtype=torch.bool)         # (vB * T, tB * T)
        val_norm = torch.ones_like(val)                           # (vB * T, tB * T)

        ### step.4 对 CTE 进行 test-time training
        sta = sta.reshape(-1) # (vB, T) -> (vB * T)
        pos = pos.reshape(-1) # (tB, T) -> (tB * T) 
        TTT_loss = self.cte.forward_TTT(
            sta, pos     ,  
            val, val_norm,
        ) # (vB * T, C, tp)
        
        return TTT_loss

        # return loss_ct, loss_eu
    def TTT_get_loss(self, 
                     x_valid: torch.Tensor, # (vB, T)
                     sta    : torch.Tensor, # (vB, T)
                     targets: torch.Tensor  # (vB, T)
    ):
        vB, T = sta.size(0), sta.size(1)
        ### step.1 获取欧式空间信息
        dyn_valid: torch.Tensor = self.forward(x_valid, None, None, return_dyn_emb=True) # type: ignore
        dyn_valid_norm          = torch.norm(
            dyn_valid, dim=-1, keepdim=True
        ).clamp_min(1e-12)                                                 # (vB, T, 1)
        sta_emb                 = self.token_embedding_table.weight        # (V ,    E)
        sta_emb_norm            = torch.norm(
            sta_emb, dim=-1, keepdim=True
        ).clamp_min(1e-12)                                                 # (V, 1)
        voc                     = da[0:1, :].repeat(vB, 1).to(device)      # (vB, V)
        norm = dyn_valid_norm * sta_emb_norm.reshape(1, 1, vocab_size)     # (vB, T, 1) * (1, 1, V) -> (vB, T, V)

        #### step.2 获取 logits
        logits_ct = self.cte.distance(
            self.cte.locations[0][sta].unsqueeze(2),       # (vB, T, 1, tp)
            self.cte.locations[0][voc].unsqueeze(1),       # (vB, 1, V, tp)
            norm[..., None],                               # (vB, T, V, 1)
            dev_num=0
        ).mean(dim=-1)                        # (vB, T, V)
        logits_eu = dyn_valid @ sta_emb.t()   # (vB, T, E) @ (E, V) -> (vB, T, V)
        
        ### step.3 获取损失
        loss_ct = F.cross_entropy(logits_ct.view(-1, vocab_size), targets.view(-1), reduction='none').reshape(vB, T) # (vB , T)
        loss_eu = F.cross_entropy(logits_eu.view(-1, vocab_size), targets.view(-1), reduction='none').reshape(vB, T) # (vB , T)
        
        return loss_eu, loss_ct


@torch.no_grad()
def evaluate(model: GPTLanguageModel):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses_ori = torch.zeros(eval_iters)
        losses_cts = torch.zeros(eval_iters)
        losses_gap = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, XI, Y, _ = get_batch(split)
            _, loss_ori, loss_cts, loss_gap = model(X, XI, targets=Y, evaluation=1)
            losses_ori[k] = loss_ori.item()
            losses_cts[k] = loss_cts.item()
            losses_gap[k] = loss_gap.item()
        out[split] = [losses_ori.mean(), losses_cts.mean(), losses_gap.mean()]
    model.train()
    return out


def train_gpt():
    model: GPTLanguageModel = GPTLanguageModel().to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    running_loss = []
    for iter in range(max_iters):
        
        xb, xi, yb, _ = get_batch('train')
        _, loss, _, _ = model(xb, xi, targets=yb)
        optimizer.zero_grad(set_to_none=True)
        running_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        if iter % gpt_save_interval == 0 or iter == max_iters - 1:
            print(f"current iter: {iter}, avg loss in last {gpt_save_interval} iters: {sum(running_loss) / len(running_loss)}")
            running_loss = []
            torch.save(model.state_dict(), os.path.join(gpt_path, f"iters_{iter}.pth"))

@torch.no_grad()
def train_cte(gpt_cktp: str, cte_cktp: str, train_cache_cktp: str):
    model: GPTLanguageModel = GPTLanguageModel().to(device)
    model_cktp = torch.load(gpt_cktp)
    load_state_dict_skip_prefixes(model, model_cktp, prefixes_to_skip=("cte",), strict=False)
    model.cte = CritiGraph(h, tp, c, eps, epoch_cte, batch_size_cte, convergence, text_size * block_size + vocab_size, block_size + vocab_size, division_fact, loss_type)
    
    model.eval()
    for iter in range(max_iters):
        if iter % cte_save_interval == 0 or iter == max_iters - 1: 
            visualization = True
        else:
            visualization = False
        
        xb, xi, yb, ix = get_batch('train')
        _train_cache.append(ix)
        
        _, loss_eu, loss_ct, _, var = model(xb, xi, targets=yb, train_cte=1, visualization=visualization)
        print(f"current train iter: {iter}, loss_eu: {fmt6w(loss_eu.item())}, loss_ct: {loss_ct.item()}")
        
        if visualization:    
            visualize_similarity(model, var, iter)
            torch.save(model.cte.state_dict(), cte_cktp.format(iter))
            torch.save(_train_cache, train_cache_cktp.format(iter))

@torch.no_grad()
def validate_cte(gpt_cktp: str, cte_cktp: str, train_cache_cktp: str):
    model: GPTLanguageModel = GPTLanguageModel().to(device)
    model_cktp = torch.load(gpt_cktp)
    load_state_dict_skip_prefixes(model, model_cktp, prefixes_to_skip=("cte",), strict=False)
    del model.cte
    torch.cuda.empty_cache()
    model.cte = CritiGraph(h, tp, c, eps, epoch_cte, batch_size_cte, convergence, text_size * block_size + vocab_size, block_size + vocab_size, division_fact, TTT_loss_type)
    
    model.cte.load_state_dict(torch.load(cte_cktp))
    del model.cte.locations
    torch.cuda.empty_cache()
    
    model.cte.locations = [model.cte.main_locations.to(model.cte.devices[i]) for i in range(len(model.cte.devices))]
    train_cache = torch.load(train_cache_cktp)
    train_cache = torch.cat(train_cache, dim=0) # (sumB)
    
    loss_dict = {
        "per_sample": [[] for _ in range(0, train_cache.size(0), train4test_val)],
        "per_lth"   : [[] for _ in range(block_size)],
        "per_epoch" : []
    }
    
    '''
    目前为写回式更改，请注意啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊
    '''
    
    
    for st_point in range(0, len(val_data), val_batch_size * block_size): # 每次取出 val_batch_size * block_size 个字符
        ### step 1: 获取当前阶段需要 val 的数据
        ix = st_point + torch.arange(0, val_batch_size) * block_size # 仅拿出单个 block，去除 batch 维度
        xb_val, idi_val, yb_val, _ = get_batch('val', ix=ix) # (vB, T), (vB, T), (vB, T)
 
 
        total_TTT_loss = torch.zeros((val_batch_size * block_size, 2 * model.cte.k * model.cte.h + 1, model.cte.tp), pin_memory=True)
        for epoch in range(epoch_cte_TTT):
            for i, pos_id in enumerate(range(0, train_cache.size(0), TTT_batch_size)): # 每次取出 TTT_batch_size 个样本
                ### step 2: 获取当前拿来 test-time training 的数据
                ix_train = train_cache[pos_id : pos_id + TTT_batch_size] # (tB)
                xb_train, idi_train, yb_train, _ = get_batch('train', ix=ix_train) # (tB, T), (tB, T), (tB, T)

                TTT_loss = model.TTT_cte_forward(
                    xb_train, xb_val,
                    idi_val[:, :block_size], idi_train[:, :block_size],
                    yb_val
                ) # (vB * T, C, tp)
                
                total_TTT_loss += TTT_loss
                print(f"epoch: {epoch} / {epoch_cte_TTT}, step: {i} / {train_cache.size(0) // TTT_batch_size}, TTT_loss: {TTT_loss.mean().item()}")

            model.cte.update_TTT(total_TTT_loss) # 全局更新
            loss_eu, loss_ct = model.TTT_get_loss(xb_val, idi_val[:, :block_size], yb_val) # (vB, T)
            print(f"epoch: {epoch} / {epoch_cte_TTT}, val_loss_eu: {loss_eu.mean().item()}, val_loss_ct: {loss_ct.mean().item()}")
            
            
        break # 只测试某一个 val
            


def visualize_similarity(model, var, iter):
    emb_eu = var['emb_normed'][0]
    emb_ct = model.cte.main_locations[var['idi'][0].cpu()]

    # --- 相似度矩阵 (eu) ---
    S = torch.matmul(emb_eu, emb_eu.T).cpu().numpy()
    np.fill_diagonal(S, 1.0)

    # 转为“距离”做聚类
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    dvec = squareform(D, checks=False)

    Z = linkage(dvec, method='average')
    order = leaves_list(Z)
    S_re = S[order][:, order]

    # --- 一张图：左树 + 热图 + 右色条 (eu) ---
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[2.5, 14, 0.5],
        height_ratios=[1.0],
        wspace=0.0, hspace=0.0
    )

    # 左侧行树
    ax_row = fig.add_subplot(gs[0, 0])
    dendrogram(Z, ax=ax_row, orientation="right", no_labels=True, color_threshold=None)
    ax_row.invert_yaxis()
    ax_row.set_xticks([]); ax_row.set_yticks([])

    # 中间热图 (eu)
    ax = fig.add_subplot(gs[0, 1])
    vmin, vmax = S_re.min(), S_re.max()
    norm = PowerNorm(gamma=2.0, vmin=vmin, vmax=vmax)
    im = ax.imshow(S_re, cmap="inferno", norm=norm,
                   interpolation="nearest", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Hierarchical Cosine Similarity (eu)")

    # 右侧颜色条
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("eu-cosine similarity", rotation=270, labelpad=25)

    fig.savefig(sim_eu_path.format(iter), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Similarity + tree visualization saved to {sim_eu_path.format(iter)}")

    # --- 相似度矩阵 (ct) ---
    distance_ct = model.cte.main_distance(
        emb_ct.unsqueeze(1), emb_ct.unsqueeze(0),
        torch.ones((block_size + vocab_size, block_size + vocab_size, 1))
    ).mean(dim=-1).cpu().numpy()
    np.fill_diagonal(distance_ct, 1.0)
    S_ct = distance_ct[order][:, order]

    # --- 一张图：左树 + 热图 + 右色条 (ct) ---
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[2.5, 14, 0.5],
        height_ratios=[1.0],
        wspace=0.0, hspace=0.0
    )

    # 左侧行树 (沿用同一个 Z 顺序)
    ax_row = fig.add_subplot(gs[0, 0])
    dendrogram(Z, ax=ax_row, orientation="right", no_labels=True, color_threshold=None)
    ax_row.invert_yaxis()
    ax_row.set_xticks([]); ax_row.set_yticks([])

    # 中间热图 (ct)
    ax = fig.add_subplot(gs[0, 1])
    vmin, vmax = S_ct.min(), S_ct.max()
    norm = PowerNorm(gamma=2.0, vmin=vmin, vmax=vmax)
    im = ax.imshow(S_ct, cmap="inferno", norm=norm,
                   interpolation="nearest", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Hierarchical Cosine Similarity (ct)")

    # 右侧颜色条
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("ct-cosine similarity", rotation=270, labelpad=25)

    fig.savefig(sim_ct_path.format(iter), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Similarity + tree visualization saved to {sim_ct_path.format(iter)}")




if __name__ == "__main__":
    gpt_ckpt = "iters_4000.pth"
    cte_ckpt = "gpt_iters_4000_cte_iters_{}.pth"
    train_cache_ckpt = "gpt_iters_4000_cte_iters_{}_train_cache.pth"
    
    # train_gpt()
    # train_cte(os.path.join(gpt_path, gpt_ckpt),
    #           os.path.join(cte_path, cte_ckpt),
    #           os.path.join(train_cache_path, train_cache_ckpt))
    
    
    validate_cte(os.path.join(gpt_path, gpt_ckpt),
              os.path.join(cte_path, cte_ckpt.format(100)),
              os.path.join(train_cache_path, train_cache_ckpt.format(100)))
                 
