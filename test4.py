import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from test2 import *

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        total_losses = torch.zeros(eval_iters)
        losses_ori = torch.zeros(eval_iters)
        losses_gap = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, total_loss, loss_ori, loss_gap = model(X, Y)
            total_losses[k] = total_loss.item()
            losses_ori[k] = loss_ori.item()
            losses_gap[k] = loss_gap.item()
        out[split] = [total_losses.mean(), losses_ori.mean(), losses_gap.mean()]
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
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        # print('wei3', wei[0][0][:5])
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        # print("v requires grad:", v.requires_grad)
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
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        # self.lm_head = nn.Linear(n_embd, vocab_size)
        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

        model_CT = CritiGraph(h=27, tp=3, c=1, eps=1e-5, epoch=500, batch_size=65, convergence=0.8, vocab_size=vocab_size)
        model_CT.load_state_dict(torch.load('gpt_model_CT.pth'))
        model_CT.eval()
        print("Model loaded from gpt_model_CT.pth") 
        self.token_emb_CT = model_CT.locations  # (vocab_size, n_embd)
        # print(model_CT.locations[0])
        self.distance_matrix_CT = (model_CT.distance(self.token_emb_CT.unsqueeze(1), self.token_emb_CT.unsqueeze(0)).sum(dim=-1) / model_CT.tp)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        # logits = self.lm_head(x) # (B,T,vocab_size)
        token_embeddings = self.token_embedding_table.weight  # (vocab_size, n_embd)
        logits = torch.matmul(x, token_embeddings.t())

        if targets is None:
            total_loss, loss_ori, loss_gap = None, None, None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss_ori = F.cross_entropy(logits, targets)

            gap_weight = 10
            token_emb_norm = token_embeddings / torch.norm(token_embeddings, dim=1, keepdim=True)
            dismatrix_eu = torch.matmul(token_emb_norm, token_emb_norm.t())
            delt1 = dismatrix_eu - self.distance_matrix_CT
            loss_gap1 = torch.abs(delt1).mean()

            reg = (torch.norm(x, dim=2).view(B*T).unsqueeze(1)) * (torch.norm(token_embeddings, dim=1).unsqueeze(0)).detach()
            logits2 = (torch.matmul(x, token_embeddings.detach().t())).view(B*T, C)
            delt2 = logits2 / reg - self.distance_matrix_CT[targets]
            loss_gap2 = torch.abs(delt2).mean()

            loss_gap = 1 * loss_gap1 + 1 * loss_gap2 

            total_loss = loss_ori + gap_weight * loss_gap

        return logits, total_loss, loss_ori, loss_gap

def train_model():
    model = GPTLanguageModel().to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {iter}: train: total={losses['train'][0]:.4f}, ori={losses['train'][1]:.4f}, gap={losses['train'][2]:.8f}\
                | val: total={losses['val'][0]:.4f}, ori={losses['val'][1]:.4f}, gap={losses['val'][2]:.8f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        _, loss, _, _ = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    # 保存训练好的模型
    torch.save(model.state_dict(), 'gpt_model0.pth')
    print("Model saved to gpt_model0.pth")

def visualize_similarity():
    # 初始化模型
    model0 = GPTLanguageModel().to(device)
    # 加载训练好的权重
    model0.load_state_dict(torch.load('gpt_model0.pth'))
    model0.eval()
    print("Model loaded from gpt_model0.pth")
    # 获取token嵌入权重
    token_emb0 = model0.token_embedding_table.weight.data  # (vocab_size, n_embd)
    token_emb_norm0 = token_emb0 / torch.norm(token_emb0, dim=1, keepdim=True)
    # 计算余弦相似度矩阵
    distance_matrix0 = torch.mm(token_emb_norm0, token_emb_norm0.t()).cpu().numpy()  # (vocab_size, vocab_size)

    # 初始化模型
    model = GPTLanguageModel().to(device)
    # 加载训练好的权重
    model.load_state_dict(torch.load('gpt_model.pth'))
    model.eval()
    print("Model loaded from gpt_model.pth")
    # 获取token嵌入权重
    token_emb = model.token_embedding_table.weight.data  # (vocab_size, n_embd)
    token_emb_norm = token_emb / torch.norm(token_emb, dim=1, keepdim=True)
    # 计算余弦相似度矩阵
    distance_matrix = torch.mm(token_emb_norm, token_emb_norm.t()).cpu().numpy()  # (vocab_size, vocab_size)    
    
    # 使用层次聚类对token进行排序
    Z = linkage(distance_matrix, method='ward')
    cluster_order = leaves_list(Z)  # 获取聚类叶节点顺序
    
    # 根据聚类结果重新排列相似度矩阵
    reordered_sim = distance_matrix0[cluster_order, :][:, cluster_order]
    # 应用指数函数增强可视化效果
    # similarity_matrix = np.exp(reordered_sim)
    similarity_matrix = reordered_sim
    # 3. 可视化相似度矩阵
    plt.figure(figsize=(15, 15))
    plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
    # 添加颜色条
    cbar = plt.colorbar()
    cbar.set_label('ct-cosine similarity', rotation=270, labelpad=20)
    # 设置标题和轴标签
    plt.title('Token Embedding Similarity Matrix')
    plt.xlabel('Token Index')
    plt.ylabel('Token Index')
    # 保存可视化结果
    plt.savefig('token_similarity_heatmap0.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Similarity matrix visualization saved to token_similarity_heatmap0.png")

if __name__ == "__main__":
    train_model()
    visualize_similarity()
