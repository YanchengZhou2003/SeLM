import os

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# hyperparameters - gpt
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 10000
save_interval = 1000
eval_interval = 1
learning_rate = 3e-4
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
eval_iters = 10
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# hyperparameters - ct
h=27
tp=2
c=1 
eps=1e-5 
epoch_cte=50
batch_size_cte=63
convergence=0.8
division_fact=4
loss_type = {
    400: {
        'cos_loss'  : 'square',
        'prob_loss' : 'kl' ,
        'method'    : {
            'name'  :'alternated' , 
        }
    }, # epoch <= 400 时使用混合策略
    500: {
        'cos_loss'  : 'square',
        'prob_loss' : 'kl' ,
        'method'    : {
            'name'      :'weighted' , 
            'ratio_cos' : 0.5       , 
            'ratio_prob': 0.5       ,
        }
    } # 400 < epoch <= 500 时使用加权策略
}
### cos_loss  可选: 'abs', 'square', 'lap'
### prob_loss 可选: 'kl' , 'jl'
### method    可选: 
##### 'name': 'alternated', 表示交替优化. 不需要额外参数
##### 'name': 'weighted'  , 表示加权优化. 需要额外指定 'ratio_cos' 和 'ratio_prob', 且它们加和为 1
##### 'name': 'merged'    , 表示合并优化. 需要额外指定 'k_loss' 和 'k',             表示对哪个 loss 选 top-k、以及 k 值

# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('./data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# 额外内容
gpt_path = './ckpt/gpt'