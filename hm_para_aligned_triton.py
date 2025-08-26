import os

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# hyperparameters - gpt
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 3e-4
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
eval_iters = 10
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
loss_calc = 1 # 若为 0 则代表 abs, 为 1 则代表 square

# hyperparameters - cte
h=27
tp=2
c=1 
eps=1e-5 
epoch_cte=50
batch_size_cte=64
convergence=0.8
division_fact=4

# hyperparameters - combination
ratio_cb = 100
epoch_threshold = 30

torch.manual_seed(1337)