import os

import torch

from src.utils import LossTypeDict

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

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
epoch_cte=500
batch_size_cte=32
convergence=0.8
division_fact=2


loss_type: LossTypeDict = {
    'dyn_loss'  : 'square', # dyn:  dynamic
    'sta_loss'  : 'square', # sta:  static
    'prob_loss' : 'js' ,    # prob: probability
    50: {
        'target'  : 'dyn_only' , 
        'converge': 30
    }, #  0  < epoch <=  50 时仅优化 dyn_loss，在第 30 个 epoch 开始 converge
    100: {
        'target'  : 'sta_only' , 
        'converge': 80
    }, # 50  < epoch <= 100 时仅优化 sta_loss，在第 80 个 epoch 开始 converge
    125: {
        'target'  : 'alternated' , 
        'converge': 115
    }, # 100 < epoch <= 125 时交替优化，在第 115 个 epoch 开始 converge
    150: {
        'target'  : 'prob_only' , 
        'converge': 125
    }, # 125 < epoch <= 150 时交替优化，在第 125 个 epoch 开始 converge
    500: {
        'target'  : 'weighted_dyn_prob',
        'converge': 150,
        'ratio_dyn' : 0.98,
        'ratio_prob': 0.02   
    }
}
### dyn_loss / sta_loss 可选: 'abs', 'square'
### prob_loss 可选: 'kl' , 'js'
### method    可选: 
##### 'name': 'dyn_only'  , 表示当前仅优化 dyn_loss， 其它 loss 只计算、不优化
##### 'name': 'sta_only'  , 表示当前仅优化 sta_loss， 其它 loss 只计算、不优化
##### 'name': 'prob_only' , 表示当前仅优化 prob_loss，其它 loss 只计算、不优化
##### 'name': 'alternated', 表示交替优化, 一个 epoch 依次优化 dyn / sta / prob
##### 'name': 'weighted_dyn_prob'  , 表示加权优化. 需要额外指定 'ratio_dyn' 和 'ratio_prob', 且它们加和为 1

# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('./data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


print(f"数据集总长度：{len(text)}")

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# 额外内容
gpt_path = './ckpt/gpt'
sim_eu_path = './vis/sim_eu.png'
sim_ct_path = './vis/sim_ct.png'