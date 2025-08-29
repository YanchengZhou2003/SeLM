import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from torch.nn import functional as F

from src.para import batch_size, block_size, vocab_size
from src.loom_kernel import triton_loom_wrapper
from src.utils import *

main_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
from typing import List, Tuple

# ------------

torch.manual_seed(1337)

class CritiGraph(torch.nn.Module):
    main_distance_lookup_table: torch.Tensor
    main_locations: torch.Tensor
    
    def __init__(self, h, tp, c, eps, epoch, batch_size, convergence, emb_size, blk_size, division_fact, loss_type):
        super().__init__() 
        self.h = h
        self.tp = tp
        self.n = int(2**h)
        self.c = c
        self.k = int(c*h // division_fact)
        
        ### ---- 设备/进程信息 ---- ###
        self.devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        self.streams = [torch.cuda.Stream(device=dev) for dev in self.devices]
        
        # self.k = 1
        self.eps = eps
        self.epoch = epoch  
        self.batch_size = batch_size
        self.flip_masks = (1 << torch.arange(self.h, dtype=torch.int64, device=main_device)).unsqueeze(0).unsqueeze(2) # (1, H, 1)
        self.flip_masks = [self.flip_masks.clone().to(dev) for dev in self.devices] 
        self.distance_lookup_table = self.generate_distance_lookup_table()
        self.distance_lookup_table = [self.distance_lookup_table.clone().to(dev) for dev in self.devices]
        self.convergence = convergence
        self.emb_size = emb_size
        self.blk_size = blk_size
        self.locations = torch.randint(1 - self.n, self.n, (self.emb_size, self.tp), dtype=torch.int64, device=main_device)
        self.locations = [self.locations.clone().to(dev) for dev in self.devices]
        self.loss_type = loss_type
        
        self.register_buffer('main_locations', self.locations[0].clone().cpu())
        self.main_distance_lookup_table = self.distance_lookup_table[0].clone().cpu()
        
    def generate_distance_lookup_table(self):
        xor_results = torch.arange(self.n, dtype=torch.int64, device=main_device)
        return (torch.floor(torch.log2(xor_results.float() + 1)) + 1) / self.h
    
    # @torch.jit
    def distance(self, coord1: torch.Tensor, coord2: torch.Tensor, norm: torch.Tensor, dev_num=0):
        # sg = torch.sign(coord1) * torch.sign(coord2)
        sg = (((coord1 >= 0).to(torch.int16) << 1) - 1) * (((coord2 >= 0).to(torch.int16) << 1) - 1)
        # sg = 1 - (((coord1 >= 0) ^ (coord2 >= 0)).to(torch.int16) << 1) 
        
        xor_result = torch.bitwise_xor(torch.abs(coord1), torch.abs(coord2))
        # 关键替换：用 frexp 得到 log2_floor+1
        # _, exp = torch.frexp((xor_result + 1).to(torch.float32))
        # s = exp.float() / self.h
        s = self.distance_lookup_table[dev_num][xor_result]
        # sg * (1 - s): shape = (B, T, T, D) 或者 (B, T, T, C, D)
        return sg * (1 - s) * norm
    
    def main_distance(self, coord1, coord2, x_norm):
        coord1, coord2, x_norm = coord1.detach().clone().cpu(), coord2.detach().clone().cpu(), x_norm.detach().clone().cpu()
        sg = (((coord1 >= 0).to(torch.int16) << 1) - 1) * (((coord2 >= 0).to(torch.int16) << 1) - 1)
        xor_result = torch.bitwise_xor(torch.abs(coord1), torch.abs(coord2))
        s = self.main_distance_lookup_table[xor_result]
        cosine_similarity = sg * (1 - s)
        inner_similarity = cosine_similarity * x_norm[..., None] # (B, T, T, ...) * (B, T, T)
        return inner_similarity
    
    def generate_random_masks(self, sz, dev_num=0):
        upper_bounds = 2**torch.arange(self.h, dtype=torch.int64, device=self.devices[dev_num])
        random_numbers = torch.randint(0, self.n, (self.h, sz, self.k, self.tp), dtype=torch.int64, device=self.devices[dev_num]) # (H, B*T, K, D)
        masks = random_numbers % upper_bounds.view(-1, 1, 1, 1)
        return masks.permute(1, 0, 2, 3) # (B*T, H, K, D)
    def connection(self, ori_int, dev_num=0):
        flipped_ints = ori_int.unsqueeze(1) ^ self.flip_masks[dev_num] # (B*T1, H, D)
        random_masks = self.generate_random_masks(flipped_ints.size(0), dev_num=dev_num)
        result = (flipped_ints.unsqueeze(2) ^ random_masks).view(flipped_ints.size(0), self.h*self.k, self.tp)
        # (B*T1, H, 1, D) ^ (B*T1, H, K, D) -> (B*T1, H*K, D)
        loc = torch.cat((result, ori_int.unsqueeze(1), -result), dim=1) # (B*T1, H*K + 1 + H*K, D)
        indices = torch.randperm(loc.size(2), device=self.devices[dev_num]) 
        return loc[:, :, indices, :]
    
    def calc_loss(self, 
                  ct_val : torch.Tensor, eu_val : torch.Tensor, 
                  mask   : torch.Tensor, lth    : torch.Tensor,
                  epoch  : int = 0     , mode   : str = 'dyn' ,loss_type: Dict = {}
    ) -> torch.Tensor:
        ### ct_val: (B, T1, T2, C, D), eu_val: (B, T1, T2, C, D)
        ### mask  : (B, T1, T2, 1, 1), lth   : (
        
        ### step 1: 获取基本信息
        strategy = get_strategy(loss_type, epoch)
        cos_loss_type, prob_loss_type, method = strategy['cos_loss'], strategy['prob_loss'], strategy['method']['name']
        
        
        ### step 2: 计算 loss
        if mode == 'dyn' or mode == 'sta':
            if cos_loss_type == 'square':
                loss = torch.square(ct_val - eu_val) # (B, T1, T2, C, tp)
                loss = loss * mask                   # (B, T1, T2, C, tp)
        
        if mode == 'prob':
            if prob_loss_type == 'kl':
                loss = F.kl_div(
                    F.log_softmax(ct_val, dim=2),
                    F.log_softmax(eu_val, dim=2),
                    log_target=True,
                    reduction='none'
                ).sum(dim=2)        # (B, T1, T2, C, tp) -> (B, T1, C, tp) 
        
        
        return loss # (B, T+V, C, tp)
        
        
    
    @timeline(name='loom')
    @torch.no_grad()
    def loom(self, 
             epoch     : int,           #                , 代表当前是第几个 epoch 
             b_sta     : torch.Tensor,  # (cB, T1)       , 代表 每个样本在 locations 中的 id        (v 代指 value)
             b_pos     : torch.Tensor,  # (cB, T2)       , 代表 每个 pos 样本在 locations 中的 id               
             b_val_v   : torch.Tensor,  # (cB, T1, T2)   , 代表 需要拟合的值
             # b_val_m   : torch.Tensor,  # (cB, T1, T2)   , 代表 对应位置的值是否有效
             b_val_n   : torch.Tensor, # (cB, T1, T2)   , 代表 欧式空间的范数乘积
             # cB 是 current batch size 的意思
             # (cB, T1, T2) = (B, T, T+B)     的时候，表示仅拟合 dynamic embeddings
             # (cB, T1, T2) = (1, V, V)       的时候，表示仅拟合 static  embeddings
             # (cB, T1, T2) = (B, T, V)       的时候，表示仅拟合 probability distribution
             # (cB, T1, T2) = (B+1, T, T+B+V) 的时候，表示三者联合优化
             mode: str
    ):

        cB, T1, T2 = b_val_v.shape[0], b_val_v.shape[1], b_val_v.shape[2]
        splits = torch.linspace(0, cB, len(self.devices) + 1, dtype=torch.int64).tolist()
        splits = list(map(int, splits))
        
        for i, (dev, stream, (s, e)) in enumerate(zip(self.devices, self.streams, zip(splits[:-1], splits[1:]))):
            if s == e: continue
            B = e - s
            dev_num = i
            
            with torch.cuda.device(dev), torch.cuda.stream(stream): # type: ignore
                ### 通信1：数据传输开始 ###        
                sta, pos, val_v, val_n = to_dev(
                    b_sta, b_pos, b_val_v, b_val_n, 
                    device=dev, s=s, e=e
                )
                ### 通信1：数据传输结束 ###
                
                
                ### 计算：计算开始 ###
                #### step 1: 获取基本信息
                sta_loc = self.locations[i][sta] # (B, T1, tp)
                pos_loc = self.locations[i][pos] # (B, T2, tp)
                cnc_loc = self.connection(
                    torch.abs(sta_loc.view(-1, self.tp)), 
                    dev_num=dev_num
                ).view(B, T1, -1, self.tp)       # (B, T1, C, tp)
                
                
                '''
                # 用 Triton 算子替换掉之前的所有计算, 传递所有需要的输入张量
                # selected_locs, min_loss = triton_loom_wrapper(
                #     sta_loc=sta_loc,
                #     cnc_loc=cnc_loc,
                #     logits=logits,
                #     x_norm=x_norm,
                #     lg=lg,
                #     mask=mask,
                # )
                # tl = min_loss.mean() * B
                '''
                
                #### step 2: 获取候选位置（的值）
                dis_sta_pos     = self.distance(
                    sta_loc[:, :, None, :]   , pos_loc[:, None, :, :]     , val_n[..., None]      , dev_num=dev_num
                )            # (B, T1, T2, tp)
                dis_sta_pos_sum = dis_sta_pos.sum(dim=-1) 
                             # (B, T1, T2)
                dis_cnc_pos     = self.distance(
                    cnc_loc[:, :, None, :, :], pos_loc[:, None, :, None,:], val_n[..., None, None], dev_num=dev_num
                )            # (B, T1, T2, C, tp)
                ct_val          = (
                    dis_sta_pos_sum[:, :, :, None, None] - dis_sta_pos[:, :, :, None, :] + dis_cnc_pos
                ) / self.tp  #   (B, T1, T2, C, tp)
                             #    对于 B 个 batch，T1 个 starting point，向 T2 个 positive sample 连边。此时，我们把其中某个 positive sample 替换为 connected sample，共有 C 个；此时，D 个维度上的的距离是多少？
                
                #### step 3: 计算 loss
                ct_val    = ct_val                                                                # (B, T1, T2, C, tp)
                eu_val    = val_v.expand(ct_val.shape)                                            # (B, T1, T2, C, tp)
                mask, lth = self.neighbor_batch(cB, T1, T2, epoch, dev_num=dev_num)               # (B, T1, T2)
                loss      = self.calc_loss(ct_val, eu_val, mask[..., None, None], lth[..., None, None], 
                                           epoch=epoch, mode=mode, loss_type=self.loss_type)                 # (B, T1, C, D)
                
                
                
                
                indices = torch.argmin(loss, dim=2) 
                # total_loss = torch.abs(delt).sum(dim=2) / n_lth[:,:,None, None] # (B, T1, C, D)               
                # index = torch.argmin(total_loss, dim=2) # (B, T, D)
                batch_indices = torch.arange(B, device=dev)[:, None, None]
                time_indices = torch.arange(self.blk_size, device=dev)[None, :, None]
                dim_indices = torch.arange(self.tp, device=dev)[None, None, :]
                selected_locs = cnc_loc[batch_indices, time_indices, index, dim_indices]
                tl = total_loss[batch_indices, time_indices, index, dim_indices].mean() * B
                # '''
                ### 计算：计算结束 ###
                
                ### 通信2：数据传输开始 ###
                all_selected_locs[s:e].copy_(selected_locs.to(dtype=torch.int64), non_blocking=True)
                all_loss[i].copy_(tl, non_blocking=True)
                ### 通信2：数据传输结束 ###
        
        for i, stream in enumerate(self.streams):
            stream.synchronize()
        
        ### 更新与计算 ###
        for i, dev in enumerate(self.devices):
            self.locations[i].index_copy_(
                0,                                             # dim=0, 沿行更新
                batch[0].to(dev, non_blocking=True).view(-1),     # 哪些行
                all_selected_locs.to(dev, non_blocking=True).view(-1, self.tp)   # 更新的数据
            )
        
        return all_loss.sum().item() / all_B

    '''
    def neighbor_batch(self, B: int, T1: int, T2: int, epoch: int, dev_num=0):
        if epoch > self.convergence * self.epoch:
            return torch.ones((B, T1, T2), dtype=torch.bool, device=self.devices[dev_num])
        random_probs = torch.rand((B, T1), device=self.devices[dev_num]) # (B, T1)
        choosing_mask = random_probs > 0.2 # (B, T1)
        valid_lengths = torch.where(choosing_mask, T1, 1) # (B, T1)
        valid_mask = torch.ones((B, T1, T2), dtype=torch.bool, device=self.devices[dev_num])
        mask_1 = valid_lengths == 1  # (B, T1)
        if mask_1.any():
            batch_idx, row_idx = torch.where(mask_1)
            col_idx = torch.randint(0, T2, (len(batch_idx), ), device=self.devices[dev_num], dtype=torch.long)
            indices = torch.stack([batch_idx, row_idx, col_idx], dim=1)
            valid_mask[mask_1] = False
            valid_mask[tuple(indices.T)] = True
        return valid_mask # (B, T1, T2)   
    ''' 
    
    def neighbor_batch(self, B: int, T1: int, T2: int, epoch: int, dev_num: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.devices[dev_num]

        # 收敛时行为
        if epoch > self.convergence * self.epoch:
            return torch.ones((B, T1, T2), dtype=torch.bool, device=device)
        
        # 随机时行为
        choosing_mask = torch.rand((B, T1), device=device) > 0.2              # (B, T1), 有 80% 的位置为 True, 代表全选
        valid_mask = torch.ones((B, T1, T2), dtype=torch.bool, device=device) # (B, T1, T2), 全部置为 True
        not_chosen = ~choosing_mask                                           # (B, T1), 有 20% 的位置为 True, 代表只留 1 个邻居
        if not_chosen.any():
            b_idx, t1_idx = torch.where(not_chosen)                           # 找出需要保留 1 个邻居的索引
            valid_mask[b_idx, t1_idx, :] = False                              # 对应索引的邻居全部置为 0
            t2_idx = torch.randint(T2, (b_idx.size(0),), device=device)       # 随机一个点
            valid_mask[b_idx, t1_idx, t2_idx] = True                          # 置为 True

        return valid_mask, valid_mask.sum(dim=-1).float()                     # (B, T1, T2)
    
    @timeline(name=f'cte 函数主体')
    def forward(self,        
                sta     : torch.Tensor, # (B, T+V)     , 代表 每个 sta 样本在 locations 中的 id 
                pos     : torch.Tensor, # (B, T+V)     , 代表 每个 pos 样本在 locations 中的 id 
                val_v   : torch.Tensor, # (B, T+V, T+V), 代表 欧式空间待拟合的值              
                val_m   : torch.Tensor, # (B, T+V, T+V), 代表 余弦相似度是否有效
                val_n   : torch.Tensor, # (B, T+V, T+V), 代表 欧式空间的范数乘积，除了 prob 对应的位置之外用 1.0 填充
                targets: torch.Tensor, mark: Optional[Mark] = None
        ): 
        assert mark is not None # 保证计时器存在
        T, V = block_size, vocab_size
        ### 1. 生成用于传输数据的张量
        mark("pinned 张量生成")
        pinned = pinned_copy_by_name(
            named(sta=sta, pos=pos, 
                  val_v=val_v, val_m=val_m, val_n=val_n)
        )
        (sta, pos, val_v, val_m, val_n) = (
            pinned['sta'], pinned['pos'], 
            pinned['val_v'], pinned['val_m'], pinned['val_n']
        ) 
        
        all_selected_locs = torch.zeros((idi.shape[0], idi.shape[1], self.tp), dtype=torch.int64, pin_memory=True)

        
        for epoch in range(self.epoch):   
            dyn  = self.loom(epoch, sta[:, :T]   , pos[:,   :T] , val_v[:,  :T,    :T]  , val_n[:,  :T,    :T]  ,
                             mode='dyn')
            sta  = self.loom(epoch, sta[0, T:T+V], pos[0, T:T+V], val_v[0, T:T+V, T:T+V], val_n[0, T:T+V, T:T+V],
                             mode='sta')
            prob = self.loom(epoch, sta[:, :T]   , pos[:, T:T+V], val_v[:,  :T,   T:T+V], val_n[:,  :T,   T:T+V],
                             mode='prob')
        
        

        
        self.main_locations = self.locations[0].clone().cpu()
        # return self.distance(self.locations[0][idi].unsqueeze(2), 
        #                      self.locations[0][idi].unsqueeze(1), 
        #                      (x_norm.view(batch_size, block_size + vocab_size, 1) * \
        #                      x_norm.view(batch_size, 1, block_size + vocab_size)).unsqueeze(-1)).mean(dim=-1)


if __name__ == "__main__":
    pass
