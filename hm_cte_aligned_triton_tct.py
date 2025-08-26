import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from loom_tritontest import triton_loom_wrapper

main_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
from typing import List, Tuple

# ------------

torch.manual_seed(1337)

class CritiGraph(torch.nn.Module):
    main_distance_lookup_table: torch.Tensor
    main_locations: torch.Tensor
    
    def __init__(self, h, tp, c, eps, epoch, batch_size, convergence, emb_size, blk_size, loss_calc):
        super().__init__() 
        self.h = h
        self.tp = tp
        self.n = int(2**h)
        self.c = c
                                            # 此处是个很重要的改动
        self.k = int(c*h)
        
        ### ---- 设备/进程信息 ---- ###
        self.devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        self.streams = [torch.cuda.Stream(device=dev) for dev in self.devices]
        
        # self.k = 1
        self.eps = eps
        self.epoch = epoch  
        self.batch_size = batch_size
        self.flip_masks = (1 << torch.arange(self.h, dtype=torch.int64, device=main_device)).unsqueeze(0).unsqueeze(2)
        self.flip_masks = [self.flip_masks.clone().to(dev) for dev in self.devices] 
        self.distance_lookup_table = self.generate_distance_lookup_table()
        self.distance_lookup_table = [self.distance_lookup_table.clone().to(dev) for dev in self.devices]
        self.convergence = convergence
        self.emb_size = emb_size
        self.blk_size = blk_size
        self.locations = torch.randint(1 - self.n, self.n, (self.emb_size, self.tp), dtype=torch.int64, device=main_device)
        self.locations = [self.locations.clone().to(dev) for dev in self.devices]
        
        self.register_buffer('main_locations', self.locations[0].clone().cpu())
        self.main_distance_lookup_table = self.distance_lookup_table[0].clone().cpu()
        self.loss_calc = loss_calc
        
    def generate_distance_lookup_table(self):
        xor_results = torch.arange(self.n, dtype=torch.int64, device=main_device)
        return (torch.floor(torch.log2(xor_results.float() + 1)) + 1) / self.h
    
    # @torch.jit
    def distance(self, coord1, coord2, dev_num=0):
        # sg = torch.sign(coord1) * torch.sign(coord2)
        sg = (((coord1 >= 0).to(torch.int16) << 1) - 1) * (((coord2 >= 0).to(torch.int16) << 1) - 1)
        # sg = 1 - (((coord1 >= 0) ^ (coord2 >= 0)).to(torch.int16) << 1) 
        
        xor_result = torch.bitwise_xor(torch.abs(coord1), torch.abs(coord2))
        # 关键替换：用 frexp 得到 log2_floor+1
        # _, exp = torch.frexp((xor_result + 1).to(torch.float32))
        # s = exp.float() / self.h
        s = self.distance_lookup_table[dev_num][xor_result]
        return sg * (1 - s)
    
    def main_distance(self, coord1, coord2):
        coord1, coord2 = coord1.clone().cpu(), coord2.clone().cpu()
        sg = 1 - (((coord1 >= 0) ^ (coord2 >= 0)).to(torch.int16) << 1) 
        xor_result = torch.bitwise_xor(torch.abs(coord1), torch.abs(coord2))
        s = self.main_distance_lookup_table[xor_result]
        return sg * (1 - s)
    
    def generate_random_masks(self, sz, dev_num=0):
        upper_bounds = 2**torch.arange(self.h, dtype=torch.int64, device=self.devices[dev_num])
        random_numbers = torch.randint(0, self.n, (self.h, sz, self.k, self.tp), dtype=torch.int64, device=self.devices[dev_num])
        masks = random_numbers % upper_bounds.view(-1, 1, 1, 1)
        return masks.permute(1, 0, 2, 3)
    def connection(self, ori_int, dev_num=0):
        flipped_ints = ori_int.unsqueeze(1) ^ self.flip_masks[dev_num]
        random_masks = self.generate_random_masks(flipped_ints.size(0), dev_num=dev_num)
        result = (flipped_ints.unsqueeze(2) ^ random_masks).view(flipped_ints.size(0), self.h*self.k, self.tp)
        return torch.cat((result, ori_int.unsqueeze(1), -result), dim=1) 
    
    @torch.no_grad()
    def loom(self, epoch, batch: Tuple[torch.Tensor, torch.Tensor],
             all_selected_locs: torch.Tensor,
             all_loss: torch.Tensor): # 注：传入张量在 CPU 的 pinned memory 上
        all_B, T = batch[0].shape
        splits = torch.linspace(0, all_B, len(self.devices) + 1, dtype=torch.int64).tolist()
        splits = list(map(int, splits))
        
        for i, (dev, stream, (s,e)) in enumerate(zip(self.devices, self.streams, zip(splits[:-1], splits[1:]))):
            if s == e: continue

            # print(f"tring device {dev}")
            with torch.cuda.device(dev), torch.cuda.stream(stream):
                ### 通信1：数据传输开始 ###        
                sta_ind, logits = batch[0][s:e], batch[1][s:e]
                sta_ind, logits = sta_ind.to(dev, non_blocking=True), logits.to(dev, non_blocking=True)
                ### 通信1：数据传输结束 ###
                
                
                ### 计算：计算开始 ###
                B = e - s
                dev_num = i
                sta_loc = self.locations[i][sta_ind]
                
                lg, mask = self.neighbor_batch(sta_ind, epoch, dev_num=dev_num) # (B, T), (B, T, T)
                abs_sta_loc = torch.abs(sta_loc.view(-1, self.tp))
                cnc_loc = self.connection(abs_sta_loc, dev_num=dev_num).view(B, self.blk_size, -1, self.tp) #(B, T, C, D)
                indices = torch.randperm(cnc_loc.size(2), device=dev) 
                cnc_loc = cnc_loc[:, :, indices, :]
                
                # 用 Triton 算子替换掉之前的所有计算, 传递所有需要的输入张量
                selected_locs, min_loss = triton_loom_wrapper(
                    sta_loc=sta_loc,
                    cnc_loc=cnc_loc,
                    logits=logits,
                    lg=lg,
                    mask=mask,
                    distance_lookup_table=self.distance_lookup_table[dev_num],
                    tp=self.tp,
                    H=self.h,
                    loss_calc=self.loss_calc
                )
                tl = min_loss.mean() * B
                
                '''原来的 pytorch 逻辑
                mask0 = mask.unsqueeze(-1).unsqueeze(-1)
                dis_sta_pos = self.distance(sta_loc[:,:,None,:], sta_loc[:,None,:,:], dev_num=dev_num) # (B, T, T, D)
                dis_sta_posum = dis_sta_pos.sum(dim=-1) # (B, T, T)
                # # -------- Origin --------- #
                logits_ct = self.distance(sta_loc[:,None,:,None,:], cnc_loc[:,:,None,:,:], dev_num=dev_num) # (B, T, T, C, D)
                logits_ct = (logits_ct-dis_sta_pos[:,:,:,None,:]+dis_sta_posum[:,:,:,None,None]) / self.tp # (B, T, T, C, D)
                logits_ct = logits_ct * mask0 # (B, T, T, C, D)
                logits_eu = logits.unsqueeze(3).unsqueeze(4) * mask0 # (B, T, T, C, D)
                delt = logits_ct - logits_eu
                total_loss = torch.abs(delt).sum(dim=2) / lg[:,:,None, None] # (B, T, C, D)               
                index = torch.argmin(total_loss, dim=2) # (B, T, D)
                batch_indices = torch.arange(B, device=dev)[:, None, None]
                time_indices = torch.arange(self.blk_size, device=dev)[None, :, None]
                dim_indices = torch.arange(self.tp, device=dev)[None, None, :]
                selected_locs = cnc_loc[batch_indices, time_indices, index, dim_indices]
                tl = total_loss[batch_indices, time_indices, index, dim_indices].mean() * B
                '''
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

    def neighbor_batch(self, sta_ind, epoch, dev_num=0):
        B, T = sta_ind.size(0), self.blk_size
        if epoch > self.convergence * self.epoch:
            return torch.full((B, T), T, dtype=torch.int64, device=self.devices[dev_num]), \
                   torch.ones((B, T, T), dtype=torch.bool, device=self.devices[dev_num])
        random_probs = torch.rand((B, T), device=self.devices[dev_num]) # (B,T)
        choosing_mask = random_probs > 0.2 # (B, T)
        batch_lengths = torch.where(choosing_mask, T, 1) # (B, T)
        batch_mask = torch.ones((B, T, T), dtype=torch.bool, device=self.devices[dev_num])
        mask_1 = batch_lengths == 1  # 维度 (B, T)
        if mask_1.any():
            # batch_idx, row_idx = torch.where(mask_1)
            # col_idx = torch.randint(0, T, (len(batch_idx),), device=main_device)
            # row_mask = mask_1.unsqueeze(-1).expand(B, T, T)
            # batch_mask[row_mask] = False  # 先将所有 mask_1 的行设为 False
            # batch_mask[batch_idx, row_idx, col_idx] = True  
            batch_idx, row_idx = torch.where(mask_1)
            col_idx = torch.randint(0, T, (len(batch_idx),), device=self.devices[dev_num], dtype=torch.long)
            indices = torch.stack([batch_idx, row_idx, col_idx], dim=1)
            batch_mask[mask_1] = False
            batch_mask[tuple(indices.T)] = True
        return batch_lengths, batch_mask   
    
    def forward(self, idi, dismatrix_eu):
        st = time.perf_counter()
        
        current_time = datetime.now()
        # print("current_time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
        # dataset = TensorDataset(idi, dismatrix_eu)
        # dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        idi_pinned = torch.empty_like(idi, device='cpu', pin_memory=True, dtype=torch.int64)
        dismatrix_eu_pinned = torch.empty_like(dismatrix_eu, device='cpu', pin_memory=True)
        idi_pinned.copy_(idi)
        dismatrix_eu_pinned.copy_(dismatrix_eu)
        
        
        all_selected_locs = torch.zeros((idi.shape[0], idi.shape[1], self.tp), dtype=torch.int64, pin_memory=True)
        all_loss = torch.zeros(len(self.devices), dtype=torch.float, pin_memory=True)
        # jug = 0
        epoch = 0
        # for epoch in range(self.epoch):
        epoch_max = self.epoch
        # epoch_thr = int(self.convergence * epoch_max)
        while epoch < epoch_max:    
            current_time = datetime.now()
            perm = torch.randperm(idi_pinned.size(0), device='cpu')
            idi_pinned = idi_pinned[perm]
            dismatrix_eu_pinned = dismatrix_eu_pinned[perm]
            # print("current_time:", current_time.strftime("%Y-%m-%d %H:%M:%S.%f"))
            total, ite = 0, 0
            
            # st = time.perf_counter()
            tot = self.loom(epoch, (idi_pinned, dismatrix_eu_pinned),
                            all_selected_locs,
                            all_loss)
            # if jug == 0 and epoch < epoch_thr:
            #     if tot <= 0.025:
            #         epoch_max = int(epoch * 1.25) + 1
            #         jug = 1
            #         print('epoch_max:', epoch_max)
            epoch += 1       
            total += tot
            ite += 1
            total /= ite
            # ed = time.perf_counter()
            # print(f"Time Elapsed: {(ed - st) * 1000} ms")
            
            # print(epoch, 'average KL divergence:', total)
        
        ed = time.perf_counter()
        
        # print(f"Time Elapsed: {(ed - st) * 1000.0} ms")
        
        self.main_locations = self.locations[0].clone().cpu()
        return self.distance(self.locations[0][idi].unsqueeze(2), self.locations[0][idi].unsqueeze(1)).mean(dim=-1)


if __name__ == "__main__":
    pass
