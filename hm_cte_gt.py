import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ------------

torch.manual_seed(1337)

class CritiGraph(torch.nn.Module):
    def __init__(self, h, tp, c, eps, epoch, batch_size, convergence, emb_size, blk_size):
        super().__init__() 
        self.h = h
        self.tp = tp
        self.n = int(2**h)
        self.c = c
                                            # 此处是个很重要的改动
        self.k = int(c*h) // 4
        # self.k = 1
        self.eps = eps
        self.epoch = epoch  
        self.batch_size = batch_size
        self.flip_masks = (1 << torch.arange(self.h, dtype=torch.int32, device=device)).unsqueeze(0).unsqueeze(2)
        self.distance_lookup_table = self.generate_distance_lookup_table()
        self.convergence = convergence
        self.emb_size = emb_size
        self.blk_size = blk_size
        locations = torch.randint(1 - self.n, self.n, (self.emb_size, self.tp), dtype=torch.int32, device=device)
        self.register_buffer('locations', locations)
    def generate_distance_lookup_table(self):
        xor_results = torch.arange(self.n, dtype=torch.int32, device=device)
        return (torch.floor(torch.log2(xor_results.float() + 1)) + 1) / self.h
    def distance(self, coord1, coord2):
        sg = torch.sign(coord1) * torch.sign(coord2)
        xor_result = torch.bitwise_xor(torch.abs(coord1), torch.abs(coord2))
        s = self.distance_lookup_table[xor_result]
        return sg * (1 - s)
    def generate_random_masks(self, sz):
        upper_bounds = 2**torch.arange(self.h, dtype=torch.int32, device=device)
        random_numbers = torch.randint(0, self.n, (self.h, sz, self.k, self.tp), dtype=torch.int32, device=device)
        masks = random_numbers % upper_bounds.view(-1, 1, 1, 1)
        return masks.permute(1, 0, 2, 3)
    def connection(self, ori_int):
        flipped_ints = ori_int.unsqueeze(1) ^ self.flip_masks
        random_masks = self.generate_random_masks(flipped_ints.size(0))
        result = (flipped_ints.unsqueeze(2) ^ random_masks).view(flipped_ints.size(0), self.h*self.k, self.tp)
        return torch.cat((result, ori_int.unsqueeze(1), -result), dim=1)
    def loom(self, epoch, batch):
        sta_ind, logits = batch
        lg, mask = self.neighbor_batch(sta_ind, epoch) # (B, T), (B, T, T)
        with torch.no_grad():
            sta_loc = self.locations[sta_ind] # (B, T, D)
            abs_sta_loc = torch.abs(sta_loc.view(-1, self.tp))
            cnc_loc = self.connection(abs_sta_loc).view(self.batch_size, self.blk_size, -1, self.tp) # (B, T, C, D)
            indices = torch.randperm(cnc_loc.size(2), device=device) 
            cnc_loc = cnc_loc[:, :, indices, :]
            mask0 = mask.unsqueeze(-1).unsqueeze(-1)
            dis_sta_pos = self.distance(sta_loc[:,:,None,:], sta_loc[:,None,:,:]) # (B, T, T, D)
            dis_sta_posum = dis_sta_pos.sum(dim=-1) # (B, T, T)
            logits_ct = self.distance(sta_loc[:,None,:,None,:], cnc_loc[:,:,None,:,:]) # (B, T, T, C, D)
            logits_ct = (logits_ct-dis_sta_pos[:,:,:,None,:]+dis_sta_posum[:,:,:,None,None]) / self.tp # (B, T, T, C, D)
            logits_ct = logits_ct * mask0 # (B, T, T, C, D)
            logits_eu = logits.unsqueeze(3).unsqueeze(4).repeat(1, 1, 1, cnc_loc.size(2), self.tp) * mask0 # (B, T, T, C, D)
            delt = logits_ct - logits_eu
            # print(logits_eu)
            # total_loss = (delt * delt).sum(dim=2) / lg[:,:,None, None] # (B, T, C, D)
            total_loss = torch.abs(delt).sum(dim=2) / lg[:,:,None, None] # (B, T, C, D)
            index = torch.argmin(total_loss, dim=2) # (B, T, D)
            batch_indices = torch.arange(self.batch_size, device=device)[:, None, None]
            time_indices = torch.arange(self.blk_size, device=device)[None, :, None]
            dim_indices = torch.arange(self.tp, device=device)[None, None, :]
            selected_locs = cnc_loc[batch_indices, time_indices, index, dim_indices]
            self.locations[sta_ind] = selected_locs
            tl = total_loss[batch_indices, time_indices, index, dim_indices].mean()
            # print(tl)
            del cnc_loc, logits_ct, logits_eu, delt
        return tl

    def neighbor_batch(self, sta_ind, epoch):
        B, T = sta_ind.size(0), self.blk_size
        if epoch > self.convergence * self.epoch:
            return torch.full((B, T), T, dtype=torch.int32, device=device), \
                   torch.ones((B, T, T), dtype=torch.bool, device=device)
        random_probs = torch.rand((B, T), device=device) # (B,T)
        choosing_mask = random_probs > 0.2 # (B, T)
        batch_lengths = torch.where(choosing_mask, T, 1) # (B, T)
        batch_mask = torch.ones((B, T, T), dtype=torch.bool, device=device)
        mask_1 = batch_lengths == 1  # 维度 (B, T)
        if mask_1.any():
            # batch_idx, row_idx = torch.where(mask_1)
            # col_idx = torch.randint(0, T, (len(batch_idx),), device=device)
            # row_mask = mask_1.unsqueeze(-1).expand(B, T, T)
            # batch_mask[row_mask] = False  # 先将所有 mask_1 的行设为 False
            # batch_mask[batch_idx, row_idx, col_idx] = True  
            batch_idx, row_idx = torch.where(mask_1)
            col_idx = torch.randint(0, T, (len(batch_idx),), device=device)
            indices = torch.stack([batch_idx, row_idx, col_idx], dim=1)
            batch_mask[mask_1] = False
            batch_mask[tuple(indices.T)] = True
        return batch_lengths, batch_mask   
    
    def forward(self, idi, dismatrix_eu):
        st = time.perf_counter()
        
        current_time = datetime.now()
        # print("current_time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
        dataset = TensorDataset(idi, dismatrix_eu)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epoch):    
            current_time = datetime.now()
            # print("current_time:", current_time.strftime("%Y-%m-%d %H:%M:%S.%f"))
            total, ite = 0, 0
            for batch in dataloader:
                tot = self.loom(epoch, batch)
                total += tot
                ite += 1
            total /= ite
            # print(epoch, 'average KL divergence:', total.item())
        
        ed = time.perf_counter()
        
        print(f"Time Elapsed: {(ed - st) * 1000.0} ms")
        
        return self.distance(self.locations[idi].unsqueeze(2), self.locations[idi].unsqueeze(1)).mean(dim=-1)

if __name__ == "__main__":
    pass
