import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from torch.utils.data import Dataset, DataLoader
import networkx as nx
import torch
import sys
from hm_gpt_aligned_triton import *

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

torch.manual_seed(1337)

# >>> Dataset and Model <<< #
class MyDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        sta_ind = self.indices[idx]
        return sta_ind.clone().detach().to(dtype=torch.int64).cuda() 
class CritiGraph(torch.nn.Module):
    def __init__(self, h, tp, c, eps, epoch, batch_size, convergence, vocab_size):
        super().__init__() 
        self.h = h
        self.tp = tp
        self.n = int(2**h)
        self.c = c
        # self.k = int(c*h)
        # self.k = 1
        self.k = int(int(c*h) // 2)
        self.eps = eps
        self.epoch = epoch  
        self.batch_size = batch_size
        self.flip_masks = (1 << torch.arange(self.h, dtype=torch.int64, device=device)).unsqueeze(0).unsqueeze(2)
        self.distance_lookup_table = self.generate_distance_lookup_table()
        self.convergence = convergence
        self.vocab_size = vocab_size
        locations = torch.randint(1 - self.n, self.n, (self.vocab_size, self.tp), dtype=torch.int64, device=device)
        self.register_buffer('locations', locations)
    def generate_distance_lookup_table(self):
        xor_results = torch.arange(self.n, dtype=torch.int64, device=device)
        return (torch.floor(torch.log2(xor_results.float() + 1)) + 1) / self.h
    def distance(self, coord1, coord2):
        sg = torch.sign(coord1) * torch.sign(coord2)
        xor_result = torch.bitwise_xor(torch.abs(coord1), torch.abs(coord2))
        s = self.distance_lookup_table[xor_result]
        return sg * (1 - s)
    def generate_random_masks(self, sz):
        upper_bounds = 2**torch.arange(self.h, dtype=torch.int64, device=device)
        random_numbers = torch.randint(0, self.n, (self.h, sz, self.k, self.tp), dtype=torch.int64, device=device)
        masks = random_numbers % upper_bounds.view(-1, 1, 1, 1)
        return masks.permute(1, 0, 2, 3)
    def connection(self, ori_int):
        flipped_ints = ori_int.unsqueeze(1) ^ self.flip_masks
        random_masks = self.generate_random_masks(flipped_ints.size(0))
        result = (flipped_ints.unsqueeze(2) ^ random_masks).view(flipped_ints.size(0), self.h*self.k, self.tp)
        return torch.cat((result, ori_int.unsqueeze(1), -result), dim=1) 
    def loom(self, epoch, sta_ind, logits):
        pos_ind, lg, mask = self.neighbor_batch(sta_ind, epoch) # (B, V), (B), (B, V)
        sta_loc = self.locations[sta_ind] # (B, D)
        with torch.no_grad():
            pos_loc = self.locations[pos_ind] # (B, V, D)
            cnc_loc = self.connection(torch.abs(self.locations[sta_ind])) #(B, C, D)
            indices = torch.randperm(cnc_loc.size(1))
            cnc_loc = cnc_loc[:, indices, :]
            mask0 = mask.unsqueeze(2).unsqueeze(3).repeat(1, 1, cnc_loc.size(1), self.tp).int() # (B, V, C, D)
            dis_sta_pos = self.distance(sta_loc[:,None,:], pos_loc) # (B, V, D)
            dis_sta_posum = torch.sum(dis_sta_pos, dim=-1) # (B, V)
            dis_pos_cnc = self.distance(cnc_loc[:,None,:,:], pos_loc[:,:,None,:]) # (B, V, C, D)
            dis_new_pos = (dis_pos_cnc-dis_sta_pos[:,:,None,:]+dis_sta_posum[:,:,None,None]) / self.tp # (B, V, C, D)
            logits_ct = dis_new_pos * mask0 # (B, V, C, D)
            logits_eu = logits[sta_ind].unsqueeze(2).unsqueeze(3).repeat(1, 1, cnc_loc.size(1), self.tp) * mask0 # (B, V, C, D)
            delt = logits_ct - logits_eu
            # total_loss = (delt * delt).sum(dim=1) / lg[:,None, None]
            total_loss = torch.abs(delt).sum(dim=1) / lg[:,None, None]
            index = torch.argmin(total_loss, dim=1)
            i_indices, j_indices = torch.meshgrid(torch.arange(sta_ind.size(0)), torch.arange(self.tp), indexing='ij')
            self.locations[sta_ind[i_indices], j_indices] = cnc_loc[i_indices, index[i_indices, j_indices], j_indices]
            tl = torch.mean(total_loss[i_indices, index[i_indices, j_indices], j_indices])
        return tl

    def get_neighbor(self):
        neighbor = self.li.clone().detach().repeat(self.vocab_size, 1)
        neighbor_dict = {ii: neighbor[ii] for ii in range(self.vocab_size)}
        neighbor_tensor = torch.full((self.vocab_size, self.vocab_size), -1, dtype=torch.int64, device=device)
        for n, nbs in neighbor_dict.items():
            neighbor_tensor[n, :len(nbs)] = nbs
        return neighbor_dict, neighbor_tensor
    def neighbor_batch(self, sta_ind, epoch):
        bs = sta_ind.size(0)
        batch_degree = self.degree[sta_ind] # (bs)
        batch_max_degree = batch_degree.max().item()
        batch_neighbor1 = self.neighbor_tensor[sta_ind, :batch_max_degree] # (bs, batch_max_degree), -1 for padding
        
        if epoch > self.convergence * self.epoch:
            batch_lengths = batch_degree
            batch_mask = batch_neighbor1 != -1
        else:
            random_probs = torch.rand(bs, device=device) # (bs, )
            choosing_mask = random_probs > 0.2 # (bs, )
            batch_lengths = torch.where(choosing_mask, batch_degree, 1)
            
            one_random_neighbor = (torch.rand(bs, device=device) * batch_degree).floor().long()
            random_neighbor_mask = torch.zeros((bs, batch_max_degree), dtype=torch.bool, device=device) # (bs, max_degree)
            random_neighbor_mask[torch.arange(bs), one_random_neighbor] = True # (bs, max_degree)
            one_random_neighbor = torch.full((bs, batch_max_degree), -1, dtype=torch.int64, device=device)
            one_random_neighbor[:, 0] = batch_neighbor1[random_neighbor_mask] # (bs, max_degree)            
            
            batch_neighbor = torch.where(choosing_mask.unsqueeze(1), 
                                         batch_neighbor1,
                                         one_random_neighbor) # (bs, max_degree)
            batch_neighbor = batch_neighbor[:, :batch_lengths.max()]
            batch_mask = batch_neighbor != -1
        return batch_neighbor1, batch_lengths, batch_mask   
    
    def forward(self, eu_emb):
        current_time = datetime.now()
        print("current_time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
        print("start to load data")        
        print("current_time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
        self.li = torch.arange(self.vocab_size, dtype=torch.int64, device=device)
        # self.degree = torch.IntTensor([self.vocab_size for i in range(self.vocab_size)]).cuda().to(torch.int64)
        self.degree = torch.full((self.vocab_size,), self.vocab_size, dtype=torch.int64).cuda() 
        self.neighbor, self.neighbor_tensor = self.get_neighbor()
        # self.locations = torch.randint(1 - self.n, self.n, (self.vocab_size, self.tp), dtype=torch.int64, device=device)
        dataset = MyDataset(self.li)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        logits = torch.matmul(eu_emb, eu_emb.t()) # (V, V)
        # print(logits)

        for epoch in range(self.epoch):    
            current_time = datetime.now()
            print("current_time:", current_time.strftime("%Y-%m-%d %H:%M:%S.%f"))
            total, ite = 0, 0
            for batch in dataloader:
                tot = self.loom(epoch, batch, logits)
                total += tot
                ite += 1
            total /= ite
            print(epoch, 'average KL divergence:', total.item())

def visualize_similarity():
    model_CT.load_state_dict(torch.load('gpt_model_CT.pth'))
    model_CT.eval()
    print("Model loaded from gpt_model_CT.pth") 
    token_emb_CT = model_CT.locations  # (vocab_size, n_embd)
    distance_matrix_CT = (model_CT.distance(token_emb_CT.unsqueeze(1), token_emb_CT.unsqueeze(0)).sum(dim=-1) / model_CT.tp).cpu().numpy()  # (vocab_size, vocab_size)
    model.load_state_dict(torch.load('gpt_model_dy.pth'))
    model.eval()
    print("Model loaded from gpt_model_dy.pth")
    emb_eu = model.x[0]
    distance_matrix = torch.matmul(emb_eu, emb_eu.transpose(0, 1)).cpu().numpy()

    # 使用层次聚类对token进行排序
    Z = linkage(distance_matrix, method='ward')
    cluster_order = leaves_list(Z)  # 获取聚类叶节点顺序
    # 根据聚类结果重新排列相似度矩阵
    reordered_sim = distance_matrix[cluster_order, :][:, cluster_order]
    reordered_sim_CT = distance_matrix_CT[cluster_order, :][:, cluster_order]
    # 应用指数函数增强可视化效果
    similarity_matrix = reordered_sim
    similarity_matrix_CT = reordered_sim_CT
    
    # 3. 可视化相似度矩阵
    plt.figure(figsize=(15, 15))
    plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
    # 添加颜色条
    cbar = plt.colorbar()
    cbar.set_label('cosine similarity', rotation=270, labelpad=20)
    # 设置标题和轴标签
    plt.title('Token Embedding Similarity Matrix')
    plt.xlabel('Token Index')
    plt.ylabel('Token Index')
    # 保存可视化结果
    plt.savefig('token_similarity_heatmap.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Similarity matrix visualization saved to token_similarity_heatmap.png")

    plt.figure(figsize=(15, 15))
    plt.imshow(similarity_matrix_CT, cmap='viridis', interpolation='nearest')
    # 添加颜色条
    cbar = plt.colorbar()
    cbar.set_label('complete-tree similarity', rotation=270, labelpad=20)
    # 设置标题和轴标签
    plt.title('Token Embedding Similarity Matrix_CT')
    plt.xlabel('Token Index')
    plt.ylabel('Token Index')
    # 保存可视化结果
    plt.savefig('token_similarity_heatmap_CT.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Similarity matrix visualization saved to token_similarity_heatmap_CT.png")


if __name__ == "__main__":
    model = GPTLanguageModel().to(device)
    model.load_state_dict(torch.load('gpt_model_dy.pth'))
    model.eval()
    print("Model loaded from gpt_model_dy.pth")
    
    # 获取token嵌入权重
    token_emb = model.x[0]

    vocab_size = 321
    # h尽可能大，tp在临界值下尽可能大, batch_size在存储限制下尽可能大, L1范数对齐
    model_CT = CritiGraph(h=27, tp=2, c=1, eps=1e-5, epoch=50, batch_size=321, convergence=0.8, vocab_size=vocab_size)
    model_CT(token_emb)
    torch.save(model_CT.state_dict(), 'gpt_model_CT.pth')
    print("Model saved to gpt_model_CT.pth")
    visualize_similarity()
