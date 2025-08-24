import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from datetime import datetime
from scipy.cluster.hierarchy import linkage, leaves_list
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from hm_gpt import GPTLanguageModel
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ------------

torch.manual_seed(1337)

def visualize_similarity(xi):
    model = GPTLanguageModel().to(device)
    model.load_state_dict(torch.load('gpt_model_dy.pth'))
    model.eval()
    print("Model loaded from gpt_model_dy.pth")
    # 获取token嵌入权重
    emb_eu = model.x[0]  # (vocab_size, n_embd)
    emb_ct = model.cte.locations[xi[0]]
    distance_eu = torch.matmul(emb_eu, emb_eu.transpose(0, 1)).cpu().numpy()
    distance_ct = model.cte.distance(emb_ct[0].unsqueeze(1), emb_ct[0].unsqueeze(0)).mean(dim=-1)  
    
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
    plt.savefig('token_similarity_heatmap_eu.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Similarity matrix visualization saved to token_similarity_heatmap_eu.png")

    reordered_sim = distance_ct[cluster_order, :][:, cluster_order]
    similarity_matrix = reordered_sim
    plt.figure(figsize=(15, 15))
    plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
    cbar = plt.colorbar()
    cbar.set_label('ct-cosine similarity', rotation=270, labelpad=20)
    plt.title('Token Embedding Similarity Matrix')
    plt.xlabel('Token Index')
    plt.ylabel('Token Index')
    plt.savefig('token_similarity_heatmap_ct.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Similarity matrix visualization saved to token_similarity_heatmap_ct.png")

if __name__ == "__main__":
    pass
