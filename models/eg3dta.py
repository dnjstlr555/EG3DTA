import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import MW_MSG3DBlock
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.inbn = nn.BatchNorm1d(in_channels)
        self.q_proj = nn.Linear(in_channels, embed_dim)
        self.k_proj = nn.Linear(in_channels, embed_dim)
        self.v_proj = nn.Linear(in_channels, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x): 
        B, T, C = x.size()
        x = self.inbn(x.view(B*T, C)).view(B, T, C).contiguous()
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5) # (B, num_heads, T, T)
        attn_weights = F.softmax(attn_scores, dim=-1) 
        attn_output = attn_weights @ V 
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C) #Concat!
        return self.out_proj(attn_output), attn_weights

class MHAComp(nn.Module):
    def __init__(self, in_channels=96, num_heads=8, max_len=100):
        super().__init__()
        self.temporal_attention = MultiHeadAttention(in_channels, in_channels, num_heads)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, in_channels))
        self.norm1 = nn.LayerNorm(in_channels)

    def forward(self, x):
        B, C, T, V = x.shape
        x_in = x.permute(0, 3, 2, 1).contiguous().view(B * V, T, C)
        x_pe = x_in + self.pos_embed[:, :T, :]
        attn_out, weights = self.temporal_attention(self.norm1(x_pe))
        x_pe = x_pe + attn_out
        out = x_pe.view(B, V, T, C).permute(0, 3, 2, 1).contiguous()
        weights = weights.view(B, V, -1, T, T)
        return out, weights

class EG3DTA(nn.Module):
    def __init__(self,
                 in_channels=3,
                 base_channels=96,
                 num_gcn_scales=13,
                 num_g3d_scales=6,
                 num_person=2,
                 tcn_dropout=0,
                 input_size=100,
                 out_size=1,
                 graph_type="nturgb+d",
                 return_attn=False):
        super().__init__()
        if graph_type == "nturgb+d":
            neighbor_base = [
                (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                (20, 19), (22, 8), (23, 8), (24, 12), (25, 12)
            ]
            
        elif graph_type == "vicon":
            neighbor_base = [
                (1, 2), (2, 3), (3, 5), (4, 3),  
                (5, 6), (6, 9), 
                (7, 9), (8, 9), 
                (9, 10),
                (10, 6),
                (11, 7), (12, 11), (13, 12), (14, 13), (15, 14), (16, 15),
                (17, 8), (18, 17), (19, 18), (20, 19), (21, 20), (22, 21), (23, 22), 
                (24, 10), (25, 10),
                (26, 25), (27, 26),
                (28, 24), (29, 28), (30, 29), (31, 30), (32, 31), (33, 31),
                (34, 25), (35, 34), (36, 35), (37, 36), (38, 37), (39, 37) 
            ]
        neighbor_base = [(i - 1, j - 1) for (i, j) in neighbor_base]
        max_node = max(max(i, j) for (i, j) in neighbor_base) + 1
        adj = np.zeros((max_node, max_node))
        for i, j in neighbor_base:
            adj[i, j] = 1
            adj[j, i] = 1
        
        A = torch.tensor(adj, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        self.num_point = A.shape[-1]
        self.in_channels = in_channels
        self.base_channels = base_channels

        self.data_bn = nn.BatchNorm1d(self.num_point * in_channels * num_person)
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        
        self.gcn3d1 = MW_MSG3DBlock(in_channels, c1, A, num_g3d_scales,
                                    window_sizes=[3,5],
                                    window_stride=1,
                                    window_dilations=[1,1])
        
        self.attn1 = MHAComp(in_channels=c1, num_heads=4, max_len=input_size)
        
        self.gcn3d2 = MW_MSG3DBlock(c1, c2, A, num_g3d_scales,
                                    window_sizes=[3,5],
                                    window_stride=1,
                                    window_dilations=[1,1])
        self.attn2 = MHAComp(in_channels=c2, num_heads=4, max_len=input_size)
        
        self.gcn3d3 = MW_MSG3DBlock(c2, c3, A, num_g3d_scales,
                                    window_sizes=[3,5],
                                    window_stride=1,
                                    window_dilations=[1,1])
        self.attn3 = MHAComp(in_channels=c3, num_heads=4, max_len=input_size)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_cls = nn.Linear(c3, out_size) 
        self.return_attn = return_attn

    def forward(self, x):
        N, M, T, V, C = x.size()
        if self.return_attn:
            attns = []
        x = x.permute(0, 1, 3, 4, 2).contiguous().reshape(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.reshape(N * M, V, C, T).permute(0, 2, 3, 1).contiguous() # (N*M, C, T, V)
        
        x = F.relu(self.gcn3d1(x), inplace=True)
        x, attn = self.attn1(x) # N*M, C, T, V
        x = F.relu(x, inplace=True)
        if self.return_attn:
            attns.append(attn)

        x = F.relu(self.gcn3d2(x), inplace=True)
        x, attn = self.attn2(x) # N*M, C, T, V
        x = F.relu(x, inplace=True)
        if self.return_attn:
            attns.append(attn)

        x = F.relu(self.gcn3d3(x), inplace=True)
        x, attn = self.attn3(x) # N*M, C, T, V
        x = F.relu(x, inplace=True)
        if self.return_attn:
            attns.append(attn)
        
        x = self.pool(x)
        x = x.reshape(N, M, -1)
        x = x.mean(dim=1)

        cls_score = self.fc_cls(x)
        if self.return_attn:
            return cls_score, attns
        return cls_score


    def init_weights(self):
        pass
