# import torch
# from torch_geometric.nn import GraphConv
# class Model(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = GraphConv(-1, 16)
#         self.conv2 = GraphConv(16, 1)
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index).relu()
#         x = self.conv2(x, edge_index)
#         return x

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm, global_mean_pool

class Model(torch.nn.Module):
    def __init__(self, input_dim=-1, hidden_dim=16, output_dim=1, num_heads=4, dropout=0.3):
        super().__init__()
        
        # 第一層圖注意力層
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.bn1 = BatchNorm(hidden_dim * num_heads)
        
        # 第二層圖注意力層
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
        self.bn2 = BatchNorm(hidden_dim * num_heads)
        
        # 第三層圖注意力層
        self.conv3 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout)
        self.bn3 = BatchNorm(hidden_dim)
        
        # 全連接層
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = torch.nn.Linear(hidden_dim // 2, output_dim)
        
        # 激活函數和正則化
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        # 第一層圖注意力層 + 批量正規化 + 激活 + Dropout
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # 第二層圖注意力層 + 批量正規化 + 激活 + Dropout
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # 第三層圖注意力層 + 批量正規化 + 激活 + Dropout
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # 全連接層
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
