# dataset/custom_opf_dataset.py

import os
import json
import torch
from torch_geometric.data import Dataset, HeteroData
import pypower.idx_brch as pybrch
import pypower.idx_bus as pybus

class CustomOPFDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(CustomOPFDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data_files = [f for f in os.listdir(os.path.join(root, 'case9_data')) if f.endswith('.json')]
    
    def len(self):
        return len(self.data_files)
    
    def get(self, idx):
        file_path = os.path.join(self.root, 'case9_data', self.data_files[idx])
        with open(file_path, 'r') as f:
            data_json = json.load(f)
        
        ppc = data_json['ppc']
        res = data_json['res']
        
        # 初始化 HeteroData 物件
        data = HeteroData()
        
        # 處理 bus 節點
        buses = ppc['bus']
        num_buses = len(buses)
        
        # 節點特徵：
        bus_features = []
        for bus in buses:
            bus_features.append([
                bus[pybus.BUS_I],
                bus[pybus.BUS_TYPE],
                bus[pybus.PD],
                bus[pybus.QD],
                bus[pybus.GS],
                bus[pybus.BS],
                bus[pybus.BUS_AREA],
                bus[pybus.VM],
                bus[pybus.VA],
                bus[pybus.BASE_KV],
                bus[pybus.ZONE],
                bus[pybus.VMAX],
                bus[pybus.VMIN]
            ])
        
        data['bus'].x = torch.tensor(bus_features, dtype=torch.float)
        
        # 處理 branch 邊
        branches = ppc['branch']
        num_branches = len(branches)
        
        # 邊連接
        from_bus = []
        to_bus = []
        for branch in branches:
            from_bus.append(int(branch[pybrch.F_BUS]) - 1)
            to_bus.append(int(branch[pybrch.T_BUS]) - 1)
        
        edge_index = torch.tensor([from_bus, to_bus], dtype=torch.long)
        data['bus', 'connected_to', 'bus'].edge_index = edge_index
        
        # 邊特徵：
        edge_features = []
        for branch in branches:
            edge_features.append([
                branch[pybrch.BR_R],
                branch[pybrch.BR_X],
                branch[pybrch.BR_B],
                branch[pybrch.RATE_A],
                branch[pybrch.RATE_B],
                branch[pybrch.RATE_C],
                branch[pybrch.TAP],
                branch[pybrch.SHIFT],
                branch[pybrch.ANGMIN],
                branch[pybrch.ANGMAX]
            ])
        
        data['bus', 'connected_to', 'bus'].edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        # 目標標籤：根據 res 的結構提取
        # 假設我們要預測潮流計算的結果，例如 branch 中的 PF
        # 這裡假設每條邊的 PF 是一個標量目標
        y = torch.tensor([branch[pybrch.PF] for branch in res['branch']], dtype=torch.float).unsqueeze(1)  # 形狀 [num_branches, 1]
        
        # 將 y 與邊對齊
        # 確保 y 的順序與 edge_index 對應
        data['bus', 'connected_to', 'bus'].y = y
        
        return data
