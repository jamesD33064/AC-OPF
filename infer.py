import torch
import torch.nn.functional as F
from torch_geometric.nn import to_hetero
from torch_geometric.loader import DataLoader
import model as M
import json
import os
from torch_geometric.datasets import opf
import numpy as np

# 設置設備
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# 加載 config.json 文件
config_path = 'runs/opf_experiment_20240910-113802/config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

# 載入模型架構
train_ds = opf.OPFDataset('data', case_name='pglib_opf_case14_ieee', split='train')
data = train_ds[0] 
model = to_hetero(M.Model(), data.metadata())
model = model.to(device)

# 載入已訓練的模型權重
model_path = os.path.join(os.path.dirname(config_path), 'best_model.pth')
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# 模型設置為推斷模式
model.eval()

inference_dataset = opf.OPFDataset('data', case_name='pglib_opf_case14_ieee', split='test')
inference_loader = DataLoader(inference_dataset, batch_size=config['batch_size'], shuffle=False)

# 開始推斷
predictions = []
with torch.no_grad():
    for batch in inference_loader:
        batch = batch.to(device)
        out = model(batch.x_dict, batch.edge_index_dict)
        preds = out['generator'].cpu().numpy()
        predictions.append(preds)

predictions = np.concatenate(predictions, axis=0)  # 拼接所有批次的結果
output_path = os.path.join(os.path.dirname(config_path), 'predictions.npy')
np.save(output_path, predictions)

print(f"推斷完成，結果已儲存到 {output_path}")
