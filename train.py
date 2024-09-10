import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, to_hetero
from torch_geometric.datasets import opf
from torch_geometric.loader import DataLoader
import model as M
from torch.utils.tensorboard import SummaryWriter
import datetime
from tqdm import tqdm
import os
import argparse
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 參數解析
parser = argparse.ArgumentParser(description='OPF Training Script')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
args = parser.parse_args()

# 設置隨機種子
torch.manual_seed(42)

# 加載數據集
train_ds = opf.OPFDataset('data', case_name='pglib_opf_case14_ieee', split='train')
val_ds = opf.OPFDataset('data', case_name='pglib_opf_case14_ieee', split='val')

# 創建數據加載器
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size)

# 初始化模型
data = train_ds[0]
model = to_hetero(M.Model(), data.metadata())

# 檢查是否有可用的GPU (apple silicon使用MPS)
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
model = model.to(device)

# 初始化優化器和學習率調度器
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

# 設置TensorBoard
log_dir = "runs/opf_experiment_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir)

# 驗證函數
def validate(model, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)
            loss = F.mse_loss(out['generator'], batch['generator'].y)
            total_loss += loss.item()
    return total_loss / len(val_loader)

# 訓練循環
best_val_loss = float('inf')
for epoch in tqdm(range(args.epochs), desc="Epochs"):
    model.train()
    epoch_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x_dict, batch.edge_index_dict)
        loss = F.mse_loss(out['generator'], batch['generator'].y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    avg_train_loss = epoch_loss / len(train_loader)
    val_loss = validate(model, val_loader)
    
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    writer.add_scalar('Loss/validation', val_loss, epoch)
    
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # 保存最佳模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
        }, os.path.join(log_dir, 'best_model.pth'))

# 保存最終模型和配置
final_model_path = os.path.join(log_dir, 'final_model.pth')
torch.save({
    'epoch': args.epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': avg_train_loss,
    'val_loss': val_loss,
}, final_model_path)

# 儲存超參數設定
config = {
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'learning_rate': args.lr,
    'model_architecture': str(model),
    'optimizer': str(optimizer),
    'scheduler': str(scheduler),
}

with open(os.path.join(log_dir, 'config.json'), 'w') as f:
    json.dump(config, f)

print("訓練結束.")
print(f"模型儲存位置 {final_model_path}")
print(f"TensorBoard logs 位置 {log_dir}")
