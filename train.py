import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, to_hetero
from torch_geometric.datasets import opf
# from torch_geometric.datasets import OPFDataset
from torch_geometric.loader import DataLoader
import model as M

# Load the 14-bus OPFData FullTopology dataset training split and store it in the
# directory 'data'. Each record is a `torch_geometric.data.HeteroData`.
train_ds = opf.OPFDataset('data', case_name='pglib_opf_case14_ieee', split='train')
# train_ds = OPFDataset('data', case_name='pglib_opf_case14_ieee', split='train')

# Batch and shuffle.
training_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

# Initialise the model.
# data.metadata() here refers to the PyG graph metadata, not the OPFData metadata.
data = train_ds[0]
model = to_hetero(M.Model(), data.metadata())
with torch.no_grad(): # Initialize lazy modules.
    out = model(data.x_dict, data.edge_index_dict)
    # Train with MSE loss for one epoch.
    # In reality we would need to account for AC-OPF constraints.
    optimizer = torch.optim.Adam(model.parameters())
    model.train()
for data in training_loader:
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    loss = F.mse_loss(out['generator'], data['generator'].y)
    loss.backward()
    optimizer.step()
