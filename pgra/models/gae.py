"""
Graph Autoencoder used as the per-client reconstruction model.

The encoder is a two-layer GCN (Kipf and Welling, 2017) that maps the
hydraulic pressure field at every node down to a low-dimensional latent
representation; the decoder is a two-layer MLP that projects back to
the pressure space. The model is trained client-side with a denoising
objective on locally-corrupted samples (one sensor per training step
is scaled by a documented factor in physical units), so that the
honest cohort learns to project H-domain false-data-injection (FDI)
spoofs onto the physically valid manifold.
"""
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GraphAutoEncoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, latent_dim=32):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, edge_index):
        z = F.relu(self.conv1(x, edge_index))
        z = self.conv2(z, edge_index)
        out = F.relu(self.fc1(z))
        return self.fc2(out)

    def get_reconstruction_error(self, data):
        h_out = self.forward(data.x, data.edge_index)
        return F.mse_loss(h_out, data.x, reduction='mean')
