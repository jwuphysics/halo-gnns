import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    MessagePassing, GCNConv, PPFConv, MetaLayer, EdgeConv,
    global_mean_pool, global_max_pool, global_add_pool
)
from torch_cluster import knn_graph, radius_graph
from torch_scatter import scatter_mean, scatter_sum, scatter_max, scatter_min

model_params = dict(
    k_nn=0.14,
    n_layers=1,  # currently doesn't work for more than 1 layer
    n_hidden=128,
    n_latent=96,
    loop=False
)

feature_params = dict(
    use_stellarhalfmassradius=True,
    use_velocity=True,
    use_only_positions=False,
    use_central_galaxy_frame=False, # otherwise use center of mass frame
)

class EdgePointLayer(MessagePassing):
    """Adapted from https://github.com/PabloVD/HaloGraphNet"""
    def __init__(self, in_channels, mid_channels, out_channels, aggr='sum', use_mod=True):
        # Message passing with "max" aggregation.
        super(EdgePointLayer, self).__init__(aggr)

        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3, or 1 if only modulus is used).
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels - 2, mid_channels, bias=True),
            nn.SiLU(),
            nn.Linear(mid_channels, mid_channels, bias=True),
            nn.SiLU(),
            nn.Linear(mid_channels, out_channels, bias=True),
            nn.SiLU()
        )

        self.messages = 0.
        self.input = 0.
        self.use_mod = use_mod

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_j defines the features of neighboring nodes as shape [num_edges, in_channels]
        # pos_j defines the position of neighboring nodes as shape [num_edges, 3]
        # pos_i defines the position of central nodes as shape [num_edges, 3]

        pos_i, pos_j = x_i[:,:3], x_j[:,:3]

        input = pos_j - pos_i  # Compute spatial relation.
        input = input[:,0]**2.+input[:,1]**2 + input[:,2]**2.
        input = input.view(input.shape[0], 1)
        input = torch.cat([x_i, x_j[:, 3:], input], dim=-1)

        self.input = input
        self.messages = self.mlp(input)

        return self.messages

class EdgePointGNN(nn.Module):
    def __init__(self, node_features, n_layers, k_nn, hidden_channels=128, latent_channels=64, loop=False):
        super(EdgePointGNN, self).__init__()

        in_channels = node_features

        layers = [
            EdgePointLayer(in_channels, hidden_channels, latent_channels) for _ in range(n_layers)
        ]

        self.layers = nn.ModuleList(layers)
        self.lin = nn.Sequential(
            nn.Linear(latent_channels * 3 + 2, latent_channels, bias=True),
            nn.SiLU(),
            nn.Linear(latent_channels, latent_channels, bias=True),
            nn.SiLU(),
            nn.Linear(latent_channels, 2, bias=True)
        )
        self.k_nn = k_nn
        self.loop = loop
        self.pooled = 0.
        self.h = 0.
    
    def forward(self, data):
        x, pos, batch, u = data.x, data.pos, data.batch, data.u

        # determine edges by getting neighbors within radius defined by `k_nn`
        edge_index = radius_graph(pos, r=self.k_nn, batch=batch, loop=self.loop)

        for layer in self.layers:
            x = layer(x, edge_index=edge_index)
        
        self.h = x
            
        # use all the pooling! (and also the extra global features `u`)
        addpool = global_add_pool(x, batch) # [num_examples, hidden_channels]
        meanpool = global_mean_pool(x, batch)
        maxpool = global_max_pool(x, batch)
        self.pooled = torch.cat([addpool, meanpool, maxpool, u], dim=1)

        # final linear layer
        return self.lin(self.pooled)
        