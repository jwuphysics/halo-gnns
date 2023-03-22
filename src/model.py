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
    k_nn=100.,
    n_layers=1,
    n_hidden=128,
    n_latent=96,
    n_out=2,
    loop=False,
    estimate_all_subhalos=False,
    use_global_pooling=True,
)

class EdgePointLayer(MessagePassing):
    """Adapted from https://github.com/PabloVD/HaloGraphNet.
    Initialized with `sum` aggregation, although `max` or others are possible.
    """
    def __init__(self, in_channels, mid_channels, out_channels, aggr='sum', use_mod=False, use_bias=True):
        super(EdgePointLayer, self).__init__(aggr)

        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3, or 1 if only modulus is used).
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels - 2, mid_channels, bias=use_bias),
            nn.LayerNorm(mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, mid_channels, bias=use_bias),
            nn.LayerNorm(mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, out_channels, bias=use_bias),
        )

        self.messages = 0.
        self.input = 0.
        self.use_mod = use_mod

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        """Message passing function:
            x_j defines the features of neighboring nodes as shape [num_edges, in_channels]
            pos_j defines the position of neighboring nodes as shape [num_edges, 3]
            pos_i defines the position of central nodes as shape [num_edges, 3]
        """

        pos_i, pos_j = x_i[:,:3], x_j[:,:3]

        input = pos_j - pos_i  # Compute spatial relation.
        input = input[:,0]**2.+input[:,1]**2 + input[:,2]**2.
        input = input.view(input.shape[0], 1)
        input = torch.cat([x_i, x_j[:, 3:], input], dim=-1)

        self.input = input
        self.messages = self.mlp(input)

        return self.messages

class EdgePointGNN(nn.Module):
    def __init__(self, node_features, n_layers, k_nn, hidden_channels=128, latent_channels=64, n_out=2, loop=False, estimate_all_subhalos=False, use_global_pooling=True, use_bias=True):
        super(EdgePointGNN, self).__init__()

        in_channels = node_features
        
        layers = [EdgePointLayer(in_channels, hidden_channels, latent_channels, use_bias=use_bias)]
        for _ in range(n_layers-1):
            layers += [EdgePointLayer(latent_channels, hidden_channels, latent_channels)]
        self.n_out = n_out
        self.layers = nn.ModuleList(layers)
        self.estimate_all_subhalos = estimate_all_subhalos
        self.use_global_pooling = use_global_pooling
        self.fc = nn.Sequential(
            (
                nn.Linear(latent_channels, latent_channels, bias=use_bias) if self.estimate_all_subhalos
                else nn.Linear(latent_channels * 3 + 2, latent_channels, bias=use_bias)
            ),
            nn.LayerNorm(latent_channels),
            nn.ReLU(),
            nn.Linear(latent_channels, latent_channels, bias=use_bias),
            nn.LayerNorm(latent_channels),
            nn.ReLU(),
            nn.Linear(latent_channels, 2 * n_out, bias=use_bias)
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
        x = x.relu()
        
        if not self.estimate_all_subhalos:
            # use all the pooling! (and also the extra global features `u`)
            addpool = global_add_pool(x, batch)
            meanpool = global_mean_pool(x, batch)
            maxpool = global_max_pool(x, batch)
            self.pooled = torch.cat([addpool, meanpool, maxpool, u], dim=1)

            # final fully connected layer
            return self.fc(self.pooled)
        else:
            if self.estimate_all_subhalos:
                # returns all of the subhalos
                return self.fc(x)
            else:
                # retuns just the central subhalo
                _, counts = torch.unique(batch, return_counts=True)
                idx = torch.cumsum(counts, dim=0) - counts[0]
                return self.fc(x[idx, :])
