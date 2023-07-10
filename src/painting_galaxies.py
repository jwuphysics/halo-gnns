import argparse
import cmasher as cmr
import gc
import matplotlib.pyplot as plt
import os
import pickle
import random
import scipy.spatial
from scipy.stats import median_abs_deviation
from sklearn.ensemble import RandomForestRegressor
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    MessagePassing, GCNConv, PPFConv, MetaLayer, EdgeConv,
    global_mean_pool, global_max_pool, global_add_pool
)
from torch_cluster import radius_graph

from data import *
from train import *

parser = argparse.ArgumentParser(description='Supply aggregation function and whether loops are used.')
parser.add_argument('--aggr', help='Aggregation function: "sum", "max", or "multi"', required=True, type=str)
parser.add_argument('--loops', help='Whether to use self-loops: "True" or "False"', required=True, type=int)

args = parser.parse_args()

ROOT = Path(__file__).parent.parent.resolve()
tng_base_path = f"{ROOT}/illustris_data/TNG300-1/output"
results_path = f"{ROOT}/results/painting-galaxies"

seed = 255
rng = np.random.RandomState(seed)
random.seed(seed)
torch.manual_seed(seed)

c0, c1, c2, c3, c4 = '#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600'

device = "cuda" if torch.cuda.is_available() else "cpu"

### params for generating data products, visualizations, etc.
recompile_data = False
retrain = False
revalidate = False
make_plots = True
save_models = True

### simulation and selection criteria params
use_gal = False # gal -> dark matter, or vice versa


cuts = {
    "minimum_log_stellar_mass": 9,
    "minimum_log_halo_mass": 10,
    "minimum_n_star_particles": 50
}

snapshot = 99 # z=0
h = 0.6774    # Planck 2015 cosmology

### training and optimization params
batch_size = 9

training_params = dict(
    batch_size=batch_size,
    learning_rate=1e-2,
    weight_decay=1e-4,
    n_epochs=1000,
)

split = 6 # N_subboxes = split**3
train_test_frac_split = split**2


### GNN params
undirected = True
periodic = False

def make_webs(
    tng_base_path="../illustris_data/TNG300-1/output", 
    data_path=None,
    snapshot=99, 
    r_link=5,
    pad=2.5,
    split=6,
    cuts=cuts, 
    use_gal=False, 
    h=0.6774, 
    undirected=True, 
    periodic=False, 
    use_loops=True,
    in_projection=False,
    normalization_params=normalization_params
):
    
    if use_gal:
        # use_cols = ['subhalo_x', 'subhalo_y', 'subhalo_z', 'subhalo_vx', 'subhalo_vy', 'subhalo_vz','subhalo_logstellarmass', 'subhalo_stellarhalfmassradius']
        use_cols = ['subhalo_x', 'subhalo_y', 'subhalo_z', 'subhalo_vx', 'subhalo_vy', 'subhalo_vz','subhalo_logstellarmass']
        y_cols = ['subhalo_loghalomass', 'subhalo_logvmax'] 
    else:
        use_cols = ['subhalo_x', 'subhalo_y', 'subhalo_z', 'subhalo_vx', 'subhalo_vy', 'subhalo_vz', 'subhalo_loghalomass', 'subhalo_logvmax'] 
        # y_cols = ['subhalo_logstellarmass', 'subhalo_stellarhalfmassradius']
        y_cols = ['subhalo_logstellarmass']

    if in_projection:
        for c in ['subhalo_z', 'subhalo_vx', 'subhalo_vy']:
            use_cols.remove(c)

    subhalo_fields = [
        "SubhaloPos", "SubhaloMassType", "SubhaloLenType", "SubhaloHalfmassRadType", 
        "SubhaloVel", "SubhaloVmax", "SubhaloGrNr", "SubhaloFlag"
    ]
    subhalos = il.groupcat.loadSubhalos(tng_base_path, snapshot, fields=subhalo_fields) 

    pos = subhalos["SubhaloPos"][:,:3]
    min_box, max_box = np.rint(np.min(pos)), np.rint(np.max(pos))
    box_size = max_box/(h*1e3) # in Mpc

    halo_fields = ["Group_M_Crit200", "GroupFirstSub", "GroupPos", "GroupVel"]
    halos = il.groupcat.loadHalos(tng_base_path, snapshot, fields=halo_fields)

    subhalo_pos = subhalos["SubhaloPos"][:] / (h*1e3)
    subhalo_stellarmass = subhalos["SubhaloMassType"][:,4]
    subhalo_halomass = subhalos["SubhaloMassType"][:,1]
    subhalo_n_stellar_particles = subhalos["SubhaloLenType"][:,4]
    subhalo_stellarhalfmassradius = subhalos["SubhaloHalfmassRadType"][:,4]  / normalization_params["norm_half_mass_radius"]
    subhalo_vel = subhalos["SubhaloVel"][:] /  normalization_params["norm_velocity"]
    subhalo_vmax = subhalos["SubhaloVmax"][:] / normalization_params["norm_velocity"]
    subhalo_flag = subhalos["SubhaloFlag"][:]
    halo_id = subhalos["SubhaloGrNr"][:].astype(int)

    halo_mass = halos["Group_M_Crit200"][:]
    halo_primarysubhalo = halos["GroupFirstSub"][:].astype(int)
    group_pos = halos["GroupPos"][:] / (h*1e3)
    group_vel = halos["GroupVel"][:]  / normalization_params["norm_velocity"]

    halos = pd.DataFrame(
        np.column_stack((np.arange(len(halo_mass)), group_pos, group_vel, halo_mass, halo_primarysubhalo)),
        columns=['halo_id', 'halo_x', 'halo_y', 'halo_z', 'halo_vx', 'halo_vy', 'halo_vz', 'halo_mass', 'halo_primarysubhalo']
    )
    halos['halo_id'] = halos['halo_id'].astype(int)
    halos.set_index("halo_id", inplace=True)

    # get subhalos/galaxies      
    subhalos = pd.DataFrame(
        np.column_stack([halo_id, subhalo_flag, np.arange(len(subhalo_stellarmass)), subhalo_pos, subhalo_vel, subhalo_n_stellar_particles, subhalo_stellarmass, subhalo_halomass, subhalo_stellarhalfmassradius, subhalo_vmax]), 
        columns=['halo_id', 'subhalo_flag', 'subhalo_id', 'subhalo_x', 'subhalo_y', 'subhalo_z', 'subhalo_vx', 'subhalo_vy', 'subhalo_vz', 'subhalo_n_stellar_particles', 'subhalo_stellarmass', 'subhalo_halomass', 'subhalo_stellarhalfmassradius', 'subhalo_vmax'],
    )
    subhalos["is_central"] = (halos.loc[subhalos.halo_id]["halo_primarysubhalo"].values == subhalos["subhalo_id"].values)

    subhalos = subhalos[subhalos["subhalo_flag"] != 0].copy()
    subhalos['halo_id'] = subhalos['halo_id'].astype(int)
    subhalos['subhalo_id'] = subhalos['subhalo_id'].astype(int)

    subhalos.drop("subhalo_flag", axis=1, inplace=True)

    # impose stellar mass and particle cuts
    subhalos = subhalos[subhalos["subhalo_n_stellar_particles"] > cuts["minimum_n_star_particles"]].copy()
    subhalos["subhalo_logstellarmass"] = np.log10(subhalos["subhalo_stellarmass"] / h)+10

    subhalos["subhalo_loghalomass"] = np.log10(subhalos["subhalo_halomass"] / h)+10
    subhalos["subhalo_logvmax"] = np.log10(subhalos["subhalo_vmax"])
    subhalos["subhalo_logstellarhalfmassradius"] = np.log10(subhalos["subhalo_stellarhalfmassradius"])

    subhalos = subhalos[subhalos["subhalo_loghalomass"] > cuts["minimum_log_halo_mass"]].copy()

    subhalos = subhalos[subhalos["subhalo_logstellarmass"] > cuts["minimum_log_stellar_mass"]].copy()

    data = []
    for n in tqdm(range(split), position=0):
        for g in tqdm(range(split), position=1, leave=False):
            for k in tqdm(range(split), position=2, leave=False):
                # print(n,g,k)
                xlims = np.array([box_size/split*n+pad, box_size/split*(n+1)-pad])
                ylims = np.array([box_size/split*g+pad, box_size/split*(g+1)-pad])
                zlims = np.array([box_size/split*k+pad, box_size/split*(k+1)-pad])

                pos = np.vstack(subhalos[['subhalo_x', 'subhalo_y', 'subhalo_z']].to_numpy())

                xmask = np.logical_and(pos[:,0]>xlims[0],pos[:,0]<xlims[1])
                ymask = np.logical_and(pos[:,1]>ylims[0],pos[:,1]<ylims[1])
                zmask = np.logical_and(pos[:,2]>zlims[0],pos[:,2]<zlims[1])
                mask = np.logical_and(zmask, np.logical_and(xmask, ymask))

                df = subhalos.iloc[mask].copy()
                df.reset_index(drop=True)

                # remove extraneous columns
                df.drop(["subhalo_n_stellar_particles", "subhalo_stellarmass", "subhalo_halomass"], axis=1, inplace=True)

                # set new zero point

                df[['subhalo_x', 'subhalo_y', 'subhalo_z']] = df[['subhalo_x', 'subhalo_y', 'subhalo_z']] - np.array([box_size/split*n+pad, box_size/split*g+pad, box_size/split*k+pad])

                #make positions for clustering

                if in_projection:
                    pos = np.vstack(df[['subhalo_x', 'subhalo_y']].to_numpy())    
                else:
                    pos = np.vstack(df[['subhalo_x', 'subhalo_y', 'subhalo_z']].to_numpy())

                kd_tree = scipy.spatial.KDTree(pos, leafsize=25, boxsize=box_size)
                edge_index = kd_tree.query_pairs(r=r_link, output_type="ndarray")

                # normalize positions

                df[['subhalo_x', 'subhalo_y', 'subhalo_z']] = df[['subhalo_x', 'subhalo_y', 'subhalo_z']]/(box_size/2)

                if undirected:
                # Add reverse pairs
                    reversepairs = np.zeros((edge_index.shape[0],2))
                    for i, pair in enumerate(edge_index):
                        reversepairs[i] = np.array([pair[1], pair[0]])
                    edge_index = np.append(edge_index, reversepairs, 0)

                    edge_index = edge_index.astype(int)

                    # Write in pytorch-geometric format
                    edge_index = edge_index.reshape((2,-1))
                    num_pairs = edge_index.shape[1]

                row, col = edge_index

                diff = pos[row]-pos[col]
                dist = np.linalg.norm(diff, axis=1)

                use_gal = True

                if periodic:
                    # Take into account periodic boundary conditions, correcting the distances
                    for i, pos_i in enumerate(diff):
                        for j, coord in enumerate(pos_i):
                            if coord > r_link:
                                diff[i,j] -= box_size  # Boxsize normalize to 1
                            elif -coord > r_link:
                                diff[i,j] += box_size  # Boxsize normalize to 1

                centroid = np.mean(pos,axis=0) # define arbitrary coordinate, invarinat to translation/rotation shifts, but not stretches
                # centroid+=1.2

                unitrow = (pos[row]-centroid)/np.linalg.norm((pos[row]-centroid), axis=1).reshape(-1,1)
                unitcol = (pos[col]-centroid)/np.linalg.norm((pos[col]-centroid), axis=1).reshape(-1,1)
                unitdiff = diff/dist.reshape(-1,1)
                # Dot products between unit vectors
                cos1 = np.array([np.dot(unitrow[i,:].T,unitcol[i,:]) for i in range(num_pairs)])
                cos2 = np.array([np.dot(unitrow[i,:].T,unitdiff[i,:]) for i in range(num_pairs)])

                edge_attr = np.concatenate([dist.reshape(-1,1), cos1.reshape(-1,1), cos2.reshape(-1,1)], axis=1)

                if use_loops:
                    loops = np.zeros((2,pos.shape[0]),dtype=int)
                    atrloops = np.zeros((pos.shape[0], edge_attr.shape[1]))
                    for i, posit in enumerate(pos):
                        loops[0,i], loops[1,i] = i, i
                        atrloops[i,0], atrloops[i,1], atrloops[i,2] = 0., 1., 0.
                    edge_index = np.append(edge_index, loops, 1)
                    edge_attr = np.append(edge_attr, atrloops, 0)
                edge_index = edge_index.astype(int)

                x = torch.tensor(np.vstack(df[use_cols].to_numpy()), dtype=torch.float)
                y = torch.tensor(np.vstack(df[y_cols].to_numpy()), dtype=torch.float)
                edge_index = torch.tensor(edge_index, dtype=torch.long)
                edge_attr=torch.tensor(edge_attr, dtype=torch.float)
                pos = torch.tensor(pos, dtype=torch.float)
                is_central = torch.tensor(df.is_central.values, dtype=bool)
                overdensity = torch.zeros(len(x), dtype=x.dtype)
                for i in range(len(x)):
                    neighbors = edge_index[1, edge_index[0] == i] # get neighbor indices
                    overdensity[i] = torch.log10((10**x[neighbors, -2]).sum()) # get sum of masses of neighbors (2nd to last index in `x`)

                data.append(Data(x=x, y=y, pos=pos, is_central=is_central, edge_index=edge_index, edge_attr=edge_attr, overdensity=overdensity))

                proj_str = "-projected" if in_projection else ""

                if data_path is None:
                    data_path = os.path.join(tng_base_path, 'cosmic_graphs', f'split_{split**3}_link_{int(r_link)}_pad{int(pad)}_gal{int(use_gal)}{proj_str}.pkl')

                if not os.path.isdir(os.path.join(tng_base_path, 'cosmic_graphs')):
                    os.mkdir(os.path.join(tng_base_path, 'cosmic_graphs'))

                with open(data_path, 'wb') as handle:
                    pickle.dump(data, handle)

                    
def visualize_graph(data, draw_edges=True, projection="3d", edge_index=None, boxsize=302.6, fontsize=12, results_path=None):

    fig = plt.figure(figsize=(6, 6), dpi=300)

    if projection=="3d":
        ax = fig.add_subplot(projection="3d")
        pos = boxsize/2*data.x[:,:3]
        mass = data.x[:,-2]
    elif projection=="2d":
        ax = fig.add_subplot()
        pos = boxsize/2*data.x[:,:2]
        mass = data.x[:,-2]

    # Draw lines for each edge
    if data.edge_index is not None and draw_edges:
        for (src, dst) in data.edge_index.t().tolist():

            src = pos[src].tolist()
            dst = pos[dst].tolist()
            if projection=="3d":
                ax.plot([src[0], dst[0]], [src[1], dst[1]], zs=[src[2], dst[2]], linewidth=0.2/r_link, color='black')
            elif projection=="2d":
                ax.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=0.2/r_link, color='black')

    # Plot nodes
    if projection=="3d":
        sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=4**(mass - 10), zorder=1000, vmin=11, vmax=13.5, alpha=0.9, edgecolor='k', c=mass, cmap="plasma", linewidths=0.1)
    elif projection=="2d":
        sc = ax.scatter(pos[:, 0], pos[:, 1], s=4**(mass - 10), zorder=1000, alpha=0.9, edgecolor='k', c=mass, vmin=11, vmax=13.5,  cmap="plasma", linewidths=0.1)
    plt.subplots_adjust(right=0.8)

    if projection == "3d":
        cb = fig.colorbar(sc, shrink=0.8, aspect=50, location='top', pad=-0.03)
    else:
        cb = fig.colorbar(sc, shrink=0.805, aspect=40, location="top", pad=0.03)
    cb.set_label("log($M_{\\rm halo}/M_{\\odot})$", fontsize=fontsize)

    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    ax.set_xlabel("X [Mpc]", fontsize=fontsize)
    ax.set_ylabel("Y [Mpc]", fontsize=fontsize)

    if projection=="3d": 
        ax.zaxis.set_tick_params(labelsize=fontsize)
        ax.set_zlabel("Z [Mpc]", fontsize=fontsize)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.2)
        ax.yaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.2)
        ax.zaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.2)
    else:
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)

    fig.tight_layout()
    if projection == "3d":
        fig.savefig(f"{results_path}/cosmic-graph.png", dpi=300)
    else:
        fig.savefig(f"{results_path}/cosmic-graph-projection.png", dpi=300)
    
    plt.close()

        
class EdgePointLayer(MessagePassing):
    """Graphnet with point + edge layers.
    
    Very loosely inspired by https://github.com/PabloVD/HaloGraphNet.
    Initialized with `sum` aggregation, although `max` or others are possible.
    """
    def __init__(self, in_channels, mid_channels, out_channels, aggr='sum', use_mod=False):
        super(EdgePointLayer, self).__init__(aggr)

        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3, or 1 if only modulus is used).
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels - 2, mid_channels, bias=True),
            nn.LayerNorm(mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, mid_channels, bias=True),
            nn.LayerNorm(mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, out_channels, bias=True),
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
    """Graph net over nodes and edges with multiple unshared layers, and sequential layers with residual connections.
    Self-loops also get their own MLP (i.e. galaxy-halo connection).
    """
    def __init__(self, node_features, n_layers, D_link, hidden_channels=64, aggr="sum", latent_channels=64, n_out=2, n_unshared_layers=4, loop=True, estimate_all_subhalos=True, use_global_pooling=True):
        super(EdgePointGNN, self).__init__()

        in_channels = node_features
        
        layers = [
            nn.ModuleList([
                EdgePointLayer(in_channels, hidden_channels, latent_channels, aggr=aggr)
                for _ in range(n_unshared_layers)
            ])
        ]
        for _ in range(n_layers-1):
            layers += [
                nn.ModuleList([
                    EdgePointLayer(3 * latent_channels * n_unshared_layers, hidden_channels, latent_channels, aggr=aggr) 
                    for _ in range(n_unshared_layers)
                ])
            ]
        self.n_out = n_out
        self.layers = nn.ModuleList(layers)
        self.estimate_all_subhalos = estimate_all_subhalos
        self.use_global_pooling = use_global_pooling

        n_pool = (len(aggr) if isinstance(aggr, list) else 1) 
        self.fc = nn.Sequential(
            (
                nn.Linear((n_unshared_layers * n_pool )* latent_channels, latent_channels, bias=True) if self.estimate_all_subhalos
                else nn.Linear(n_unshared_layers * latent_channels * 3, latent_channels, bias=True)
            ),
            nn.LayerNorm(latent_channels),
            nn.ReLU(),
            nn.Linear(latent_channels, latent_channels, bias=True),
            nn.LayerNorm(latent_channels),
            nn.ReLU(),
            nn.Linear(latent_channels, 2 * n_out, bias=True)
        )
        
        self.galaxy_halo_mlp = nn.Sequential(
            nn.Linear(node_features, latent_channels, bias=True),
            nn.LayerNorm(latent_channels),
            nn.ReLU(),
            nn.Linear(latent_channels, latent_channels, bias=True),
            nn.LayerNorm(latent_channels),
            nn.ReLU(),
            nn.Linear(latent_channels, 2 * n_out, bias=True)
        )
        
        self.D_link = D_link
        self.loop = loop
        self.pooled = 0.
        self.h = 0.
    
    def forward(self, data):
        
        # determine edges by getting neighbors within radius defined by `D_link`
        edge_index = radius_graph(data.pos, r=self.D_link, batch=data.batch, loop=self.loop)

        x = torch.cat([unshared_layer(data.x, edge_index=edge_index) for unshared_layer in self.layers[0]], axis=1)
        self.h = x
        x = x.relu()
        
        for layer in self.layers[1:]:
            # use residual
            x = self.h + torch.cat([unshared_layer(x, edge_index=edge_index) for unshared_layer in layer], axis=1)
        
            self.h = x
            x = x.relu()
        
        # x = torch.concat([x, self.galaxy_halo_mlp(data.x)], axis=1)
        
        if self.estimate_all_subhalos:
            # returns all of the subhalos
            return self.fc(x) + self.galaxy_halo_mlp(data.x)
        else:
            import sys
            sys.exit()

                    
        
def train_cosmic_gnn(data, k, split=6, r_link=5, aggr="sum", use_loops=True, in_projection=False, make_plots=True, results_path=None, hidden_channels=256, latent_channels=128, n_layers=1, n_unshared_layers=1):
    """Trains GNN using global optimization params"""    
    proj_str = "-projected" if in_projection else ""
    
    print(f"Begin training{proj_str}")
    
    
    gc.collect();
    print(f"Training fold {k+1}/{split}" + "\n")
    
    node_features = data[0].x.shape[1]
    out_features = data[0].y.shape[1]

    model = EdgePointGNN(
        node_features=node_features, 
        n_layers=n_layers, 
        D_link=r_link,
        hidden_channels=hidden_channels,
        latent_channels=latent_channels,
        loop=use_loops,
        n_unshared_layers=n_unshared_layers,
        estimate_all_subhalos=True,
        use_global_pooling=False,
        n_out=out_features,
        aggr=(["sum", "max", "mean"] if aggr == "multi" else aggr)
    )

    model.to(device);

    # assumes that data is a list of PyG Data objects, otherwise this will fail
    data_train = data[:k*train_test_frac_split] + data[(k+1)*train_test_frac_split:]
    data_valid = data[k*train_test_frac_split:(k+1)*train_test_frac_split]

    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(data_valid, batch_size=batch_size, shuffle=False)

    print("Epoch    train loss   valid loss   RMSE   avg std")

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=training_params["learning_rate"], 
        weight_decay=training_params["weight_decay"]
    )

    train_losses = []
    valid_losses = []
    for epoch in range(training_params["n_epochs"]):

        # anneal at 25%, 50%, and 75%
        if (epoch == int(training_params["n_epochs"] * 0.5)):
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=training_params["learning_rate"] / 5, 
                weight_decay=training_params["weight_decay"] / 5
            )
        if (epoch == int(training_params["n_epochs"] * 0.75)):
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=training_params["learning_rate"] / 25, 
                weight_decay=training_params["weight_decay"] / 25
            )
        # if (epoch == int(training_params["n_epochs"] * 0.75)):
        #     optimizer = torch.optim.AdamW(
        #         model.parameters(), 
        #         lr=training_params["learning_rate"] / 125, 
        #         weight_decay=training_params["weight_decay"] / 125
        #     )

        train_loss = train(train_loader, model, optimizer, device, in_projection=in_projection)
        valid_loss, valid_std, p, y, logvar_p  = validate(valid_loader, model, device, in_projection=in_projection)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)


        if (epoch + 1) % 10 == 0:
            print(f" {epoch + 1: >4d}      {train_loss: >6.2f}       {valid_loss: >6.2f}    {np.sqrt(np.mean((p - y.flatten())**2)): >6.3f}  {np.mean(valid_std): >6.3f}")

    if make_plots:
        plt.figure(figsize=(6, 3), dpi=150)
        plt.plot(train_losses, c=c0, label="Train")
        plt.plot(valid_losses, c=c3, label="Valid")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(alpha=0.15)
        plt.ylim(plt.ylim()[0], min(plt.ylim()[1], -2))
        plt.tight_layout()

        plt.savefig(f"{results_path}/training-logs/losses{proj_str}-fold{k+1}.png")

    if save_models:
        torch.save(
            model.state_dict(),
            f"{results_path}/models/EdgePointGNN-link{r_link}-hidden{hidden_channels}-latent{latent_channels}-selfloops{int(use_loops)}-agg{aggr}-epochs{training_params['n_epochs']}{proj_str}_fold{k+1}.pth", 
        )
        plt.close()
    return model

def validate_cosmic_gnn(model, data, k, split=6, in_projection=False, make_plots=True, results_path=None):
    """Validates and compares GNN model against RF models"""
    data_train = data[:k*train_test_frac_split] + data[(k+1)*train_test_frac_split:]
    data_valid = data[k*train_test_frac_split:(k+1)*train_test_frac_split]

    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(data_valid, batch_size=batch_size, shuffle=False)
    
    # actually validate model
    _, _, p_valid, y_valid, logvar_p = validate(valid_loader, model, device)
    p_valid = p_valid.reshape((-1, 1))
    y_valid = y_valid[:, 0]
    

    X_train = np.concatenate([d.x[:, -2:] for d in data_train]).reshape((-1, 2))
    y_train = np.concatenate([d.y[:, 0] for d in data_train])
    
    X_valid = np.concatenate([d.x[:, -2:] for d in data_valid])

    # compare against random forest models
    print("Comparing against random forest models")

    # case one: Vmax
    X_train_Vmax = np.concatenate([d.x[:, -1] for d in data_train]).reshape((-1, 1))
    y_train = np.concatenate([d.y[:, 0] for d in data_train])

    rf_Vmax = RandomForestRegressor()
    rf_Vmax.fit(X_train_Vmax, y_train)
    X_valid_Vmax = np.concatenate([d.x[:, -1] for d in data_valid]).reshape((-1, 1))
    p_log_Mstar_rf_Vmax = rf_Vmax.predict(X_valid_Vmax)

    # case two: Mhalo
    X_train_Mh = np.concatenate([d.x[:, -2] for d in data_train]).reshape((-1, 1))
    y_train = np.concatenate([d.y[:, 0] for d in data_train])

    rf_Mh = RandomForestRegressor()
    rf_Mh.fit(X_train_Mh, y_train)
    X_valid_Mh = np.concatenate([d.x[:, -2] for d in data_valid]).reshape((-1, 1))
    p_log_Mstar_rf_Mh = rf_Mh.predict(X_valid_Mh)

    # case three: Vmax+Mhalo
    X_train_MhVmax = np.concatenate([d.x[:, -2:] for d in data_train]).reshape((-1, 2))
    y_train = np.concatenate([d.y[:, 0] for d in data_train])

    rf_MhVmax = RandomForestRegressor()
    rf_MhVmax.fit(X_train_MhVmax, y_train)
    X_valid_MhVmax = np.concatenate([d.x[:, -2:] for d in data_valid]).reshape((-1, 2))
    p_log_Mstar_rf_MhVmax = rf_MhVmax.predict(X_valid_MhVmax)

    # case four: Vmax+Mhalo+delta_X
    X_train_overdensity = np.concatenate([torch.hstack([d.x[:, -2:], d.overdensity.view(-1, 1)]) for d in data_train], 0)
    rf_overdensity = RandomForestRegressor()
    rf_overdensity.fit(X_train_overdensity, y_train)
    X_valid_overdensity = np.concatenate([torch.hstack([d.x[:, -2:], d.overdensity.view(-1, 1)]) for d in data_valid], 0)
    p_log_Mhalo_rf_overdensity = rf_overdensity.predict(X_valid_overdensity)

    p_gnn_key = "p_GNN_2d" if in_projection else "p_GNN_3d" 
    overdensity_key = f"delta_{r_link}"
    df = pd.DataFrame({
        "log_Mhalo": X_valid_overdensity[:, 0].flatten(),
        "log_Vmax": X_valid_overdensity[:, 1].flatten(),
        overdensity_key: X_valid_overdensity[:, 2].flatten(),
        "log_Mstar": y_valid.flatten(),
        "p_RF_Mhalo": p_log_Mstar_rf_Mh.flatten(),
        "p_RF_Vmax": p_log_Mstar_rf_Vmax.flatten(),
        "p_RF_MhVmax": p_log_Mstar_rf_MhVmax.flatten(),
        "p_RF_overdensity": p_log_Mhalo_rf_overdensity.flatten(),
        p_gnn_key: p_valid.flatten(),
    })

    proj_str = "-projected" if in_projection else ""
    df.to_csv(f"{results_path}/validation{proj_str}-fold{k+1}.csv", index=False)
     

def combine_results(split=6, centrals=None, results_path=None):
    """Combine all results, including 3d and 2d GNN"""
    results = []
    for k in range(split):
        valid_k = pd.read_csv(f"{results_path}/validation-fold{k+1}.csv")
        valid_proj_k = pd.read_csv(f"{results_path}/validation-projected-fold{k+1}.csv", usecols=["p_GNN_2d"])
        
        valid_k["p_GNN_2d"] = valid_proj_k
        results.append(valid_k)
    
    results = pd.concat(results, axis=0, ignore_index=True)
    
    if centrals is not None:
        results["is_central"] = centrals
    results.to_csv(f"{results_path}/cross-validation.csv", index=False)
    
    return results


def get_metrics(p, y):
    """Returns a bunch of metrics for any model (RF, GNN) prediction"""
    rmse = np.sqrt(np.mean((p-y)**2))
    nmad = median_abs_deviation((p-y), scale="normal")
    mae = np.mean(np.absolute(p-y))
    pearson_rho = np.corrcoef(p, y)[0,1]
    r2 = 1 - (np.sum((p-y)**2) / np.sum((y - y.mean())**2))
    bias = np.mean(p - y)*1e3
    f_outlier = np.mean(np.absolute(p-y) > 3*nmad) * 100

    return rmse, nmad, mae, pearson_rho, r2, bias, f_outlier


def save_metrics(df, results_path=None):
    """Save LaTeX table of results"""
    
    with open(f"{results_path}/metrics.tex", "w") as f:
        
        metrics_Mh = get_metrics(df.p_RF_Mhalo, df.log_Mstar)
        f.write("RF - $M_{\\rm halo}$ & 1 & " + " & ".join([f"{m:.3f}" for m in metrics_Mh]) + "\\\\" + "\n")

        metrics_Vmax = get_metrics(df.p_RF_Vmax, df.log_Mstar)
        f.write("RF - $V_{\\rm max}$ & 1 & " + " & ".join([f"{m:.3f}" for m in metrics_Vmax]) + "\\\\" + "\n")

        metrics_MhVmax = get_metrics(df.p_RF_MhVmax, df.log_Mstar)
        f.write("RF - $M_{\\rm halo}+V_{\\rm max}$ & 2 & " + " & ".join([f"{m:.3f}" for m in metrics_MhVmax]) + "\\\\" + "\n")

        metrics_overdensity = get_metrics(df.p_RF_overdensity, df.log_Mstar)
        f.write("RF - $M_{\\rm halo}+V_{\\rm max}+" + f"\\delta_{r_link}$ & 2 & " + " & ".join([f"{m:.3f}" for m in metrics_overdensity]) + "\\\\" + "\n")

        metrics_GNN_proj = get_metrics(df.p_GNN_2d, df.log_Mstar)
        f.write("GNN ($2d$ projection) & 5 & " + " & ".join([f"{m:.3f}" for m in metrics_GNN_proj]) + "\\\\" + "\n")

        metrics_GNN = get_metrics(df.p_GNN_3d, df.log_Mstar)
        f.write("\\bf GNN $\\bm{(3d)}$ & 8 & \\bf " + " & \\bf ".join([f"{m:.3f}" for m in metrics_GNN]) + "\\\\" + "\n")
        
        if "is_central" in df.columns:

            metrics_GNN_centrals = get_metrics(df[df.is_central].p_GNN_3d, df[df.is_central].log_Mstar)
            f.write("GNN $(3d)$ - centrals & 8 & " + " & ".join([f"{m:.3f}" for m in metrics_GNN_centrals]) + "\\\\"+ "\n")

            metrics_GNN_satellites = get_metrics(df[~df.is_central].p_GNN_3d, df[~df.is_central].log_Mstar)
            f.write("GNN $(3d)$ - satellites & 8 & " + " & ".join([f"{m:.3f}" for m in metrics_GNN_satellites]) + "\\\\"+ "\n")
    
def plot_comparison_figure(df, results_path=None):
    
    sc_kwargs = dict(edgecolor='white', s=3, linewidths=0.1, cmap=cmr.dusk, vmin=11, vmax=14)
    
    fig = plt.figure(figsize=(14, 3.75), dpi=300, constrained_layout=True)
    
    gs = fig.add_gridspec(1, 4, wspace=0.05, left=0.05, right=0.95, bottom=0.025, top=0.975, )
    ax1, ax2, ax3, ax4 = gs.subplots(sharey="row")

    ax1.scatter(df.log_Mstar, df.p_RF_Vmax, c=df.log_Mhalo, **sc_kwargs)
    ax1.text(0.025, 0.96, f"RF: $V_{{\\rm max}}$\n{np.sqrt(np.mean((df.p_RF_Vmax - df.log_Mstar)**2)):.3f} dex", va="top", transform=ax1.transAxes, fontsize=16)

    ax2.scatter(df.log_Mstar, df.p_RF_MhVmax, c=df.log_Mhalo, **sc_kwargs)
    ax2.text(0.025, 0.96, f"RF: $V_{{\\rm max}}+M_{{\\rm halo}}$\n{np.sqrt(np.mean((df.p_RF_MhVmax - df.log_Mstar)**2)):.3f} dex", va="top", transform=ax2.transAxes, fontsize=16)

    ax3.scatter(df.log_Mstar, df.p_RF_overdensity, c=df.log_Mhalo, **sc_kwargs)
    ax3.text(0.025, 0.96, f"RF: $V_{{\\rm max}}+M_{{\\rm halo}}+\\delta$\n{np.sqrt(np.mean((df.p_RF_overdensity - df.log_Mstar)**2)):.3f} dex", va="top", transform=ax3.transAxes, fontsize=16)
    
    sc = ax4.scatter(df.log_Mstar, df.p_GNN_3d, c=df.log_Mhalo, **sc_kwargs)
    ax4.text(0.025, 0.96, f"GNN\n{np.sqrt(np.mean((df.p_GNN_3d - df.log_Mstar)**2)):.3f} dex", va="top", transform=ax4.transAxes, fontsize=16)

    cb = fig.colorbar(sc, ax=[ax1, ax2, ax3, ax4], pad=0.02, shrink=0.83)
    cb.set_label("True log($M_{\\rm halo}/M_{\\odot}}$)", fontsize=14)


    for ax in [ax1, ax2, ax3, ax4]:
        ax.plot([0, 50], [0, 50], lw=1.5, c='w', zorder=9)
        ax.plot([0, 50], [0, 50], lw=1, c='0.5', zorder=10)
        ax.grid(alpha=0.15)
        ax.set_xlim(9, 12.5)
        ax.set_ylim(9, 12.5)
        ax.set_xticks([9, 10, 11, 12])
        ax.set_yticks([9, 10, 11, 12])

        if ax == ax1:
            ax.set_ylabel("Predicted log($M_{\\bigstar}/M_{\\odot}}$)", fontsize=14)
        ax.set_xlabel("True log($M_{\\bigstar}/M_{\\odot}}$)", fontsize=14)
        ax.set_aspect("equal")


    plt.savefig(f'{results_path}/GNN-vs-RF.png')
    plt.close()

    
def main(
    r_link, aggr, use_loops,
    n_hidden=64,
    n_latent=32,
    n_layers=4,
    n_unshared_layers=4,
):
    """Run the full pipeline"""

    results_path = f"{ROOT}/results/predicting-Mstar/gnns-upgraded_{aggr}_loops-{int(use_loops)}/r_link{r_link}"
    
    # make paths in case they don't exist
    Path(f"{results_path}/data").mkdir(parents=True, exist_ok=True)
    Path(f"{results_path}/training-logs").mkdir(parents=True, exist_ok=True)
    Path(f"{results_path}/models").mkdir(parents=True, exist_ok=True)
    

    
    pad = 5 # r_link / 2
    
    if not os.path.isfile(f"{results_path}/cross-validation.csv") or recompile_data or retrain or revalidate:
        for in_projection in [False, True]:
            import gc; gc.collect()
            proj_str = "-projected" if in_projection else ""

            data_path = f"{results_path}/data/" + f'gal2halo_split_{split**3}_link_{int(r_link)}_pad{int(pad)}_gal0{proj_str}.pkl'

            if os.path.isfile(data_path) and not recompile_data:
                print('File already exists: ', end='')
            else:
                print('Remaking dataset: ', end='')
                make_webs(
                    tng_base_path=tng_base_path, 
                    data_path=data_path,
                    snapshot=snapshot, 
                    r_link=r_link, 
                    pad=pad, 
                    split=split,
                    cuts=cuts, 
                    h=h, 
                    undirected=undirected, 
                    periodic=periodic,
                    use_loops=use_loops, 
                    in_projection=in_projection,
                )

            print(data_path)
            data = pickle.load(open(data_path, 'rb'))

            # retrain all data
            if retrain or not os.path.isfile(f"{results_path}/cross-validation.csv"):
                for k in range(split): 
                    print("Training!")
                    model = train_cosmic_gnn(
                        data, k=k, r_link=r_link, aggr=aggr, use_loops=use_loops, split=split, in_projection=in_projection, make_plots=make_plots, results_path=results_path, hidden_channels=n_hidden, latent_channels=n_latent, n_layers=n_layers
                    )
                    
                    validate_cosmic_gnn(model, data, k=k, split=split, in_projection=in_projection, make_plots=make_plots, results_path=results_path)
            # make sure cross-validation results exist (at least the final one)
            elif (revalidate and os.path.isfile(f"{results_path}/validation{proj_str}-fold{split}.csv")):
                print("Validating!")
                for k in range(split):
                    gc.collect();

                    node_features = data[0].x.shape[1]
                    out_features = data[0].y.shape[1]

                    model = EdgePointGNN(
                        node_features=node_features, 
                        n_layers=n_layers, 
                        n_unshared_layers=n_unshared_layers,
                        D_link=r_link,
                        hidden_channels=n_hidden,
                        latent_channels=n_latent,
                        loop=use_loops,
                        estimate_all_subhalos=True,
                        use_global_pooling=False,
                        n_out=out_features,
                        aggr=(["sum", "max", "mean"] if aggr == "multi" else aggr)
                    )

                    model.to(device);
                    model.load_state_dict(torch.load(f"{results_path}/models/EdgePointGNN-link{r_link}-hidden256-latent128-selfloops1-agg{aggr}-epochs1000_fold{k+1}.pth"))
                    validate_cosmic_gnn(model, data, k=k, split=split, in_projection=in_projection)
            else:
                print("Loaded 3d data, skipping projected version")
                break


        # keep track of which are centrals/sats
        is_central = np.concatenate([d.is_central for d in data])

        # combine results together and write outputs
        results = combine_results(split=split, centrals=is_central, results_path=results_path)
    else:
        print("Loading previous results (if you want to refresh the results, set retrain=True and/or revalidate=True)")
        results = pd.read_csv(f"{results_path}/cross-validation.csv")
    
    print("Saved out metrics in LaTeX table format")
    save_metrics(results, results_path=results_path)
    
    if make_plots:
        print("Visualizing graphs")
        for in_projection in [True, False]:
            proj_str = "-projected" if in_projection else ""
            data_path = f"{results_path}/data/" + f'gal2halo_split_{split**3}_link_{int(r_link)}_pad{int(pad)}_gal0{proj_str}.pkl'
            data = pickle.load(open(data_path, 'rb'))
            visualize_graph(data[-1], projection=("2d" if in_projection else "3d"), results_path=results_path)



    if make_plots:
        print("Saved RF and GNN comparison figure")
        plot_comparison_figure(results, results_path=results_path)

    
if __name__ == "__main__":
    aggr = args.aggr
    use_loops = args.loops
        
    for r_link in [3, 5, 7.5, 2.5, 4, 1.5]:
        main(r_link=r_link, aggr=aggr, use_loops=use_loops)
