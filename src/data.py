import h5py
import illustris_python as il
import numpy as np
import pandas as pd
from pathlib import Path
import torch

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

config_params = dict(
    boxsize=51.7e3,    # box size in comoving kpc/h
    h_reduced=0.704,   # reduced Hubble constant
    snapshot=99,       # z = 0
)

normalization_params = dict(
    minimum_n_star_particles=10., # min star particles to be considered a galaxy
    norm_half_mass_radius=8., 
    norm_velocity=100., # note: use value of 1 if `use_central_galaxy_frame=True`
)

# predict outputs: M_acc_dyn, log_stellar_halo_mass_ratio, log_halo_mass
science_params = dict(
    minimum_log_stellar_mass=8.,   # see https://arxiv.org/abs/2109.02713 (halo structure catalog)
    predict_output="log_halo_mass",     # predict log mass accretion in dynamical time (https://arxiv.org/abs/1703.09712)
)

feature_params = dict(
    use_stellarhalfmassradius=True,
    use_velocity=True,
    use_only_positions=False,
    use_central_galaxy_frame=False, # otherwise use center of mass frame
    in_projection=True, # only use projected positions and radial velocity
)

def correct_boundary(pos, boxlength=1.):
    """Correct periodic boundary conditions. 
    Originally from https://github.com/PabloVD/HaloGraphNet"""
    for i, pos_i in enumerate(pos):
        for j, coord in enumerate(pos_i):
            if coord > boxlength / 2.:
                pos[i, j] -= boxlength
            elif -coord > boxlength / 2.:
                pos[i, j] += boxlength

    return pos

def split_datasets(dataset, rng, valid_frac=0.15, test_frac=0.15, batch_size=128):
    """Adapted from https://github.com/PabloVD/HaloGraphNet"""

    rng.shuffle(dataset)

    num_train = len(dataset)
    split_valid = int(np.floor(valid_frac * num_train))
    split_test = split_valid + int(np.floor(test_frac * num_train))

    train_dataset = dataset[split_test:]
    valid_dataset = dataset[:split_valid]
    test_dataset = dataset[split_valid:split_test]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

def load_data(
    tng_base_path="../illustris_data/TNG50-1/output", 
    snapshot=99,  # i.e. z=0
    use_stellarhalfmassradius=True, 
    use_velocity=True, 
    use_only_positions=False, 
    in_projection=True
):
    """Loads Pandas DataFrame of halos, subhalos, and supplementary data in a given file."""

    # see https://www.tng-project.org/data/docs/specifications/
    subhalo_fields = [
        "SubhaloPos", "SubhaloMassType", "SubhaloLenType", "SubhaloHalfmassRadType", 
        "SubhaloVel", "SubhaloGrNr", "SubhaloFlag"
    ]
    subhalos = il.groupcat.loadSubhalos(tng_base_path, snapshot, fields=subhalo_fields) 

    halo_fields = ["Group_M_Crit200", "GroupFirstSub", "GroupPos", "GroupVel"]
    halos = il.groupcat.loadHalos(tng_base_path, snapshot, fields=halo_fields)

    subhalo_pos = subhalos["SubhaloPos"][:] / config_params["boxsize"]
    subhalo_stellarmass = subhalos["SubhaloMassType"][:,4]
    subhalo_n_stellar_particles = subhalos["SubhaloLenType"][:,4]
    subhalo_stellarhalfmassradius = subhalos["SubhaloHalfmassRadType"][:,4] / normalization_params["norm_half_mass_radius"]
    subhalo_vel = subhalos["SubhaloVel"][:] / normalization_params["norm_velocity"]
    subhalo_flag = subhalos["SubhaloFlag"][:]
    halo_id = subhalos["SubhaloGrNr"][:]

    halo_mass = halos["Group_M_Crit200"][:]
    halo_primarysubhalo = halos["GroupFirstSub"][:]  # currently not used but might be good for magnitude gap
    group_pos = halos["GroupPos"][:] / config_params["boxsize"]
    group_vel = halos["GroupVel"][:] / normalization_params["norm_velocity"]

    # get accretion history
    with h5py.File(f"{tng_base_path}/../postprocessing/halo_structure/halo_structure_099.hdf5", "r") as halo_structure:
        M_acc_dyn = halo_structure["M_acc_dyn"][:]

    # get subhalos/galaxies      
    subhalos = pd.DataFrame(
        np.column_stack([halo_id, subhalo_flag, np.arange(len(subhalo_stellarmass)), subhalo_pos, subhalo_vel, subhalo_n_stellar_particles, subhalo_stellarmass, subhalo_stellarhalfmassradius]), 
        columns=['halo_id', 'subhalo_flag', 'subhalo_id', 'subhalo_x', 'subhalo_y', 'subhalo_z', 'subhalo_vx', 'subhalo_vy', 'subhalo_vz', 'subhalo_n_stellar_particles', 'subhalo_stellarmass', 'subhalo_stellarhalfmassradius'],
    )
    subhalos = subhalos[subhalos["subhalo_flag"] != 0].copy()
    subhalos['halo_id'] = subhalos['halo_id'].astype(int)
    subhalos['subhalo_id'] = subhalos['subhalo_id'].astype(int)

    subhalos.drop("subhalo_flag", axis=1, inplace=True)

    # impose stellar mass and particle cuts
    subhalos = subhalos[subhalos["subhalo_n_stellar_particles"] > normalization_params["minimum_n_star_particles"]].copy()
    subhalos["subhalo_logstellarmass"] = np.log10(subhalos["subhalo_stellarmass"])
    subhalos = subhalos[subhalos["subhalo_logstellarmass"] + 10 > science_params["minimum_log_stellar_mass"]].copy()

    # get central halos (and only keep those with positive mass)
    halos = pd.DataFrame(
        np.column_stack((np.arange(len(halo_mass)), group_pos, group_vel, halo_mass, halo_primarysubhalo, M_acc_dyn)),
        columns=['halo_id', 'halo_x', 'halo_y', 'halo_z', 'halo_vx', 'halo_vy', 'halo_vz', 'halo_mass', 'halo_primarysubhalo', "M_acc_dyn"]
    )
    halos = halos[(halos["halo_mass"] > 0) & (halos["M_acc_dyn"].notna())].copy()
    halos["halo_id"] = halos['halo_id'].astype(int)
    halos["halo_logmass"] = np.log10(halos["halo_mass"])

    df = halos.join(subhalos.set_index('halo_id'), on='halo_id').set_index('halo_id')

    # remove extraneous columns
    df.drop(["subhalo_n_stellar_particles", "subhalo_stellarmass", "halo_mass"], axis=1, inplace=True)

    if not use_stellarhalfmassradius:
        df.drop(["subhalo_stellarhalfmassradius"], axis=1, inplace=True)
    
    if not use_velocity:
        df.drop(['subhalo_vx', 'subhalo_vy', 'subhalo_vz'], axis=1, inplace=True)
    
    if use_only_positions:
        df = df[[
            'halo_x', 'halo_y', 'halo_z', 'halo_logmass', 'halo_primarysubhalo', 
            'subhalo_id', 'subhalo_x', 'subhalo_y', 'subhalo_z', 
        ]].copy()

    if in_projection:
        df.drop(['subhalo_z', 'subhalo_vx', 'subhalo_vy'], axis=1, inplace=True)

    return df

def generate_dataset(df, use_velocity=True, use_central_galaxy_frame=False, use_only_positions=False, in_projection=True):
    """Iterate through a dataframe and create a PyG Data object"""

    dataset = []
    n_subhalos = 0

    for halo_id, subs in df.groupby(df.index):

        # select halos with at least one satellite (besides central)
        if subs.shape[0] <= 1: 
            continue

        if use_central_galaxy_frame:
            # shift positions and velocities to central galaxy rest frame
            for x in ['x', 'y', 'z', 'vx', 'vy', 'vz']:
                if x in subs.columns:
                    subs[f'subhalo_{x}'] -= subs.iloc[0][f'subhalo_{x}']
        else:
            # shift positions and velocities to halo rest frame
            for x in ['x', 'y', 'z', 'vx', 'vy', 'vz']:
                if x in subs.columns:
                    subs[f'subhalo_{x}'] -= subs[f'halo_{x}']

        # correct for periodic boundary conditions
        if in_projection:
            subs[['subhalo_x', 'subhalo_y']] = correct_boundary(
                subs[['subhalo_x', 'subhalo_y']].values
            )
        else:    
            subs[['subhalo_x', 'subhalo_y', 'subhalo_z']] = correct_boundary(
                subs[['subhalo_x', 'subhalo_y', 'subhalo_z']].values
            )

        # normalize velocities if using them
        if use_velocity:
            if use_central_galaxy_frame:
                subhalo_vel = np.log10(1.+np.sqrt(np.sum(subs[['subhalo_vx', 'subhalo_vy', 'subhalo_vz']].values**2., 1)))
            else:
                if in_projection:
                    subhalo_vel = np.log10(np.sqrt(np.sum(subs[['subhalo_vz']].values**2., 1)))
                else:  
                    subhalo_vel = np.log10(np.sqrt(np.sum(subs[['subhalo_vx', 'subhalo_vy', 'subhalo_vz']].values**2., 1)))
            if in_projection:
                features = np.column_stack((subs[['subhalo_x', 'subhalo_y', 'subhalo_logstellarmass', 'subhalo_stellarhalfmassradius']].values, subhalo_vel))
            else:
                features = np.column_stack((subs[['subhalo_x', 'subhalo_y', 'subhalo_z', 'subhalo_logstellarmass', 'subhalo_stellarhalfmassradius']].values, subhalo_vel))
        else:
            if in_projection:
                features = subs[['subhalo_x', 'subhalo_y', 'subhalo_stellarmass', 'subhalo_stellarhalfmassradius']].values
            else:
                features = subs[['subhalo_x', 'subhalo_y', 'subhalo_z', 'subhalo_stellarmass', 'subhalo_stellarhalfmassradius']].values

        # global features: N_subhalos, total stellar mass, 
        u = np.zeros((1,2), dtype=np.float32)
        u[0, 0] = subs.shape[0] 
        if not use_only_positions:
            u[0, 1] = np.log10(np.sum(10.**subs["subhalo_logstellarmass"]))

        # create a new feature, which is the stellar mass to halo mass ratio 
        if science_params["predict_output"] == "log_stellar_halo_mass_ratio":
            y = torch.tensor((subs["subhalo_logstellarmass"] - subs["halo_logmass"]).iloc[0], dtype=torch.float32)
        elif science_params["predict_output"] == "log_halo_mass":
            y = torch.tensor(subs[["halo_logmass"]].values[0], dtype=torch.float32)
        elif science_params["predict_output"] == "M_acc_dyn":
            y = torch.tensor(subs[["M_acc_dyn"]].values[0], dtype=torch.float32)
            
        # create pyg dataset
        graph = Data(
            x=torch.tensor(features, dtype=torch.float32), 
            pos=(
                torch.tensor(subs[['subhalo_x', 'subhalo_y']].values, dtype=torch.float32)
                if in_projection else
                torch.tensor(subs[['subhalo_x', 'subhalo_y', 'subhalo_z']].values, dtype=torch.float32)
            ),
            y=y,
            u=torch.tensor(u, dtype=torch.float32)
        )

        dataset.append(graph)
        n_subhalos += graph.x.shape[0]

    return dataset, n_subhalos

