#----------------------------------------------------
# Initial notebook for making the Illustris cosmic graph from galaxy catalogues
# Author: Christian Kragh Jespersen
# First created: 11/03/23 @KITP
# note: to make into routine later
#----------------------------------------------------

import h5py
import illustris_python as il
from torch_geometric.data import Data
import scipy.spatial as ss
import os, sys, torch, pickle
import os.path as osp
import pandas as pd
import numpy as np


# ## get illustris positions
# tng_base_path = osp.expanduser("~/../../scratch/gpfs/cj1223/TNG50")
# snapshot = 99
# h = 0.7 # cosmology h, set to 1 if you want comoving Mpc

# r_link = 4 #Mpc

# cuts = {"minimum_log_stellar_mass": 9,
#         "minimum_log_halo_mass": 8,
#        "minimum_n_star_particles": 100}

# undirected = True
# periodic = False
# use_loops = False

# use_gal = True # True = use galaxy params to infer dark matter, False = use DM params to infer galaxy stuff

# pad = 1 # how much padding to add around each subbox, so the separation in total will be 2 times this

cuts = {"minimum_log_stellar_mass": 9,
        "minimum_log_halo_mass": 8,
       "minimum_n_star_particles": 100}


def make_webs(tng_base_path = osp.expanduser("~/../../scratch/gpfs/cj1223/TNG50"), snapshot = 99, r_link = 4, pad = 1,\
              cuts = cuts, use_gal = True, h = 0.7, undirected = True, periodic = False, use_loops = False):
    
    if use_gal:
        use_cols = ['subhalo_x', 'subhalo_y', 'subhalo_z', 'subhalo_vx', 'subhalo_vy', 'subhalo_vz','subhalo_logstellarmass', 'subhalo_stellarhalfmassradius']
    y_cols = ['subhalo_loghalomass', 'subhalo_logvmax'] 
    if not use_gal:
        use_cols = ['subhalo_x', 'subhalo_y', 'subhalo_z', 'subhalo_vx', 'subhalo_vy', 'subhalo_vz', 'subhalo_loghalomass', 'subhalo_logvmax'] 
    y_cols = ['subhalo_logstellarmass', 'subhalo_stellarhalfmassradius']

    subhalo_fields = ["SubhaloPos", "SubhaloMassType"] #just an initial check here

    subhalos = il.groupcat.loadSubhalos(tng_base_path, snapshot, fields=subhalo_fields) 

    pos = subhalos["SubhaloPos"][:,:3]
    min_box, max_box = np.rint(np.min(pos)), np.rint(np.max(pos))

    box_size = max_box/(h*1e3) #/(h*1000), pos units are in kpc


    subhalo_fields = [
            "SubhaloPos", "SubhaloMassType", "SubhaloLenType", "SubhaloHalfmassRadType", 
            "SubhaloVel", "SubhaloVmax", "SubhaloGrNr", "SubhaloFlag"
        ]
    subhalos = il.groupcat.loadSubhalos(tng_base_path, snapshot, fields=subhalo_fields) 

    halo_fields = ["Group_M_Crit200", "GroupFirstSub", "GroupPos", "GroupVel"]
    halos = il.groupcat.loadHalos(tng_base_path, snapshot, fields=halo_fields)

    subhalo_pos = subhalos["SubhaloPos"][:] / (h*1e3) #/(h*1000), pos units are in comoving kpc, so now in Mpc
    subhalo_stellarmass = subhalos["SubhaloMassType"][:,4]
    subhalo_halomass = subhalos["SubhaloMassType"][:,1]
    subhalo_n_stellar_particles = subhalos["SubhaloLenType"][:,4]
    subhalo_stellarhalfmassradius = subhalos["SubhaloHalfmassRadType"][:,4] #normalize?
    subhalo_vel = subhalos["SubhaloVel"][:] #normalize?
    subhalo_vmax = subhalos["SubhaloVmax"][:] #normalize?
    subhalo_flag = subhalos["SubhaloFlag"][:]
    halo_id = subhalos["SubhaloGrNr"][:]

    halo_mass = halos["Group_M_Crit200"][:]
    halo_primarysubhalo = halos["GroupFirstSub"][:]  # currently not used but might be good for magnitude gap
    group_pos = halos["GroupPos"][:] / (h*1e3)
    group_vel = halos["GroupVel"][:]  #normalize?

    # get subhalos/galaxies      
    subhalos = pd.DataFrame(
        np.column_stack([halo_id, subhalo_flag, np.arange(len(subhalo_stellarmass)), subhalo_pos, subhalo_vel, subhalo_n_stellar_particles, subhalo_stellarmass, subhalo_halomass, subhalo_stellarhalfmassradius, subhalo_vmax]), 
        columns=['halo_id', 'subhalo_flag', 'subhalo_id', 'subhalo_x', 'subhalo_y', 'subhalo_z', 'subhalo_vx', 'subhalo_vy', 'subhalo_vz', 'subhalo_n_stellar_particles', 'subhalo_stellarmass', 'subhalo_halomass', 'subhalo_stellarhalfmassradius', 'subhalo_vmax'],
    )
    subhalos = subhalos[subhalos["subhalo_flag"] != 0].copy()
    subhalos['halo_id'] = subhalos['halo_id'].astype(int)
    subhalos['subhalo_id'] = subhalos['subhalo_id'].astype(int)

    subhalos.drop("subhalo_flag", axis=1, inplace=True)

    # impose stellar mass and particle cuts
    subhalos = subhalos[subhalos["subhalo_n_stellar_particles"] > cuts["minimum_n_star_particles"]].copy()
    subhalos["subhalo_logstellarmass"] = np.log10(subhalos["subhalo_stellarmass"])+10

    subhalos["subhalo_loghalomass"] = np.log10(subhalos["subhalo_halomass"])+10
    subhalos["subhalo_logvmax"] = np.log10(subhalos["subhalo_vmax"])
    subhalos["subhalo_logstellarhalfmassradius"] = np.log10(subhalos["subhalo_stellarhalfmassradius"])

    subhalos = subhalos[subhalos["subhalo_loghalomass"] > cuts["minimum_log_halo_mass"]].copy()

    subhalos = subhalos[subhalos["subhalo_logstellarmass"] > cuts["minimum_log_stellar_mass"]].copy()

    subhalos.reset_index(drop = True)

    data = []
    for n in range(2):
        for g in range(2):
            for k in range(2):
                print(n,g,k)
                xlims = np.array([box_size/2*n+pad, box_size/2*(n+1)-pad])
                ylims = np.array([box_size/2*g+pad, box_size/2*(g+1)-pad])
                zlims = np.array([box_size/2*k+pad, box_size/2*(k+1)-pad])

                pos = np.vstack(subhalos[['subhalo_x', 'subhalo_y', 'subhalo_z']].to_numpy())

                xmask = np.logical_and(pos[:,0]>xlims[0],pos[:,0]<xlims[1])
                ymask = np.logical_and(pos[:,1]>ylims[0],pos[:,1]<ylims[1])
                zmask = np.logical_and(pos[:,2]>zlims[0],pos[:,2]<zlims[1])
                mask = np.logical_and(zmask, np.logical_and(xmask, ymask))

                df = subhalos.iloc[mask]
                df.reset_index(drop=True)

                # remove extraneous columns
                df.drop(["subhalo_n_stellar_particles", "subhalo_stellarmass", "subhalo_halomass"], axis=1, inplace=True)

                # set new zero point

                df[['subhalo_x', 'subhalo_y', 'subhalo_z']] = df[['subhalo_x', 'subhalo_y', 'subhalo_z']] - np.array([box_size/2*n+pad, box_size/2*g+pad, box_size/2*k+pad])

                #make positions for clustering

                pos = np.vstack(df[['subhalo_x', 'subhalo_y', 'subhalo_z']].to_numpy())

                kd_tree = ss.KDTree(pos, leafsize=25, boxsize=box_size)
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
                    atrloops = np.zeros((pos.shape[0],3))
                    for i, posit in enumerate(pos):
                        loops[0,i], loops[1,i] = i, i
                        atrloops[i,0], atrloops[i,1], atrloops[i,2] = 0., 1., 0.
                    edge_index = np.append(edge_index, loops, 1)
                    edge_attr = np.append(edge_attr, atrloops, 0)
                edge_index = edge_index.astype(int)

                x =  torch.tensor(np.vstack(df[use_cols].to_numpy()), dtype=torch.float)
                y =  torch.tensor(np.vstack(df[y_cols].to_numpy()), dtype=torch.float)
                edge_index = torch.tensor(edge_index, dtype=torch.long)
                edge_attr=torch.tensor(edge_attr, dtype=torch.float)

                data.append(Data(x = x, y= y, edge_index = edge_index, edge_attr = edge_attr))

                data_path = osp.join(tng_base_path, 'cosmic_graphs', f'split_{len(data)}_link_{int(r_link)}_pad{int(pad)}_gal{int(use_gal)}.pkl')

                if not osp.isdir(osp.join(tng_base_path, 'cosmic_graphs')):
                    os.mkdir(osp.join(tng_base_path, 'cosmic_graphs'))

                with open(data_path, 'wb') as handle:
                    pickle.dump(data, handle)
                  