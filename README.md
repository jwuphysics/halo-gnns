# Graph Neural Networks (GNNs) and simulated galaxies and halos

## Data

Currently the data comes from the CAMELS `CV` runs of the IllustrisTNG simulations. One can obtain the data by downloading each simulation's `FOF_SUBIND` snapshot `033`, e.g.,

```
https://users.flatironinstitute.org/~camels/FOF_Subfind/IllustrisTNG/CV_0/fof_subhalo_tab_033.hdf5
https://users.flatironinstitute.org/~camels/FOF_Subfind/IllustrisTNG/CV_1/fof_subhalo_tab_033.hdf5
[...]
https://users.flatironinstitute.org/~camels/FOF_Subfind/IllustrisTNG/CV_26/fof_subhalo_tab_033.hdf5
```

## Requirements

You can duplicate the Python environment by installing from `./requirements.txt`:

```
conda create --file ./requirements.txt
```

which will install all of the prequisites like `pytorch-geometric` and so on. This was built on my personal machine, so you may need to relax the requirements a bit.

## Code

All of the scripts are contained in `./src`. You can train with all of the default hyperparameters by running

```
python src/main.py
```

The data collation is in `src/data.py`, the GNN model is in `src/model.py`, and the training routine is in `src/train.py`.

## Citations and acknowledgments

Much of this repository is based on the [HaloGraphNet](https://github.com/PabloVD/HaloGraphNet) repository ([Villanueva-Domingo et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...935...30V/abstract)).

This project was made possible by the KITP Program, [*Building a Physical Understanding of Galaxy Evolution with Data-driven Astronomy*
](https://www.kitp.ucsb.edu/activities/galevo23) (see also the [website](https://datadrivengalaxyevolution.github.io/)).