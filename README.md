# Graph Neural Networks (GNNs) and simulated galaxies and halos

We have found that GNNs can learn the connection between galaxies and dark matter (sub)halos by incorporating information from galaxy surroundings on several-Mpc scales. Using subhalo catalogs from from cosmological hydrodynamic simulations, we found that GNNs outperform abundance matching and other ML methods.

This paper is now on [arXiv](https://arxiv.org/abs/2306.12327) and [NASA ADS](https://ui.adsabs.harvard.edu/abs/2023arXiv230612327W/abstract).

## Data

We make use of the `SUBFIND` snapshot 99 (redshift 0) subhalo catalogs derived from the Illustris TNG300-1 hydrodynamic simulation. All data can be accessed through the [IllustrisTNG website](https://www.tng-project.org/data/).

## Requirements

The most important requirements are `pytorch` and `pytorch-geometric`; check out the [latter's documentation](https://pytorch-geometric.readthedocs.io/en/latest/) for more information about installing it.


## Code

All of the scripts are contained in `./src`. In order to train a graph neural network (GNN) to estimate stellar mass from halo mass, please run `python src/painting_galaxies.py`.

## Citation 

If you want to cite this paper, you can do so with the following BibTeX blurb from ADS:

```
@ARTICLE{2023arXiv230612327W,
       author = {{Wu}, John F. and {Kragh Jespersen}, Christian},
        title = "{Learning the galaxy-environment connection with graph neural networks}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Astrophysics of Galaxies},
         year = 2023,
        month = jun,
          eid = {arXiv:2306.12327},
        pages = {arXiv:2306.12327},
archivePrefix = {arXiv},
       eprint = {2306.12327},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv230612327W},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
``` 

## Acknowledgments
Some of this code evolved out of [HaloGraphNet](https://github.com/PabloVD/HaloGraphNet) repository ([Villanueva-Domingo et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...935...30V/abstract)).

This project was made possible by the KITP Program, [*Building a Physical Understanding of Galaxy Evolution with Data-driven Astronomy*
](https://www.kitp.ucsb.edu/activities/galevo23) (see also the [website](https://datadrivengalaxyevolution.github.io/)).