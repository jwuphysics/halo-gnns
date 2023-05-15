import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

c0, c1, c2, c3, c4 = '#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600'
pd.set_option('display.precision', 3)

ROOT = Path(__file__).parent.parent.resolve()

make_figures = True
split_by_mass = True

def rmse_2d_lowmass(df): 
    return ((df.log_Mhalo - df.p_GNN_2d)[df.log_Mstar < 10.5] ** 2).mean()**0.5

def rmse_2d_highmass(df): 
    return ((df.log_Mhalo - df.p_GNN_2d)[df.log_Mstar >= 10.5] ** 2).mean()**0.5

def rmse_3d_lowmass(df): 
    return ((df.log_Mhalo - df.p_GNN_3d)[df.log_Mstar < 10.5] ** 2).mean()**0.5

def rmse_3d_highmass(df): 
    return ((df.log_Mhalo - df.p_GNN_3d)[df.log_Mstar >= 10.5] ** 2).mean()**0.5

def rmse_2d(df):
    return ((df.log_Mhalo - df.p_GNN_2d) ** 2).mean()**0.5

def rmse_3d(df):
    return ((df.log_Mhalo - df.p_GNN_3d) ** 2).mean()**0.5


if __name__ == "__main__":
    
    D_links = [1, 2, 3, 5, 7.5, 10, 12.5, 15]
    
    
    results = dict()
    for loop_option in [0, 1]:
        for aggr_option in ["max", "sum"]:
            
            experiment_name = f"{aggr_option}_loops-{loop_option}"
            print(f"===== EXPERIMENT: {experiment_name} =====".center(72 if split_by_mass else 40))
            linking_length_results = list()
            for D in D_links:
                df = pd.read_csv(f"{ROOT}/results/inferring-halos_{experiment_name}/r_link{D}/cross-validation.csv")
                
                if split_by_mass:
                    linking_length_results.append([rmse_2d_lowmass(df), rmse_2d_highmass(df), rmse_3d_lowmass(df), rmse_3d_highmass(df)])
                else:
                    linking_length_results.append([rmse_2d(df), rmse_3d(df)])
            results_experiment = pd.DataFrame(
                linking_length_results, 
                columns=["results_2d_lo", "results_2d_hi", "results_3d_lo", "results_3d_hi"] if split_by_mass else ["results_2d", "results_3d"],
                index=D_links
            )
            print(results_experiment)
            
            if make_figures:
                figure_filename = f"{ROOT}/results/inferring-halos_{experiment_name}/linking-length-results{'_split-by-mass' if split_by_mass else ''}.png"
                plt.figure(figsize=(5, 5), dpi=150)
                
                if split_by_mass:
                    plt.plot(D_links, results_experiment.results_2d_hi, marker='o', c=c3, ls='--', lw=1.5,  label="2d hi")
                    plt.plot(D_links, results_experiment.results_3d_hi, marker='o', c=c3, ls='-', lw=3,  label="3d hi")
                    plt.plot(D_links, results_experiment.results_2d_lo, marker='o', c=c0, ls='--', lw=1.5, label="2d lo")
                    plt.plot(D_links, results_experiment.results_3d_lo, marker='o', c=c0, ls='-', lw=3,  label="3d lo")
                else: 
                    plt.plot(D_links, results_experiment.results_2d, marker='o', c=c3, ls='--', lw=1.5,  label="2d")
                    plt.plot(D_links, results_experiment.results_3d, marker='o', c=c3, ls='-', lw=3,  label="3d")
                    

                plt.legend(framealpha=1, fontsize=12, loc="lower right")
                plt.grid(alpha=0.15)
                plt.xlim(0, 16)
                plt.ylim(0.125, 0.425)
                plt.title(experiment_name, fontsize=14)
                plt.xlabel("Linking length (Mpc)", fontsize=12)
                plt.ylabel("Halo mass RMSE (dex)", fontsize=12)
                plt.tight_layout()
                plt.savefig(figure_filename)
                        