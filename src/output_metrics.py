import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from pathlib import Path
ROOT = Path(__file__).parent.parent.resolve()


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


def save_metrics(df):
    """Save dataframe of results"""

    metrics_Mh = get_metrics(df.p_RF_Mhalo, df.log_Mstar)
    metrics_Vmax = get_metrics(df.p_RF_Vmax, df.log_Mstar)
    metrics_combined = get_metrics(df.p_RF_combined, df.log_Mstar)
    metrics_GNN_proj = get_metrics(df.p_GNN_2d, df.log_Mstar)
    metrics_GNN = get_metrics(df.p_GNN_3d, df.log_Mstar)
    metrics_GNN_centrals = get_metrics(df[df.is_central].p_GNN_3d, df[df.is_central].log_Mstar)
    metrics_GNN_satellites = get_metrics(df[~df.is_central].p_GNN_3d, df[~df.is_central].log_Mstar)
    
    metrics = pd.DataFrame(
        np.array([metrics_Mh, metrics_Vmax, metrics_combined, metrics_GNN_proj, metrics_GNN, metrics_GNN_centrals, metrics_GNN_satellites]),
        columns=["rmse", "nmad", "mae", "pearson_rho", "r2", "bias", "f_outlier"],
    )
    metrics.index = ["RF_Mhalo", "RF_Vmax", "RF_combined", "GNN_2d", "GNN_3d", "GNN_centrals", "GNN_satellites"]
    
    return metrics
    
            
if __name__ == "__main__":

    metrics = np.dstack([
        save_metrics(pd.read_csv(f"{ROOT}/results/painting-galaxies/cross-validation_run{i}.csv")).values for i in range(1, 4)
    ])
    
    metrics_avg =  pd.DataFrame(
        metrics.mean(2),
        columns=["rmse", "nmad", "mae", "pearson_rho", "r2", "bias", "f_outlier"],
        index=["RF_Mhalo", "RF_Vmax", "RF_combined", "GNN_2d", "GNN_3d", "GNN_centrals", "GNN_satellites"]
    )
    
    metrics_std =  pd.DataFrame(
        metrics.std(2),
        columns=["rmse", "nmad", "mae", "pearson_rho", "r2", "bias", "f_outlier"],
        index=["RF_Mhalo", "RF_Vmax", "RF_combined", "GNN_2d", "GNN_3d", "GNN_centrals", "GNN_satellites"]
    )
    
    print("=== Averages (N=3) ===")
    print(metrics_avg)
    print("=== Standard Deviations (N=3) ===")
    print(metrics_std)
