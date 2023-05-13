import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
c0, c1, c2, c3, c4 = '#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600'
sc_kwargs = dict(s=0.1, edgecolor="none", alpha=1)

if __name__ == "__main__":
    df = pd.read_csv("~/halo-gnns/results/painting-galaxies/cross-validation_run1.csv")


    plt.figure(figsize=(4, 3.75), dpi=300)
    plt.scatter(df.log_Mhalo, df.log_Mstar, c=np.where(df.is_central, c3, c0), **sc_kwargs)
    plt.text(10.2, 12.2, "Central", color=c3, fontsize=16)
    plt.text(10.2, 11.9, "Satellite", color=c0, fontsize=16)
    plt.xlabel("$\\log(M_{\\rm halo}/M_\\odot)$", fontsize=14)
    plt.ylabel("$\\log(M_\\bigstar/M_\\odot)$", fontsize=14)
    plt.grid(alpha=0.15)
    plt.tight_layout()
    plt.xlim(10, 15)
    plt.ylim(9, 12.5)

    plt.savefig("/home/ubuntu/halo-gnns/results/painting-galaxies/smhm-relation.png")
