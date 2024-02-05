import numpy as np
import xarray as xr
import healpy as hp
import intake
import os
import sys
import time
sys.path.append('../')
from compute_fluxes_hp import *
import dask
from dask.distributed import Client

import ctypes
from datetime import datetime
from shutil import rmtree

import cartopy.crs as ccrs
import cartopy.feature as cf
#import cmocean
import matplotlib.pylab as plt
import matplotlib as mpl


season_months = {"DJF": [12, 1, 2],
                 "MAM": [3, 4, 5],
                 "JJA": [6, 7, 8],
                 "SON": [9, 10, 11],
                 "ANN": range(1, 13)}

# this file plots variables saved after running compute_power.py
# set up: what levels, truncs, seasons to plot 
levels = np.array([65, 60, 50, 41, 34, 30])
heights = np.array([6, 8, 12, 16, 20, 24])
truncs = [71, 214]
seasons = season_months.keys()


# where to save plots 
plot_dir = "/work/bm1233/icon_for_ml/spherical/plots/spectrum/"
# where nextgems data is saved - power spectrum saved as .npy files
nextgems_dir = "/work/bm1233/icon_for_ml/spherical/nextgems3/"

## Plotting functions
def kappa_from_deg(ls):
    """
        Returns wavenumber [km^-1] from spherical harmonics degree (ls)
        λ = 2π Re / sqrt[l(l + 1)]
        kappa = 1 / λ
    """
    earth_rad = 6371.2    # km
    return np.sqrt(ls * (ls + 1.0)) / (np.pi * earth_rad)

def plot_slope(x1, x2, y1, slope=-5/3, label="$k^{-5/3}$"):
    c = np.log(y1) - slope*np.log(x1)
    y2 = np.exp(slope * np.log(x2) + c)    
    plt.plot([x1, x2], [y1, y2], 'k--')
    midx = np.exp( np.log(x1)+(np.log(x2)-np.log(x1))/2)
    midy = np.exp( np.log(y1)+(np.log(y2)-np.log(y1))/2)
    plt.text(midx, midy, label,  va="bottom", ha="left")

for season in seasons:
    # Create an array of months written out in string form.
    months=[]
    N_month=60
    year=2020
    month=1       # first month 
    for month_ind in range(N_month):
        month +=1
        if month == 13:
            year+=1
            month=1
        ## Check if month is in our chosen season and add to list
        if month in season_months[season]:
            months.append(f"{year}-{month:02d}")

    print(season, months)

    vars = ["u","v"]
    KE_Cls = {}
    N_height = len(heights)

    print("Opening files")
    for trunc in truncs:
        Cls_trunc = np.zeros((N_height, trunc+1))
        for var in vars:
            Cl_var = np.zeros((N_height, trunc+1))
            # Take monthly means
            for month in months:
                filename = f"{nextgems_dir}/power/{var}/{trunc}/{month}.npy"
                with open(filename, 'rb') as f:
                    Cl_var += np.load(f)      # should be vector of length l_max+1
            Cl_var /= len(months)
        Cls_trunc += Cl_var           # u2 + v2
        KE_Cls[trunc] = Cls_trunc


    print("Plotting")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    linestyles=["solid","dashed"]
    colors = []
    cmap = mpl.cm.Reds(np.linspace(0.5,N_height+1))
    fig, ax = plt.subplots()

    for i,trunc in enumerate(truncs):
        for h, z in enumerate(heights):
            kappa = kappa_from_deg(np.arange(trunc+1))
            plt.loglog(kappa, KE_Cls[trunc][h], 
                       color=cmap[h],
                       linestyle=linestyles[i])

    custom_lines = [mpl.lines.Line2D([0], [0], color=cmap[h]) for h in range(N_height)] 
    custom_labels = [f"{z} km" for z in heights]

    ax.legend(custom_lines, custom_labels)
    plt.xlabel(r'$\kappa = 1/\lambda$ (km)')
    plt.ylabel(f'KE spectrum')
    plot_slope(2e-3, 6e-3, 2e1)
    plot_slope(5e-4, 1.5e-3, 1e3, -3, label="$k^{-3}$")
    plt.title(f"KE spectrum for {season}")

    plot_name = f'{plot_dir}/KE_spectrum_{season}.png'
    plt.savefig(plot_name)

    print(f"Plot saved as {plot_name}")

