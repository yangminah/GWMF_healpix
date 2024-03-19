import numpy as np
import xarray as xr
import healpy as hp
import intake
import os
import cartopy.crs as ccrs
import cartopy.feature as cf
import cmocean
import matplotlib.pylab as plt

# import plotting functions defined in plot_worldmap.py
from plot_worldmap import nnshow, worldmap

# set up: what variables, levels to plot
variables = ['MFx', 'MFy']
levels = np.array([50, 41, 34, 24])
heights = np.array([12, 16, 20, 30])    # km (approx. 100, 40, 50, 10 hPa)
vmax = np.array([0.05, 0.05, 0.01, 0.01])                     #define colorbar max for each height level for comparison between them
truncs = [71, 214]

statistic = "std" # mean or std

if statistic == "mean":
    cmap = 'RdBu_r'
    vmin = -vmax
elif statistic == "std":
    cmap = "Reds"
    vmin = np.zeros_like(vmax)

# where to save plots - new sub directories will be created within here for each level/variable
plot_dir = "/work/bm1233/icon_for_ml/spherical/plots/MF_maps/"
# where nextgems data is saved in zarr format
nextgems_dir = "/work/bm1233/icon_for_ml/spherical/nextgems3/"

for var in variables:
    for trunc in truncs:
        print(f'{var}, truncation {trunc}')
        
        if trunc == 71:
            scale = 0.7
        elif trunc == 214:
            scale = 0.5

        ds = xr.open_zarr(f'{nextgems_dir}/res51km_{var}_trunc{trunc}.zarr/')
        ds[var].attrs = {'long_name': var, 'units': 'Pa'}

        if statistic == "mean":
            seasonal_means_over_five_years = ds.sel(level_full = levels).groupby('time.season').mean(dim='time')
        elif statistic == "std":
            seasonal_means_over_five_years = ds.sel(level_full = levels).groupby('time.season').std(dim='time')

        seasonal_means_over_five_years.load()

        seasons = seasonal_means_over_five_years.season.values
        seasons = list(seasons)
        seasons.append("ANN")
        print(seasons)

        for s in seasons:
            for k, (lev, h) in enumerate(zip(levels, heights)):                
                folder_path =  f"{plot_dir}/level{lev}/"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                
                if s == "ANN":
                    if statistic == "mean":
                        data = ds.sel(level_full = lev).mean(dim='time')
                    elif statistic == "std":
                        data = ds.sel(level_full = lev).std(dim='time')
                else:
                    data = seasonal_means_over_five_years.sel(season = s, level_full = lev)[var]

                #maxval = np.abs(seasonal_means_over_five_years.sel(season = s, level_full = lev)[var].values).max()
                worldmap(data, vmin = vmin[k], vmax = vmax[k], cmap = cmap)
                plt.title(f'{var}: {s}, level_full = {lev}, height = {h}km')
                plt.savefig(f'{folder_path}/{var}_{h}km_{s}_{statistic}_trunc{trunc}_res51km.png',dpi = 300, bbox_inches='tight')
        

