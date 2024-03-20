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
levels = np.array([50, 41, 34, 24])[:1]
heights = np.array([12, 16, 20, 30])[:1]    # km (approx. 100, 40, 50, 10 hPa)
vmax = np.array([0.01, 0.01, 0.01, 0.01])                     #define colorbar max for each height level for comparison between them
truncs = [71, 214]


# where to save plots - new sub directories will be created within here for each level/variable
plot_dir = "/work/bm1233/icon_for_ml/spherical/plots/MF_maps/"
# where nextgems data is saved in zarr format
nextgems_dir = "/work/bm1233/icon_for_ml/spherical/nextgems3/"

for var in variables:
    for trunc in truncs:
        print(f'{var}, truncation {trunc}')

        ds = xr.open_zarr(f'{nextgems_dir}/res51km_{var}_trunc{trunc}.zarr/')
        ds[var].attrs = {'long_name': var, 'units': 'Pa'}

        seasonal_means_over_five_years = ds.sel(level_full = levels).groupby('time.season').mean(dim='time')
        seasonal_stds_over_five_years = ds.sel(level_full = levels).groupby('time.season').std(dim='time')
        ann_means_over_five_years = ds.sel(level_full = levels).mean(dim='time')
        ann_stds_over_five_years = ds.sel(level_full = levels).std(dim='time')


        #seasonal_means_over_five_years.load()

        seasons = seasonal_means_over_five_years.season.values
        seasons = list(seasons)
        seasons.append("ANN")
        print(seasons)

        for s in seasons:
            print(s)
            for k, (lev, h) in enumerate(zip(levels, heights)):                
                folder_path =  f"{plot_dir}/level{lev}/"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                
                # Plot mean
                if s == "ANN":
                    data = ann_means_over_five_years.sel(level_full = lev)[var]
                else:
                    data = seasonal_means_over_five_years.sel(season = s, level_full = lev)[var]
                worldmap(data, vmin = -vmax[k], vmax = vmax[k], cmap = "RdBu_r")
                plt.title(f'{var}: {s}, level_full = {lev}, height = {h}km')
                plt.savefig(f'{folder_path}/{var}_{h}km_{s}_mean_trunc{trunc}_res51km.png',dpi = 300, bbox_inches='tight')
                plt.close()

                # Plot std
                plt.clf()
                if s == "ANN":
                    data = ann_stds_over_five_years.sel(level_full = lev)[var]
                else:
                    data = seasonal_stds_over_five_years.sel(season = s, level_full = lev)[var]
                worldmap(data, vmin = 0., vmax = vmax[k], cmap = "Reds")
                plt.title(f'{var}: {s}, level_full = {lev}, height = {h}km')
                plt.savefig(f'{folder_path}/{var}_{h}km_{s}_std_trunc{trunc}_res51km.png',dpi = 300, bbox_inches='tight')
                plt.close()


        

