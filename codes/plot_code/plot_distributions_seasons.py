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
levels = np.array([50, 41, 34, 24 ])
heights = np.array([12, 16, 20, 30])
truncs = [71, 214]

# where to save plots - new sub directories will be created within here for each level/variable
plot_dir = "/work/bm1233/icon_for_ml/spherical/plots/MF_distributions/"
plot_dir_log = "/work/bm1233/icon_for_ml/spherical/plots/MF_log_distributions/"

# where nextgems data is saved in zarr format
nextgems_dir = "/work/bm1233/icon_for_ml/spherical/nextgems3/"

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
if not os.path.exists(plot_dir_log):
    os.makedirs(plot_dir_log)

## Functions we need
def get_lon_lat(nside, pixels):
    return hp.pix2ang(nside, pixels, lonlat=True)

def get_indices(lon, lat, lon_min, lon_max, lat_min, lat_max):
    if lon_min < lon_max:
        return (lon > lon_min) * (lon < lon_max) * (lat > lat_min) * (lat < lat_max)
    else:
        ## Need to separate LHS/RHS of lon=0 meridian
        LHS = (lon > lon_min) * (lon < 360.) * (lat > lat_min) * (lat < lat_max)
        RHS = (lon > 0.) * (lon < lon_max) * (lat > lat_min) * (lat < lat_max)
        return LHS + RHS


## regions
loon_regions = {
        "Indian Ocean"             :   (   40.,   110.,   -20.,    15. ) ,
        "Tropical Pacific"         :   (   120.,  280.,   -10.,    15. ) ,
        "Tropical Atlantic"        :   (   290.,   20.,   -10.,    15. ) ,
        "Southern Ocean"           :   (     0.,  360.,   -65.,   -35. ) ,
        "Extratropical Pacific"    :   (   120.,  260.,    20.,    45. ) ,
        "Extratropical Atlantic"   :   (   280.,   10.,    20.,    45. ) ,
        "Global"                   :   (     0.,  360.,   -90.,    90. )
        }


for var in variables:
    for trunc in truncs:
        print(f'{var}, truncation {trunc}')
        
        if trunc == 71:
            scale = 0.7
        elif trunc == 214:
            scale = 0.5

        ds = xr.open_zarr(f'{nextgems_dir}/res51km_{var}_trunc{trunc}.zarr/')
        ds[var].attrs = {'long_name': var, 'units': 'Pa'}

        seasonal_data = ds.sel(level_full = levels).groupby('time.season')
        print(seasonal_data)

        for s in  ["DJF", "MAM", "JJA", "SON", "ANN"]:
            for lev, h in zip(levels, heights):                
                if s == "ANN":
                    vals = ds.sel(level_full = lev)[var]

                else:
                    vals = seasonal_data[s].sel(level_full = lev)[var]

                # Size of arrays
                nsamples = vals.shape[0]
                pixels = ds.ring_cell
                npix = len(ds.ring_cell)
                nside = hp.npix2nside(npix)

                print(f"nsamples: {nsamples}, npix: {npix}, nside: {nside}")
                # Regions
                lon, lat = get_lon_lat(nside, pixels)
                
                for region in loon_regions.keys():
                    print(f"plotting region {region}")
                    lon_min, lon_max, lat_min, lat_max = loon_regions[region]
                    inds = get_indices(lon, lat, lon_min, lon_max, lat_min, lat_max)
                    region_vals = vals.isel(ring_cell=inds).values.flatten()*1000

                    # Plot distributions
                    plt.clf()
                    plt.hist(region_vals,  bins=np.arange(-500, 500.1, 10), histtype="step", density=True)
                    plt.xlabel(f"{var} (mPa)")
                    plt.title(f'{var}: {s}, region = {region}, level_full = {lev}, height = {h}km')
                    region_ = region.replace(' ','_')
                    plot_name = f'{plot_dir}/{region_}_{var}_{h}km_{s}trunc{trunc}.png'
                    plt.savefig(plot_name, dpi=100, bbox_inches='tight')
                    print(f"saved as {plot_name}")

                    # Plot log distributions
                    pos_inds = region_vals > 0.
                    plt.clf()
                    plt.hist(np.log(region_vals[pos_inds]),  bins=np.arange(-2, 2.1, 0.1), histtype="step", color="r", label="W", density=True)
                    plt.hist(np.log(-region_vals[~pos_inds]), bins=np.arange(-2, 2.1, 0.1), histtype="step", color="b", label="E", density=True)
                    plt.legend(loc='upper left')
                    plt.xlabel(f"log_10 of {var} in mPa")
                    plt.title(f'{var}: {s}, region = {region}, level_full = {lev}, height = {h}km')
                    region_ = region.replace(' ','_')
                    plot_name = f'{plot_dir_log}/{region_}_{var}_{h}km_{s}trunc{trunc}.png'
                    plt.savefig(plot_name, dpi=100, bbox_inches='tight')
                    print(f"saved as {plot_name}")
