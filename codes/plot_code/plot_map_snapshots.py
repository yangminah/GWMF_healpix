import numpy as np
import xarray as xr
import healpy as hp
import intake
import os
import cartopy.crs as ccrs
import cartopy.feature as cf
import cmocean
import matplotlib.pylab as plt
from IPython.display import Image
from PIL import Image

# import plotting functions defined in plot_worldmap.py
from plot_worldmap import nnshow, worldmap

# set up: what variables, levels, months to plot
variables = ['MFx', 'MFy']
levels = np.array([65, 60, 50, 41, 34, 30])
heights = np.array([6, 8, 12, 16, 20, 24])
truncs = [71, 214]

months = ['2021-01', '2021-04', '2021-07', '2021-10']

# where to save plots - new sub directories will be created within here for each level/variable/month
plot_dir = "/work/bm1233/icon_for_ml/spherical/plots/MF_maps/"
# where nextgems data is saved in zarr format
nextgems_dir = "/work/bm1233/icon_for_ml/spherical/nextgems3/"

for var in variables:
    for trunc in truncs:
        for month in months:
            ds = xr.open_zarr(f'{nextgems_dir}/res51km_{var}_trunc{trunc}.zarr/')
            ds = ds.sel(time = month, level_full = levels)
            ds[var].attrs = {'long_name': var, 'units': 'Pa'}
            ds.load()
            
            if trunc == 71:
                scale = 0.07
            elif trunc == 214:
                scale = 0.05
            
            for lev, h in zip(levels, heights):
                folder_path = f"{plot_dir}/level{lev}/snapshots_{var}/{month}/"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                    
                maxval = np.abs(ds.sel(level_full = lev)[var].values.flatten()).max()
                times = ds.time.values
                for t in times:
                    day = str(np.datetime64(t, 'D'))
                    timestamp = str(np.datetime64(t, 'h')).replace('-','')
                    worldmap(ds.sel(time = t, level_full = lev)[var], vmin = -scale*maxval, vmax = scale*maxval)
                    plt.title(f'{var}: {day}, level_full = {lev}')
                    plt.savefig(f'{folder_path}/{var}_level{lev}_trunc{trunc}_res51km_{timestamp}.png',dpi = 300, bbox_inches='tight')
                    
                image_list = []
                file_names = os.listdir(folder_path)
                file_names.sort()

                for filename in file_names:
                    if filename.endswith(".png"):
                        image_path = os.path.join(folder_path, filename)
                        img = Image.open(image_path)
                        image_list.append(img)
                
                output_gif_path = f'{plot_dir}/gifs/'
                if not os.path.exists(output_gif_path):
                    os.makedirs(output_gif_path)
                output_gif = f'{output_gif_path}/{var}_{h}km_trunc{trunc}_{month}.gif'
                image_list[0].save(output_gif, save_all=True, append_images=image_list[1:], duration=120, loop=0)
