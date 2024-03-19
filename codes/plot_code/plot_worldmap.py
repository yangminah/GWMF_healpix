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

## Define functions for plotting world map for snapshots and seasonal maps

def nnshow(var, nx=1000, ny=1000, ax=None, **kwargs):
    """
    var: variable on healpix coordinates (array-like)
    nx: image resolution in x-direction
    ny: image resolution in y-direction
    ax: axis to plot on
    kwargs: additional arguments to imshow
    """
    if ax is None:
        ax = plt.gca()

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    xvals = np.linspace(xlims[0], xlims[1], nx)
    yvals = np.linspace(ylims[0], ylims[1], ny)
    xvals2, yvals2 = np.meshgrid(xvals, yvals)
    xyz = ccrs.Geocentric().transform_points(ax.projection, xvals2, yvals2, np.zeros_like(xvals2))
    valid = np.all(np.isfinite(xyz), axis=-1)
    pix = hp.vec2pix(hp.npix2nside(len(var)), *xyz[valid].T, nest=False)
    res = np.full(xyz.shape[:-1], np.nan, dtype=var.dtype)
    res[valid] = var[pix]
    return ax.imshow(res, extent=xlims+ylims, origin="lower", **kwargs)

def worldmap(var, **kwargs):
    projection = ccrs.Robinson(central_longitude=0.0)
    fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={"projection": projection}, constrained_layout=True)
    ax.set_global()

    im = nnshow(var, ax=ax, **kwargs)
    fig.colorbar(im, shrink = 0.8, label = var.long_name + ' (' + var.units + ')')
    ax.add_feature(cf.COASTLINE, linewidth=0.8)
    ax.add_feature(cf.BORDERS, linewidth=0.4)
    
