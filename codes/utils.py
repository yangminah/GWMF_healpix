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
def nnshow(var, nx=1000, ny=1000, nest=True, ax=None, **kwargs):
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
    pix = hp.vec2pix(hp.npix2nside(len(var)), *xyz[valid].T, nest=nest)
    res = np.full(xyz.shape[:-1], np.nan, dtype=var.dtype)
    res[valid] = var[pix]
    return ax.imshow(res, extent=xlims+ylims, origin="lower", cmap = 'RdBu', **kwargs)

def worldmap(var, nest=True, **kwargs):
    projection = ccrs.Robinson(central_longitude=180.0)
    fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={"projection": projection}, constrained_layout=True)
    ax.set_global()

    im = nnshow(var, ax=ax, nest=nest, **kwargs)
    fig.colorbar(im, shrink = 0.8)#, label = var.long_name + ' (' + var.units + ')')
    ax.add_feature(cf.COASTLINE, linewidth=0.8)
    ax.add_feature(cf.BORDERS, linewidth=0.4)
def ang_range(npix, lat_start, lat_end, lon_start, lon_end):
    """
    Given an xarray dataset, return indices that belong to a box defined by
    lat_start < lat < lat_end
    lon_start < lon < lon_end.
    
    INPUTS:
    var should be something like ds.ua
    lat_start: beginning of lat range
    lat_end: end of lat range
    lon_start: beginning of lon range
    lon_end: end of lon range
    
    OUPUT:
    rect_cell_inds: an array of indices
    """
    nside=hp.pixelfunc.npix2nside(npix)
    pix_ind=np.arange(npix)
    lon_pix,lat_pix=hp.pixelfunc.pix2ang(nside, pix_ind, nest=True, lonlat=True)
    lat_inds=np.where(np.logical_and(lat_pix>=lat_start,lat_pix<=lat_end))[0]
    lon_inds=np.where(np.logical_and(lon_pix>=lon_start,lon_pix<=lon_end))[0]
    rect_cell_inds=np.intersect1d(lat_inds,lon_inds)
    return rect_cell_inds