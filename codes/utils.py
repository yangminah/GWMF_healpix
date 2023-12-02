"""
Helper functions used across various modules.
"""
import numpy as np
import torch
import healpy as hp
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pylab as plt

R_DRY = 287.04
C_P = 7 * R_DRY / 2
GRAV = 9.8


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
    xyz = ccrs.Geocentric().transform_points(
        ax.projection, xvals2, yvals2, np.zeros_like(xvals2)
    )
    valid = np.all(np.isfinite(xyz), axis=-1)
    pix = hp.vec2pix(hp.npix2nside(len(var)), *xyz[valid].T, nest=nest)
    res = np.full(xyz.shape[:-1], np.nan, dtype=var.dtype)
    res[valid] = var[pix]
    return ax.imshow(res, extent=xlims + ylims, origin="lower", cmap="RdBu", **kwargs)


def worldmap(var, nest=True, **kwargs):
    """
    Place land borders.
    """
    projection = ccrs.Robinson(central_longitude=180.0)
    fig, ax = plt.subplots(
        figsize=(8, 4), subplot_kw={"projection": projection}, constrained_layout=True
    )
    ax.set_global()

    im = nnshow(var, ax=ax, nest=nest, **kwargs)
    fig.colorbar(im, shrink=0.8)  # , label = var.long_name + ' (' + var.units + ')')
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
    nside = hp.pixelfunc.npix2nside(npix)
    pix_ind = np.arange(npix)
    lon_pix, lat_pix = hp.pixelfunc.pix2ang(nside, pix_ind, nest=True, lonlat=True)
    lat_inds = np.where(np.logical_and(lat_pix >= lat_start, lat_pix <= lat_end))[0]
    lon_inds = np.where(np.logical_and(lon_pix >= lon_start, lon_pix <= lon_end))[0]
    rect_cell_inds = np.intersect1d(lat_inds, lon_inds)
    return rect_cell_inds


def find_lowest_true(change: torch.Tensor) -> torch.Tensor:
    """
    Find lowest occuring true by converting to a sparse matrix.
    """
    levs, phase_speeds = change.to_sparse().indices()
    change_list = -torch.ones(1, change.shape[1], dtype=torch.int64)
    for i in set(phase_speeds.tolist()):
        change_list[0, i] = torch.max(levs[phase_speeds == i])
    return change_list


def find_lowest_true_vectorized(x: torch.Tensor, dimlev=1, levshape=None):
    """
    Iterate over levels to find levels of lowest occurences of True.
    """
    if levshape is None:
        levshape = list(x.shape)
        levshape.remove(levshape[dimlev])
    lev = -torch.ones(levshape, dtype=torch.int64)
    for i in range(x.shape[dimlev] - 1, -1, -1):
        lev = torch.where(
            (lev == -1) & x.select(dimlev, i),
            i * torch.ones(levshape, dtype=torch.int64),
            lev,
        )
    return lev.unsqueeze(dimlev)


def find_uppermost_true_vectorized(x: torch.Tensor, dimlev=1, levshape=None):
    """
    Iterate over levels to find levels of highest occurences of True.
    """
    if levshape is None:
        levshape = list(x.shape)
        levshape.remove(levshape[dimlev])
    lev = -torch.ones(levshape, dtype=torch.int64)
    for i in range(0, x.shape[dimlev]):
        lev = torch.where(
            (lev == -1) & x.select(dimlev, i),
            i * torch.ones(levshape, dtype=torch.int64),
            lev,
        )
    return lev.unsqueeze(dimlev)


def find_uppermost_false_vectorized(x: torch.Tensor, dimlev=1, levshape=None):
    """
    Iterate over levels to find levels of uppermost occurences of False.
    """
    if levshape is None:
        levshape = list(x.shape)
        levshape.remove(levshape[dimlev])
    lev = -torch.ones(levshape, dtype=torch.int64)
    for i in range(0, x.shape[dimlev]):
        lev = torch.where(
            (lev == -1) & (not x.select(dimlev, i)),
            i * torch.ones(levshape, dtype=torch.int64),
            lev,
        )
    return lev.unsqueeze(dimlev)


def find_lowest_true_loop(change: torch.Tensor) -> torch.Tensor:
    """
    Iterate over levels to find levels of lowest occurences of True by 
    converting to a sparse matrix over other dims.
    """
    change_shape = list(change.shape)
    change_shape[1] = 1
    change_list = -torch.ones(change_shape, dtype=torch.int64)
    for t in range(change.shape[0]):
        for l in range(change.shape[2]):
            levs, phase_speeds = change[t, :, l, :].to_sparse().indices()
            for p in set(phase_speeds.tolist()):
                change_list[t, 0, l, p] = torch.max(levs[(phase_speeds == p)])
    return change_list
