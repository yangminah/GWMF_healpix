
"""
This script contains all necessary functions to compute 
GW Momentum fluxes from data stored in healpix format.
"""
import os
from shutil import rmtree
import numpy as np
import xarray as xr
import healpy as hp
import intake
import dask
import dask.array as da
from dask.distributed import Client
import logging
from compute_fluxes_hp import (
    get_task_id_dict, 
    c_date2slice, 
    trim_memory, 
    flat2wavenumber, 
    compute_taper_coeffs, 
    map2alm_xr, 
    alm2map_xr,
    total_mn
    )
from compute_uugs_nofilter import ud_grade_xr

def tmp_loc(v, date,base_dir="/work/bm1233/icon_for_ml/spherical/nextgems3/"):
    return f"{base_dir}tmp/{v}_{date}.zarr"

def save_coarse(v, da_coarse, coarse_res, date_slice, t, locstr=None):
    """
    Save coarsened variables to preallocated space on disk.
    """
    #logger.info(coarse_res)
    #if locstr is None:
    locstr = "/work/bm1233/icon_for_ml/spherical/nextgems3/reynolds/"
    filename_coarse = f"{locstr}res{coarse_res}km_{v}_trunc{trunc}.zarr"
    da_coarse = (
        da_coarse.to_dataset(name=v)
        if da_coarse.__class__ is not xr.core.dataset.Dataset
        else da_coarse
    )
    c = da_coarse["ring_cell"].shape[0]
    lev = da_coarse["level_full"].shape[0]
    da_coarse.to_zarr(
        filename_coarse,
        mode="r+",
        region={
            "time": date_slice,
            "level_full": slice(0, lev),
            "ring_cell": slice(0, c),
            
        },
    )
def nest2ring(ds, var, nside):
    """
    Change order from nested to ring
    """
    ring_cells = hp.pixelfunc.ring2nest(nside, ds.cell)
    with dask.config.set(**{"array.slicing.split_large_chunks": True}): #False
        ds_ring = (
            ds[var]
            .isel(cell=ring_cells)
            .assign_coords(ring_cell=("cell", np.arange(len(ds.cell))))
            .swap_dims({"cell": "ring_cell"})
        )
    return ds_ring

def mul(sphharm, taper_coeffs):
    """
    Multiply spherical harmonics to the taper coefficients.
    """
    return sphharm * taper_coeffs


def main():
    logger = logging.getLogger(__name__)
    locstr = "/work/bm1233/icon_for_ml/spherical/nextgems3/"
    base_dir="/work/bm1233/icon_for_ml/spherical/nextgems3/"

    # Start dask client
    client = Client(processes=False, n_workers=2, threads_per_worker=1)
    NCELL = 12582912
    NSP = 23220

    # Get task_id and convert to date.
    task_id_dict = get_task_id_dict()
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    date, year, mon, day = task_id_dict[task_id]
    date_slice = c_date2slice(ds, year, mon, day)
    logger.info(f"Task ID: {task_id}. Date: {date}. Slice: {date_slice}.")


    # Set beginning and end resolutions.
    # coarse grained res/nside/zoom: 102km/64/6, 51km/128/7, 6km/1024/10.
    # Note that nside = exp2(zoom).
    nside_coarse = 128  
    zoom_coarse = 7
    NCELL_COARSE = 12 * (4**zoom_coarse)
    nside = 1024
    zoom = 10
    coarse_res = round(
        hp.nside2resol(nside_coarse, arcmin=True) / 60 * (2 * np.pi * 6371) / 360
    )
    trunc=214

    # NextGems Cycle 3 run:
    # Here we load ds_rho for density, ds_w for w, and ds for all remaining variables.
    # ds_w is chunked differently to optimize computing the vertical interpolation for w.
    cat = intake.open_catalog(
        "/work/bm1235/k203123/NextGEMS_Cycle3.git/experiments/ngc3028/outdata/ngc3028.yaml"
    )

    ds = cat.ngc3028(
        zoom=zoom,
        time="PT3H",
        chunks={
            "cell": NCELL,
            "time": 1,
            "level_full": 90,
            "level_half": 91,
        },
    ).to_dask().sel(time=date)

    ds_w = cat.ngc3028(
        zoom=zoom,
        time="PT3H",
        chunks={
            "cell": NCELL_COARSE,
            "time": 1,
            "level_full": 90,
            "level_half": 91,
        },
    ).to_dask().sel(time=date)

    # Interpolate w to full levels, and save to disk in a temporary location.
    logger.info("Interpolating w to full levels.")
    client.run(trim_memory)
    l_full = np.arange(0, 91)
    w_full = ds_w["wa_phy"].rolling(level_half=2, min_periods=2).mean()
    w_full = (
        w_full.assign_coords(level_full=("level_half", l_full))
        .swap_dims({"level_half": "level_full"})
        .drop("level_half")
        .drop_sel(level_full=0)
        .to_dataset()
    )
    w_full = w_full.drop_vars("zghalf")
    w_full["wa_phy"].chunk(
        chunks={"cell": 12 * 4**zoom, "time": 1, "level_full": 1}
    ).to_zarr(tmp_loc("w_full",date))

    # w_full = xr.open_zarr(tmp_loc("w_full",date)).chunk(
    # {"level_full": 1,"time":1,"cell": 12 * 4**zoom}
    # )
    client.run(trim_memory)
    ds_w.close()
    logger.info("Done interpolating w.")

    # Compute w perturbation.
    logger.info("Compute w perturbation.")
    da = {}
    taper_dict={}
    taper_coeffs = compute_taper_coeffs(t)
    v = 'w'
    dsvar = xr.open_zarr(f"{base_dir}tapered_{v}_trunc{trunc}.zarr")
    dsvar = dsvar.sel(time=date).chunk(
        {"level_full": 1,"time":1,"nsp":NSP}
        )
    tmp = dask.compute(
        nest2ring(w_full, "wa_phy", nside).rename(v) - alm2map_xr(dsvar, nside)[v]
        )[0]
    del w_full
    client.run(trim_memory)
    tmp.to_zarr(tmp_loc(v,date))
    del tmp
    client.run(trim_memory)

    # Compute u and v perturbations.
    for v in ['u','v']:
        dsvar = xr.open_zarr(f"{base_dir}tapered_{v}_trunc{trunc}.zarr")
        dsvar = dsvar.sel(time=date)
        tmp = dask.compute(nest2ring(ds, f"{v}a", nside).rename(v)- alm2map_xr(dsvar, nside)[v])[0]
        tmp.to_zarr(tmp_loc(v,date))
        del tmp
        client.run(trim_memory)

    # Place dask array in dict.
    da={}
    for v in ['u','v','w']:
        da[v] = xr.open_zarr(tmp_loc(v, date)).chunk(
            {"level_full": 1,"time":1,"ring_cell": NCELL}
            )

    # Multiply and taper in spectral space.
    for v in ['u','v']:
        # v = 'u'
        var = f"{v}w"

        # Multiply to w'.
        da[var] = (da[v][v] * da['w']['w']).rename(var).chunk(
            {"level_full": 1,"time":1,"ring_cell": NCELL}
            )
        da[var] = dask.compute(da[var])[0]
        del da[v]
        client.run(trim_memory)
        da_scattered = client.scatter(da[var])

        # Apply spectral transformation.
        tmp = client.submit(map2alm_xr, da_scattered, "ring_cell", t)

        # Taper.
        tmp = client.submit(mul, tmp, taper_coeffs)
        taper_dict[var] = dask.compute(tmp.result())[0]
        del tmp
        del da[var]
        rmtree(tmp_loc(v, date))
        client.run(trim_memory)

    # Clean up to free memory.
    del da
    client.run(trim_memory)
    rmtree(tmp_loc('w', date))

    # Filter (taper) rho.
    trho = xr.open_zarr(f"{locstr}tapered_rho_trunc{trunc}.zarr").rho.sel(time=date)
    trho = alm2map_xr(trho, nside_coarse).real
    trho = dask.compute(trho)[0]

    # Save momentum fluxes.
    for v,v2 in zip(['u','v'],['x','y']):
        var = v+'w'
        tmp = alm2map_xr(taper_dict[var], nside_coarse).real
        MF = (tmp * trho).rename("MF"+v2)
        save_coarse("MF"+v2,MF, coarse_res, date_slice, t)


    # Compute drags.
    logger.info("Compute drags.")
    from compute_drag_reynolds import *

    variables = ['MFx', 'MFy']
    truncs = [71 , 214]
    for trunc in truncs:
        for var in variables:
            calc_drag(task_id, var, trunc, coarse_res, nside_coarse)

if __name__ == "__main__":
    main()
