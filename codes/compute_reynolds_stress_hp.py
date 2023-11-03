"""
This script contains all necessary functions to compute 
GW Momentum fluxes from data stored in healpix format.
"""
import os
import ctypes
from shutil import rmtree
import numpy as np
import xarray as xr
import healpy as hp
import intake
import dask
from dask.distributed import Client
from compute_fluxes_hp import get_task_id_dict, c_date2slice, get_nside, trim_memory

def main(base_dir: str = "/work/bm1233/icon_for_ml/spherical/nextgems3/"):
    """
    Start dask client, load in the data and compute the fluxes.
    """
    client = Client(processes=False, n_workers=4, threads_per_worker=4)
    task_id_dict = get_task_id_dict()
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    date, year, mon, day = task_id_dict[task_id]
    print(f"Task ID: {task_id}. Date: {date}")
    tmp_loc = f"{base_dir}tmp/wfull_{date}.zarr"

    l_max = 214
    nside_coarse = 64  # coarse grained res: nside 64: 100 km, nside 128: 50 km
    zoom_coarse = 6 # This is closest to 100km. (101.9)
    zoom = 10
    R_s = 287  # specific gas constant
    coarse_res = round(
        hp.nside2resol(nside_coarse, arcmin=True) / 60 * (2 * np.pi * 6371) / 360
    )

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
        "cell": 12 * (4**zoom), 
        "time": 1, 
        "level_full": 15, "level_half": 15
        },
    ).to_dask()
    date_slice = c_date2slice(ds, year, mon, day)

    ds_w = cat.ngc3028(
        zoom=zoom,
        time="PT3H",
        chunks={
            "cell": 12 * 4 ** (zoom - 3),
            "time": 1,
            "level_full": 90, "level_half": 91,
        },
    ).to_dask()

    ds_rho = cat.ngc3028(
        zoom=zoom_coarse,
        time="PT3H",
        chunks={
        "cell": 12 * (4**zoom_coarse), 
        "time": 1, 
        "level_full": 15, "level_half": 15
        },
    ).to_dask()


    ds = ds.sel(time=date)
    ds_w = ds_w.sel(time=date)
    nside = get_nside(ds)

    # Interpolate w to full levels, and save to disk in a temporary location.
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
        chunks={"cell": 12 * 4**zoom, "time": 1, "level_full": 15}
    ).to_zarr(tmp_loc)
    client.run(trim_memory)
    w_full = xr.open_zarr(tmp_loc)
    ds_w.close()

    # Calculate density
    density = (ds_rho.pfull / (R_s * ds_rho.ta)).to_dataset(name="rho")

    # Hash for ease of calling:
    da={
    "w" : w_full,
    "u" : ds["ua"],
    "v" : ds["va"],
    "rho": density
    }

    # Compute Products.
    da["uw"]: da[u] * da[w]
    da["uw"]: da[v] * da[w]

    # Coarsen.




if __name__ == "__main__":
    main()
