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
from dask.distributed import Client
from compute_fluxes_hp import get_task_id_dict, c_date2slice, trim_memory


def ud_grade_xr(da, nside_coarse):
    """
    Coarsen to nside_coarse.
    """
    npix = hp.nside2npix(nside_coarse)
    return xr.apply_ufunc(
        hp.ud_grade,
        da,
        input_core_dims=[["cell"]],
        output_core_dims=[["out_cell"]],
        kwargs={"nside_out": nside_coarse},
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"out_cell": npix}, "allow_rechunk": True},
        output_dtypes=["<f4"],
    )


def save_coarse(v, da_coarse, coarse_res, date_slice, locstr=None):
    """
    Save coarsened variables to preallocated space on disk.
    """
    if locstr is None:
        locstr = "/work/bm1233/icon_for_ml/spherical/nextgems3/reynolds/"
    filename_coarse = f"{locstr}res{coarse_res}km_{v}.zarr"
    da_coarse = (
        da_coarse.to_dataset(name=v)
        if da_coarse.__class__ is not xr.core.dataset.Dataset
        else da_coarse
    )
    c = da_coarse["cell"].shape[0]
    lev = da_coarse["level_full"].shape[0]
    da_coarse.to_zarr(
        filename_coarse,
        mode="r+",
        region={
            "time": date_slice,
            "cell": slice(0, c),
            "level_full": slice(0, lev),
        },
    )


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

    nside_coarse = 64  # coarse grained res: nside 64: 100 km, nside 128: 50 km
    zoom_coarse = 6  # This is closest to 100km. (101.9)
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
            "level_full": 15,
            "level_half": 15,
        },
    ).to_dask()
    date_slice = c_date2slice(ds, year, mon, day)

    ds_w = cat.ngc3028(
        zoom=zoom,
        time="PT3H",
        chunks={
            "cell": 12 * 4 ** (zoom - 3),
            "time": 1,
            "level_full": 90,
            "level_half": 91,
        },
    ).to_dask()

    ds_rho = cat.ngc3028(
        zoom=zoom_coarse,
        time="PT3H",
        chunks={
            "cell": 12 * (4**zoom_coarse),
            "time": 1,
            "level_full": 15,
            "level_half": 15,
        },
    ).to_dask()

    ds = ds.sel(time=date)
    ds_w = ds_w.sel(time=date)
    ds_rho = ds_rho.sel(time=date)
    
    # Interpolate w to full levels, and save to disk in a temporary location.
    print("Interpolating w to full levels.")
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
    # w_full = w_full.result()
    client.run(trim_memory)
    # w_full = xr.open_zarr(tmp_loc)
    ds_w.close()
    print("Done interpolating w.")

    # Calculate density
    density = (ds_rho.pfull / (R_s * ds_rho.ta)).to_dataset(name="rho")

    # Hash for ease of calling:
    da = {
        "w": w_full["wa_phy"],
        "u": ds["ua"],
        "v": ds["va"],
        "rho": density.drop_vars("zg"),
    }

    client.run(trim_memory)

    # Compute Products.
    da["uw"] = da["u"] * da["w"]
    da["vw"] = da["v"] * da["w"]

    # Coarsen and multiply density and save.
    for var, newvar in zip(["uw", "vw"], ["MFx", "MFy"]):
        tmp = ud_grade_xr(da[var], nside_coarse)
        tmp = dask.compute(tmp)[0]
        tmp = tmp.rename({"out_cell": "cell"})
        tmp = dask.compute(tmp * da["rho"])[0]
        tmp = tmp.rename({"rho": newvar})
        save_coarse(newvar, tmp, coarse_res, date_slice)
        print(f"Saved {newvar}.")
        del tmp
        client.run(trim_memory)

    # Now, we can delete the temporary interpolated w file from disk.
    rmtree(tmp_loc)


if __name__ == "__main__":
    main()
