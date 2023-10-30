import numpy as np
import xarray as xr
import healpy as hp
import intake
import os
import sys
import time

from compute_fluxes_hp import *
import dask
from dask.distributed import Client

import ctypes
from datetime import datetime
from shutil import rmtree
from compute_fluxes_hp import *

# from plot import wavenumber2flat

# import cartopy.crs as ccrs
# import cartopy.feature as cf
# import cmocean
import matplotlib.pylab as plt
import matplotlib as mpl
import argparse


def wavenumber2flat(m, l, T):
    if np.isscalar(m) == True:
        n = 0
        for i in range(m):
            n += T + 1 - i
        n += l - m
        return n
    else:
        n = np.zeros(len(m), dtype=int)
        for ii in range(len(m)):
            n[ii] = wavenumber2flat(m[ii], l[ii], T)
        return n


def main():
    parser = argparse.ArgumentParser(description="Compute power")

    ## Arguments that define the model/data
    parser.add_argument("--var", metavar="var", type=str, default="u")
    parser.add_argument("--trunc", metavar="var", type=int, default=71)

    ## Set-up args
    args = parser.parse_args()
    var = args.var
    l_max = args.trunc

    # Job array task id
    task_id = 5 * int(
        os.getenv("SLURM_ARRAY_TASK_ID")
    )  # should range from 0 to 10 inclusive.

    # Open dask client
    client = Client(n_workers=4)

    # Open OG dataset to get z levels.
    cat = intake.open_catalog(
        "/work/bm1235/k203123/NextGEMS_Cycle3.git/experiments/ngc3028/outdata/ngc3028.yaml"
    )
    zoom = 3
    ds_OG = cat.ngc3028(
        zoom=zoom, time="PT3H", chunks={"cell": 12 * 4**zoom, "level_full": 5}
    ).to_dask()

    # Get indices of levels we're interested in.
    mean_heights = np.mean(ds_OG.zg.values, axis=1)
    height_inds = []
    N_height = 6
    for z in [6, 8, 12, 16, 20, 24]:
        height_inds.append((np.abs(mean_heights - z * 1000)).argmin())

    # Create an array of months written out in string form.
    N_month = 60
    months = {}
    year = 2020
    month = 2
    months[0] = f"{year}-{month:02d}"
    for month_ind in range(1, N_month):
        month += 1
        if month == 13:
            year += 1
            month = 1
        months[month_ind] = f"{year}-{month:02d}"

    # Load truncated coeff ds & rechunk.
    ds = xr.open_zarr(
        f"/work/bm1233/icon_for_ml/spherical/nextgems3/tapered_{var}_trunc{l_max}.zarr"
    )

    # Gather indices of nsp that correspond to each l.
    # i.e. \{ (0,l), (1,l), (2,l), ..., (l,l) \}
    flat_inds = {}
    for l in range(1, l_max + 1):
        flat_inds[l] = wavenumber2flat(
            np.arange(l + 1), l * np.ones(l + 1).astype(int), l_max
        )
    if l_max == 71:
        lstart0 = 0
        lstart1 = 1
        ds = ds.chunk(chunks={"nsp": 2628, "time": 32, "level_full": 1})
    elif l_max == 214:
        lstart0 = 43
        lstart1 = 43
        ds = ds.chunk(chunks={"nsp": ds.nsp.shape[0], "time": 4, "level_full": 1})

    # Loop over months and levels
    # for month_ind in range(N_month):
    for month_ind in range(task_id, task_id + 5):  # (5x12)
        date = months[month_ind]
        print(date)
        filename = f"/work/bm1233/icon_for_ml/spherical/nextgems3/power/{var}/{l_max}/{date}.npy"
        Cl = np.zeros((N_height, l_max + 1))
        if l_max == 214:
            with open(
                f"/work/bm1233/icon_for_ml/spherical/nextgems3/power/{var}/71/{date}.npy",
                "rb",
            ) as f:
                Cl[:, :lstart1] = np.load(f)[:, :lstart1]
        for level_ind in range(N_height):
            height_ind = height_inds[level_ind]
            client.run(trim_memory)
            # Retrieve dataset at correct month & level
            coeffs_vec = ds[var].sel(time=date).isel(level_full=height_ind)

            # Do m=0, l=lstart0, ..., l_max
            Cl[level_ind, lstart0:] = np.mean(
                np.abs(coeffs_vec.isel(nsp=np.arange(lstart0, l_max + 1)).values) ** 2,
                axis=0,
            )

            # Get the remaining modes powers (for each l, m = 1, .., l; and double the magnitudes for m=-l,...,-1)
            for l in range(lstart1, l_max + 1):
                client.run(trim_memory)
                Cl[level_ind, l] += 2 * np.mean(
                    np.sum(
                        np.abs(coeffs_vec.isel(nsp=flat_inds[l][1:]).values) ** 2,
                        axis=1,
                    ),
                    axis=0,
                )
        with open(filename, "wb") as f:
            np.save(f, Cl)
        print(f"Saved to {filename}")


if __name__ == "__main__":
    main()
