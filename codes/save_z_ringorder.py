import numpy as np
import xarray as xr
import healpy as hp
import intake
import os
from compute_fluxes_hp import nest2ring, save_coarse
import dask
from dask.distributed import Client

def z_nest2ring():
    # Get z from orig file
    zoom=7
    basedir="/work/bm1233/icon_for_ml/spherical/nextgems3/nofilter/"
    cat = intake.open_catalog("/work/bm1235/k203123/NextGEMS_Cycle3.git/experiments/ngc3028/outdata/ngc3028.yaml")
    ds = cat.ngc3028(zoom=zoom,time="PT3H",chunks={"cell": 12 * 4**zoom, "time": 1, "level_full": 90, "level_half": 91},).to_dask()
    print(ds.crs.healpix_order)     # nest
    nside = ds.crs.healpix_nside
    print(nside)                    # should be 128 for 51 km resolution
    # Swap zg from coordinate to variable
    ds = ds.reset_coords("zg")
    # Convert to ring order
    ds_ring = nest2ring(ds, "zg", nside)
    print(ds_ring)
    # Save
    locstr = "/work/bm1233/icon_for_ml/spherical/nextgems3/"
    filename_ring = f"{locstr}zg.zarr"
    ds_ring = ds_ring.to_dataset(name="zg")
    ds_ring.to_zarr(filename_ring )


def main():
    """
    Start dask client, load in the data and compute the fluxes.
    While computing, save spherical harmonics coefficients.
    """
    client = Client(processes=False, n_workers=1, threads_per_worker=1)
    z_nest2ring()

if __name__ == "__main__":
    main()
