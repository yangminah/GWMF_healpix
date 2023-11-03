"""
Compute annual cycle.
"""
import sys
import argparse
from dask.distributed import Client
import numpy as np
import xarray as xr
import healpy as hp

# from compute_timeavg import c_dates2slice
# from compute_fluxes_hp import *
sys.path.append("../")


def main():
    """
    Compute annual cycle.
    """
    parser = argparse.ArgumentParser(description="Compute latitude averages")

    ## Arguments that define the model/data
    parser.add_argument("--var", metavar="var", type=str, default="u")
    parser.add_argument("--trunc", metavar="var", type=int, default=71)

    ## Set-up args
    args = parser.parse_args()
    var = args.var
    l_max = args.trunc
    # var='u'; l_max=214;

    # Open dask client
    client = Client(processes=False, n_workers=32, threads_per_worker=1)

    # Load truncated coeff ds & rechunk.
    ds = xr.open_zarr(
        f"/work/bm1233/icon_for_ml/spherical/nextgems3/res51km_{var}_trunc{l_max}.zarr"
    )

    # Get dates starting from February.
    date_list = []
    month = 1
    day = 0
    year = 2020
    for tt in range(0, 365):
        day += 1
        day = 1 if day == 32 else day
        day = 1 if (day == 31 and month in [2, 4, 6, 9, 11]) else day
        day = 1 if day == 29 and month == 2 else day
        month = month + 1 if day == 1 else month
        month = 1 if month == 13 else month
        # year = year +1 if (month == 1 and day == 1) else year
        date_list.append(f"{month:02d}-{day:02d}")

    # Retrieve latitudes
    npix = ds[var].shape[-1]
    nside = hp.pixelfunc.npix2nside(npix)
    pix_ind = np.arange(npix)
    _, lat_pix = hp.pixelfunc.pix2ang(nside, pix_ind, nest=False, lonlat=True)
    lat = np.unique(lat_pix)
    lat_inds = {}
    new_lat = []
    new_lat_inds = {}
    for la in lat:
        lat_inds[la] = pix_ind[lat_pix == la]
    for las in np.arange(0, 511, 7):
        new_lat.append(np.mean(lat[las : las + 7]))
        new_lat_inds[new_lat[-1]] = np.concatenate(
            (
                lat_inds[lat[las]],
                lat_inds[lat[las + 1]],
                lat_inds[lat[las + 2]],
                lat_inds[lat[las + 3]],
                lat_inds[lat[las + 4]],
                lat_inds[lat[las + 5]],
                lat_inds[lat[las + 6]],
            )
        )
    new_lat = np.array(new_lat)

    # Compute indices of interesting latitudes.
    interesting_lats = [-45, 0, 45]
    interesting_lat_inds = []
    for la in interesting_lats:
        interesting_lat_inds.append(np.abs(new_lat - la).argmin())

    # Compute annual cycle for interesting latitudes.
    year = 2020
    old = 0
    u_latavg = np.zeros((365, len(interesting_lats), 90))
    for tt in range(0, 365 * 5):
        if old != ((tt % 365) // 30):
            old = (tt % 365) // 30
            print(old)
        year = year + 1 if date_list[tt % 365] == "01-01" else year
        new_date = f"{year}-{date_list[tt%365]}"
        # print(new_date,end=', ')
        for ll, lat in enumerate(interesting_lats):
            la = new_lat[lat]
            # print(la,end=', ')
            u_latavg[tt % 365, ll, :] += (
                ds[var]
                .sel(time=new_date)
                .isel(ring_cell=new_lat_inds[la])
                .mean(dim=["time", "ring_cell"])
                .values
            )
    u_latavg /= 5

    filename = f"/work/bm1233/icon_for_ml/spherical/nextgems3/AC_{var}_{l_max}.npy"
    with open(filename, "wb") as f:
        np.save(f, u_latavg)


if __name__ == "__main__":
    main()
