"""
Compute monthly means.
"""
import argparse
import numpy as np
import xarray as xr
import healpy as hp
from dask.distributed import Client
from compute_fluxes_hp import trim_memory


def c_dates2slice(ds, start_date, end_date, inds=False):
    """
    This function maps two dates to the correct slices of ds.time.values.
    """
    start_slice = np.where(ds.get_index("time") == f"{start_date} 00:00:00")[0][0]
    end_slice = np.where(ds.get_index("time") == f"{end_date} 21:00:00")[0][0] + 1
    if inds:
        return start_slice, end_slice
    return slice(start_slice, end_slice)


def main():
    """
    Compute monthly means.
    """
    parser = argparse.ArgumentParser(description="Compute latitude averages")

    ## Arguments that define the model/data
    parser.add_argument("--var", metavar="var", type=str, default="u")
    parser.add_argument("--trunc", metavar="var", type=int, default=71)
    parser.add_argument("--timeframe", metavar="var", type=str, default="DJF")
    parser.add_argument("--timechunk", metavar="var", type=int, default=24)

    ## Set-up args
    args = parser.parse_args()
    var = args.var
    l_max = args.trunc
    tf = args.timeframe
    tc = args.timechunk

    # var='u'; l_max=214;

    # Open dask client
    client = Client(processes=False, n_workers=16, threads_per_worker=1)

    # Retrieve latitudes
    ds = xr.open_zarr(
        f"/work/bm1233/icon_for_ml/spherical/nextgems3/res51km_{var}_trunc{l_max}.zarr"
    )
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

    # Get time slices
    var_avg = np.zeros((73, 90))
    for year in range(0, 5):
        # Load ds.
        ds = xr.open_zarr(
            f"/work/bm1233/icon_for_ml/spherical/nextgems3/res51km_{var}_trunc{l_max}.zarr"
        )

        client.run(trim_memory)
        print(2020 + year)
        if tf == "annual":
            start_ind, end_ind = c_dates2slice(
                ds, f"{2020+year}-02-01", f"{2020+year+1}-01-31", inds=True
            )
        elif tf == "winter":
            start_ind, end_ind = c_dates2slice(
                ds, f"{2020+year}-12-01", f"{2020+year+1}-02-28", inds=True
            )
        elif tf == "summer":
            start_ind, end_ind = c_dates2slice(
                ds, f"{2020+year}-06-01", f"{2020+year}-08-31", inds=True
            )
        ds = ds.isel(time=np.arange(start_ind, end_ind))
        ds = ds.chunk(chunks={"time": tc, "level_full": 1})

        ttotal = end_ind - start_ind
        tmax = ttotal // tc
        for ll, la in enumerate(new_lat):
            print(ll, end=":")
            for t in range(tmax):
                print(t, end=", ")
                client.run(trim_memory)
                var_avg[ll, :] += (
                    ds[var]
                    .isel(
                        time=np.arange(t * tc, (t + 1) * tc), ring_cell=new_lat_inds[la]
                    )
                    .mean(dim=["ring_cell"])
                    .sum(dim=["time"])
                    .values
                    / ttotal
                )
            if ttotal % tc != 0:
                var_avg[ll, :] += (
                    ds[var]
                    .isel(
                        time=np.arange((t + 1) * tc, ttotal), ring_cell=new_lat_inds[la]
                    )
                    .mean(dim=["ring_cell"])
                    .sum(dim=["time"])
                    .values
                    / ttotal
                )

    var_avg /= 5

    filename = f"/work/bm1233/icon_for_ml/spherical/nextgems3/{tf}_{var}_{l_max}.npy"
    with open(filename, "wb") as f:
        np.save(f, var_avg)


if __name__ == "__main__":
    main()
