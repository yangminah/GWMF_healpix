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


def get_nest(ds):
    """
    True if dataset is in nested order
    """
    return ds.crs.healpix_order == "nest"


def get_nside(ds):
    """
    Returns nside for a healpix dataset
    """
    return ds.crs.healpix_nside


def nest2ring(ds, var, nside):
    """
    Change order from nested to ring
    """
    ring_cells = hp.pixelfunc.ring2nest(nside, ds.cell)
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        ds_ring = (
            ds[var]
            .isel(cell=ring_cells)
            .assign_coords(ring_cell=("cell", np.arange(len(ds.cell))))
            .swap_dims({"cell": "ring_cell"})
        )
    return ds_ring


def map2alm_xr(da, cell_dim="cell", trunc=None):
    """
    Compute spherical harmonics coefficients
    trunc: truncation can be chosen by setting l_max, default: l_max = 3*nside-1
    """
    if trunc is None:
        nside = (da.dims[cell_dim] / 12) ** 0.5
        trunc = 3 * nside - 1
    nsp = int((trunc + 1) * (trunc + 2) / 2)
    return xr.apply_ufunc(
        hp.map2alm,
        da,
        input_core_dims=[(cell_dim,)],
        output_core_dims=[("nsp",)],
        kwargs={"lmax": trunc},
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"nsp": nsp}},
        output_dtypes=["<c16"],
    )


def alm2map_xr(da, nside):
    """
    Transformation back to real space
    """
    ring_cell_num = hp.nside2npix(nside)
    return xr.apply_ufunc(
        hp.alm2map,
        da,
        input_core_dims=[("nsp",)],
        output_core_dims=[("ring_cell",)],
        kwargs={"nside": nside},
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"ring_cell": ring_cell_num}},
        output_dtypes=["<c16"],
    )


def get_sphharm_file(da, trunc):
    """
    Get spherical harmonics.
    """
    sphharm = map2alm_xr(da, "ring_cell", trunc=trunc)
    sphharm.load()
    return sphharm


def total_mn(lmax):
    """
    lmax : truncation => 0, ..., lmax zonal wave numbers
    """
    lmax += 1
    return (lmax**2 + lmax) // 2


def flat2wavenumber(n, lmax):
    """
    Inputs
    n : index of returned set of spherical harmonic coefficients when computed with truncation lmax
    lmax: truncation
    ----
    Outputs:
    m : zonal wave number
    l : order of polynomial
    ---
    Details: let T stand for lmax.

    l=T || [0,T]|T+1+(T-1) [1,T]|(T+1)+(T)+(T-2) [2,T]| . . . |(T+2)+(T)+(T-1)+...+1 [T,T]
        ||      |               | 
        ||      |               | 
        ||      |               |
        ||      |T+1+(0)   [1,1]|(T+1)+(T)+(0)   [2,2]|
    l=0 || [0,0]|
    =================================================================================
    m=  ||  0   | 1             | 2.                  | . . . | T

    """
    if np.isscalar(n):
        j = 0
        while j < lmax + 1:
            cond = n - (lmax + 1 - j)
            if cond >= 0:
                n = cond
                j += 1
            else:
                return j, j + cond + (lmax - (j - 1))
    m, l = np.zeros(len(n), dtype=int), np.zeros(len(n), dtype=int)
    for ii,nn in enumerate(n):
        m[ii], l[ii] = flat2wavenumber(nn, lmax)
    return m, l


def compute_taper_coeffs(lmax, half_width=15):
    """
    Inputs
    half_width: half of taper_width.
    lmax: truncation for spherical harmonics.
    """
    # First, compute the taper coeffs in the 1D sense.
    # We use a cubic spline to ensure 0 derivs at the beginning and end of the taper.
    kstar = lmax - half_width + 1
    width = 2 * half_width
    taper_kernel = np.zeros(lmax+1)
    for j in range(width):
        taper_kernel[kstar - half_width + j] = (
            2 * (j / width) ** 3 - 3 * (j / width) ** 2 + 1
        )
    taper_kernel[: kstar - half_width + 1 :] = 1

    # Now, populate the flattened wave numbers w/ the coefficients for l.
    ns = np.array(range(total_mn(lmax)))
    _, l = flat2wavenumber(ns, lmax)
    taper_applied = taper_kernel[l]
    return taper_applied

def save_coarse(v, da_coarse, coarse_res, l_max, date_slice):
    """
    Save coarsened variables to preallocated space on disk.
    """
    locstr="/work/bm1233/icon_for_ml/spherical/nextgems3/"
    filename_coarse = f"{locstr}res{coarse_res}km_{v}_trunc{l_max}.zarr"
    da_coarse = (
        da_coarse.to_dataset(name=v)
        if da_coarse.__class__ is not xr.core.dataset.Dataset
        else da_coarse
    )
    rc = da_coarse["ring_cell"].shape[0]
    lev = da_coarse["level_full"].shape[0]
    da_coarse.to_zarr(
        filename_coarse,
        mode="r+",
        region={
            "time": date_slice,
            "ring_cell": slice(0, rc),
            "level_full": slice(0, lev),
        },
    )


def trim_memory() -> int:
    """
    Manual memory trimming.
    """
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)


def mul(sphharm, taper_coeffs):
    """
    Multiply spherical harmonics to the taper coefficients.
    """
    return sphharm * taper_coeffs


def c_date2slice(ds, year, mon, day=False, inds=False):
    """
    lmaxhis function maps a date to the correct slices of ds.time.values.
    """
    if day is False:
        day = 30
        if mon == 2:
            if year % 4 == 0:
                day = 29
            else:
                day = 28
        elif mon in [1, 3, 5, 7, 8, 10, 12]:
            day = 31
        if year == 2020 and mon == 1:
            start_slice = np.where(
                ds.get_index("time") == f"{year}-{mon}-20 03:00:00"
            )[0][0]
        else:
            start_slice = np.where(
                ds.get_index("time") == f"{year}-{mon}-01 00:00:00"
            )[0][0]
    else:
        if year == 2020 and mon == 1 and day == 20:
            start_slice = np.where(
                ds.get_index("time") == f"{year}-{mon}-{day} 03:00:00"
            )[0][0]
        else:
            start_slice = np.where(
                ds.get_index("time") == f"{year}-{mon}-{day} 00:00:00"
            )[0][0]
    end_slice = (
        np.where(ds.get_index("time") == f"{year}-{mon}-{day} 21:00:00")[0][0]
        + 1
    )
    if inds:
        return start_slice, end_slice
    return slice(start_slice, end_slice)


def get_task_id_dict():
    """
    Get task_id_dict that corresponds to 1 task= 1 day
    """
    count = 0
    task_id_dict = {}
    for year in [2020, 2021, 2022, 2023, 2024, 2025]:
        for mon in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            if year == 2025 and mon > 5:
                break
            if mon == 2:
                end_day = 29 if year % 4 == 0 else 28
            elif mon in [1, 3, 5, 7, 8, 10, 12]:
                end_day = 31
            else:
                end_day = 30
            if year == 2020 and mon == 1:
                start_day = 20
            else:
                start_day = 1
            for day in range(start_day, end_day + 1):
                date = f"{year}-{mon}-{day}"
                count += 1
                task_id_dict[count] = [date, year, mon, day]
    return task_id_dict


def main():
    """
    Start dask client, load in the data and compute the fluxes. 
    While computing, save spherical harmonics coefficients.
    """
    client = Client(processes=False, n_workers=4, threads_per_worker=4)
    task_id_dict = get_task_id_dict()
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    date, year, mon, day = task_id_dict[task_id]
    print(f"Task ID: {task_id}. Date: {date}")
    tmp_loc = f"/work/bm1233/icon_for_ml/spherical/nextgems3/tmp/wfull_{date}.zarr"

    l_max = 214
    nside_coarse = 128  # coarse grained res: nside 64: 100 km, nside 128: 50 km
    zoom = 10
    R_s = 287  # specific gas constant
    coarse_res = round(
        hp.nside2resol(nside_coarse, arcmin=True) / 60 * (2 * np.pi * 6371) / 360
    )

    # NextGems Cycle 3 run:
    # Here we load ds for all vars except w, and ds2 for w.
    # ds2 is chunked differently to optimize computing the vertical interpolation for w.
    cat = intake.open_catalog(
        "/work/bm1235/k203123/NextGEMS_Cycle3.git/experiments/ngc3028/outdata/ngc3028.yaml"
    )
    ds = cat.ngc3028(
        zoom=zoom,
        time="PT3H",
        chunks={"cell": 12 * 4**zoom, "time": 1, "level_full": 15, "level_half": 15},
    ).to_dask()
    date_slice = c_date2slice(ds, year, mon, day)
    ds2 = cat.ngc3028(
        zoom=zoom,
        time="PT3H",
        chunks={
            "cell": 12 * 4 ** (zoom - 3),
            "time": 1,
            "level_full": 90,
            "level_half": 91,
        },
    ).to_dask()

    ds = ds.sel(time=date)
    ds2 = ds2.sel(time=date)
    nside = get_nside(ds)

    # Interpolate w to full levels, and save to disk in a temporary location.
    client.run(trim_memory)
    l_full = np.arange(0, 91)
    w_full = ds2["wa_phy"].rolling(level_half=2, min_periods=2).mean()
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
    ds2.close()

    # Calculate density
    density = (ds.pfull / (R_s * ds.ta)).to_dataset(name="rho")

    # Reorder to ring order
    da_ring_dict = {}
    da_ring_dict["w"] = nest2ring(w_full, "wa_phy", nside)
    da_ring_dict["u"] = nest2ring(ds, "ua", nside)
    da_ring_dict["v"] = nest2ring(ds, "va", nside)
    da_ring_dict["rho"] = nest2ring(density, "rho", nside)
    da_ring_dict["uw"] = (da_ring_dict["u"] * da_ring_dict["w"]).rename("uw")
    da_ring_dict["vw"] = (da_ring_dict["v"] * da_ring_dict["w"]).rename("vw")

    # Compute taper coefficients.
    taper_coeffs = compute_taper_coeffs(l_max)

    # Taper and save.
    taper_dict = {}
    client.run(trim_memory)
    for v,vv in da_ring_dict.items():
        print(f"COMPUTING {v}!")
        da_scattered = client.scatter(vv)
        sphharm = client.submit(map2alm_xr, da_scattered, "ring_cell", l_max)
        sphharm = client.submit(xr.core.dataarray.DataArray.load, sphharm)

        # Taper
        sphharm = client.submit(mul, sphharm, taper_coeffs).result()
        taper_dict[v] = sphharm

        # Save intermediate step tapered data in spectral space
        filename = f"/work/bm1233/icon_for_ml/spherical/nextgems3/tapered_{v}_trunc{l_max}.zarr"
        nsp = sphharm["nsp"].shape[0]
        lev = sphharm["level_full"].shape[0]
        sphharm.to_dataset(name=v).to_zarr(
            filename,
            mode="r+",
            region={
                "time": date_slice,
                "nsp": slice(0, nsp),
                "level_full": slice(0, lev),
            },
        )
        taper_dict[v] = sphharm
        client.run(trim_memory)

    # Now, we can delete the temporary interpolated w file from disk.
    rmtree(tmp_loc)

    # And finally, compute Fluxes in a memory efficient way (probably not necessary at this point).
    print("COMPUTING FLUXES!")
    tu = alm2map_xr(taper_dict["u"], nside_coarse).real
    tv = alm2map_xr(taper_dict["v"], nside_coarse).real
    tw = alm2map_xr(taper_dict["w"], nside_coarse).real
    save_coarse("u", tu, coarse_res, l_max, date_slice)
    save_coarse("v", tv, coarse_res, l_max, date_slice)
    save_coarse("w", tw, coarse_res, l_max, date_slice)

    tmpu = -tu * tw
    tmpv = -tv * tw
    del tu
    del tv
    del tw

    tuw = alm2map_xr(taper_dict["uw"], nside_coarse).real
    save_coarse("uw", tuw, coarse_res, l_max, date_slice)
    tmpu += tuw
    del tuw

    tvw = alm2map_xr(taper_dict["vw"], nside_coarse).real
    save_coarse("vw", tvw, coarse_res, l_max, date_slice)
    tmpv += tvw
    del tvw

    trho = alm2map_xr(taper_dict["rho"], nside_coarse).real
    save_coarse("rho", trho, coarse_res, l_max, date_slice)

    tmpu *= trho
    tmpv *= trho
    del trho

    save_coarse("MFx", tmpu, coarse_res, l_max, date_slice)
    save_coarse("MFy", tmpv, coarse_res, l_max, date_slice)
    print(f"{date} completed!")


if __name__ == "__main__":
    main()
