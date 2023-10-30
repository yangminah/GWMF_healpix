"""
Retaper from lmax=214 to lmax=71
"""
import numpy as np
import xarray as xr
import healpy as hp
import intake
from dask.distributed import Client
from compute_fluxes_hp import (
        compute_taper_coeffs,
        c_date2slice,
        trim_memory,
        total_mn,
        alm2map_xr,
        save_coarse,
        flat2wavenumber
        )
from compute_power import wavenumber2flat

def get_task_id_dict():
    """
    Get task_id_dict that corresponds to 1 task = 1 MONTH!
    """
    task_id_dict = {}
    count = 0
    for year in [2020, 2021, 2022, 2023, 2024, 2025]:
        for mon in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            if year == 2025 and mon > 5:
                break
            count += 1
            date = f"{year}-{mon}"
            task_id_dict[count] = [date, year, mon]
    return task_id_dict

def main():
    """
    Retaper from lmax=214 to lmax=71.
    """
    client = Client(processes=False, n_workers=4, threads_per_worker=4)

    zoom = 10
    cat = intake.open_catalog(
        "/work/bm1235/k203123/NextGEMS_Cycle3.git/experiments/ngc3028/outdata/ngc3028.yaml"
    )
    ds = cat.ngc3028(
        zoom=zoom,
        time="PT3H",
        chunks={"cell": 12 * 4**zoom, "time": 1, "level_full": 15, "level_half": 15},
    ).to_dask()

    # Compute taper coefficients.
    l_max = 71
    # 214 is for 200km filter; 71 for 700km filter.
    nside_coarse = 128  # coarse grained res: nside 64: 100 km, nside 128: 50 km
    zoom = 10
    cat = intake.open_catalog(
        "/work/bm1235/k203123/NextGEMS_Cycle3.git/experiments/ngc3028/outdata/ngc3028.yaml"
    )
    ds = cat.ngc3028(
        zoom=zoom,
        time="PT3H",
        chunks={"cell": 12 * 4**zoom, "time": 1, "level_full": 15, "level_half": 15},
    ).to_dask()

    # Compute taper coefficients.
    l_max = 71
    # 214 is for 200km filter; 71 for 700km filter.
    nside_coarse = 128  # coarse grained res: nside 64: 100 km, nside 128: 50 km
    zoom = 10
    coarse_res = round(
        hp.nside2resol(nside_coarse, arcmin=True) / 60 * (2 * np.pi * 6371) / 360
    )
    taper_coeffs = compute_taper_coeffs(l_max)
    nsp = taper_coeffs.shape[0]

    for task_id in range(34,35):  ###(1,35) TEMP NEED TO RUN 34
        # task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
        task_id_dict = get_task_id_dict()
        date, year, mon = task_id_dict[task_id]
        date_slice = c_date2slice(ds, year, mon)

        # Load tapered variables.
        client.run(trim_memory)
        vecm, vecl = flat2wavenumber(np.arange(total_mn(l_max)), l_max)
        ind_flat = wavenumber2flat(vecm, vecl, 214)
        taper_dict = {}
        for v in ["u", "v", "rho", "w", "uw", "vw"]:
            # v='u'
            print(f"TAPERING {v}!")
            trunc = xr.open_zarr(
                f"/work/bm1233/icon_for_ml/spherical/nextgems3/tapered_{v}_trunc214.zarr"
            )
            tapered = trunc.sel(time=date).isel(nsp=slice(0, nsp))
            tapered[v] = xr.DataArray(
                (trunc[v].sel(time=date).isel(nsp=ind_flat) * taper_coeffs).values,
                coords={
                    "level_full": trunc["level_full"].values,
                    "nsp": np.arange(2628),
                    "time": trunc["time"].sel(time=date).values,
                },
                dims=["time", "level_full", "nsp"],
            )
            taper_dict[v] = tapered

            # Save intermediate step tapered data in spectral space
            filename = f"/work/bm1233/icon_for_ml/spherical/nextgems3/tapered_{v}_trunc{l_max}.zarr"
            lev = tapered["level_full"].shape[0]
            tapered.to_zarr(
                filename,
                mode="r+",
                region={
                    "time": date_slice,
                    "nsp": slice(0, nsp),
                    "level_full": slice(0, lev),
                },
            )
            client.run(trim_memory)

        # And finally, compute fluxes in a memory efficient way.
        print("COMPUTING FLUXES!")
        tu = alm2map_xr(taper_dict["u"], nside_coarse).real
        tv = alm2map_xr(taper_dict["v"], nside_coarse).real
        tw = alm2map_xr(taper_dict["w"], nside_coarse).real
        save_coarse("u", tu, coarse_res, l_max, date_slice)
        save_coarse("v", tv, coarse_res, l_max, date_slice)
        save_coarse("w", tw, coarse_res, l_max, date_slice)

        tmpu = -tu["u"] * tw["w"]
        tmpv = -tv["v"] * tw["w"]
        del tu
        del tv
        del tw

        tuw = alm2map_xr(taper_dict["uw"], nside_coarse).real
        save_coarse("uw", tuw, coarse_res, l_max, date_slice)
        tmpu += tuw["uw"]
        del tuw

        tvw = alm2map_xr(taper_dict["vw"], nside_coarse).real
        save_coarse("vw", tvw, coarse_res, l_max, date_slice)
        tmpv += tvw["vw"]
        del tvw

        trho = alm2map_xr(taper_dict["rho"], nside_coarse).real
        save_coarse("rho", trho, coarse_res, l_max, date_slice)

        tmpu *= trho["rho"]
        tmpv *= trho["rho"]
        del trho

        save_coarse("MFx", tmpu, coarse_res, l_max, date_slice)
        save_coarse("MFy", tmpv, coarse_res, l_max, date_slice)
        print(f"{date} completed!")


if __name__ == "__main__":
    main()
