import numpy as np
import xarray as xr
import healpy as hp
import intake
import os
from compute_fluxes_hp import get_task_id_dict, c_date2slice#, save_coarse
import dask
from dask.distributed import Client
def ud_grade_xr(da, nside_coarse):
    """
    Coarsen to nside_coarse starting from ring order.
    """
    npix = hp.nside2npix(nside_coarse)
    return xr.apply_ufunc(
        hp.ud_grade,
        da,
        input_core_dims=[["ring_cell"]],
        output_core_dims=[["out_cell"]],
        kwargs={"nside_out": nside_coarse, "order_in":"RING", "order_out":"NESTED"},
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"out_cell": npix}, "allow_rechunk": True},
        output_dtypes=["<f4"],
    )
def save_coarse(v, da_coarse, coarse_res, l_max, date_slice, locstr=None):
    """
    Save coarsened variables to preallocated space on disk.
    """
    #if locstr == None:
    locstr = "/work/bm1233/icon_for_ml/spherical/nextgems3/reynolds/"
    filename_coarse = f"{locstr}res{coarse_res}km_{v}_trunc{l_max}.zarr"
    # filename_coarse = f"{locstr}res{coarse_res}km_{v}_nofilter.zarr"
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
    print("saved to", filename_coarse)
def calc_drag(task_id, var, trunc, coarse_res,nside_coarse):
    """ Calculates GW drag for single timestep and saves to .zarr/ folder"""
    task_id_dict = get_task_id_dict()
    date, year, mon, day = task_id_dict[task_id]
    print(f"Task ID: {task_id}. Date: {date}")

    # Get var
    nextgems_dir = "/work/bm1233/icon_for_ml/spherical/nextgems3/reynolds"
    ds = xr.open_zarr(f'{nextgems_dir}/res51km_{var}_trunc{trunc}.zarr/')
    # Time slice
    date_slice = c_date2slice(ds, year, mon, day)
    print("Date slice", date_slice)
    ds = ds.sel(time=date)

    # Get rho
    ds_rho = xr.open_zarr(f'{nextgems_dir}/../res51km_rho_trunc{trunc}.zarr/').sel(time=date)

    # Get z (ring-order)
    #locstr = "/work/bm1233/icon_for_ml/spherical/nextgems3/"
    #ds_z = xr.open_zarr(f'{locstr}/zg.zarr/')
    cat = intake.open_catalog("/work/bm1235/k203123/NextGEMS_Cycle3.git/experiments/ngc3028/outdata/ngc3028.yaml")
    zoom=7
    
    ds_z = cat.ngc3028(zoom=zoom, time="PT3H", 
                       chunks={"cell": 12 * (4**zoom), "time": 1, "level_full": 90, "level_half": 91,},
                      ).to_dask()
    z = ds_z['zg'].to_numpy()
    print(z.shape)
    z = np.expand_dims(z, axis=0)

    print("files opened")
    # Get levels
    levels = ds[var].level_full
    nlev = len(levels)

    # Get top and bottom indices for differencing.
    t_idx = np.arange(-1, nlev - 1)
    t_idx[0] = 0
    b_idx = np.arange(1, nlev + 1)
    b_idx[-1] = nlev - 1
    # Compute dTdz
    print("Compute ")
    MF = ds[var].to_numpy()
    rho = ud_grade_xr(ds_rho["rho"], nside_coarse).to_numpy()

    # Calc drag
    drag = MF[:, t_idx] - MF[:, b_idx]
    drag /= (z[:, t_idx] - z[:, b_idx])
    drag /= -rho
    print("Done,, saving ...")
        
    # Save into a dataset
    # Name depends on direction x or y
    direction = var[-1] 
    drag_name = f"D{direction:s}"
    drag_da = xr.DataArray(drag, coords=ds.coords)
    drag_da.attrs = {'long_name': f"GW drag in {direction} direction", 'units': 'm/s^2'}
    drag_ds = drag_da.to_dataset(name=drag_name)
    
    save_coarse(drag_name, drag_ds, coarse_res, trunc, date_slice, locstr=None) 
    print(f"Done. {task_id}")
    return drag_ds
def main():
    """
    Start dask client, load in the data and compute the fluxes.
    While computing, save spherical harmonics coefficients.
    """
    client = Client(processes=False, n_workers=4, threads_per_worker=1)
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    # task_id must be 2-1959
    # Set up
    variables = ['MFx', 'MFy']
    truncs = [71] #, 214]
    trunc = 71
    coarse_res=102; nside_coarse=64
    for var in variables:
        calc_drag(task_id, var, trunc, coarse_res, nside_coarse)


if __name__ == "__main__":
    main()
