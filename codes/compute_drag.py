import numpy as np
import xarray as xr
import healpy as hp
import intake
import os
from compute_fluxes_hp import get_task_id_dict, c_date2slice, save_coarse
import dask
from dask.distributed import Client

def calc_drag(task_id, var, trunc):
    """ Calculates GW drag for single timestep and saves to .zarr/ folder"""
    task_id_dict = get_task_id_dict()
    date, year, mon, day = task_id_dict[task_id]
    print(f"Task ID: {task_id}. Date: {date}")

    # Get var
    nextgems_dir = "/work/bm1233/icon_for_ml/spherical/nextgems3/"
    ds = xr.open_zarr(f'{nextgems_dir}/res51km_{var}_trunc{trunc}.zarr/')
    # Time slice
    date_slice = c_date2slice(ds, year, mon, day)
    ds = ds.sel(time=date)

    # Get rho
    ds_rho = xr.open_zarr(f'{nextgems_dir}/res51km_rho_trunc{trunc}.zarr/').sel(time=date)

    # Get z (ring-order)
    locstr = "/work/bm1233/icon_for_ml/spherical/nextgems3/"
    ds_z = xr.open_zarr(f'{nextgems_dir}/zg.zarr/')
    z = ds_z['zg'].to_numpy()
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
    MFx = ds[var].to_numpy()
    rho = ds_rho["rho"].to_numpy()

    # Calc drag
    drag = MFx[:, t_idx] - MFx[:, b_idx]
    drag /= (z[:, t_idx] - z[:, b_idx])
    drag /= rho
    print("Done,, saving ...")
        
    # Save into a dataset
    # Name depends on direction x or y
    direction = var[-1] 
    drag_name = f"D{direction}"
    drag_da = xr.DataArray(drag, coords=ds.coords)
    drag_da.attrs = {'long_name': f"GW drag in {direction} direction", 'units': 'm/s^2'}
    drag_ds = drag_da.to_dataset(name=drag_name)

    save_coarse(drag_name, drag_ds, 51, trunc, date_slice, locstr=None) 
    print(f"Done. {task_id}")

def main():
    """
    Start dask client, load in the data and compute the fluxes.
    While computing, save spherical harmonics coefficients.
    """
    client = Client(processes=False, n_workers=1, threads_per_worker=1)
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    # task_id must be 2-1959
    # Set up
    variables = ['MFx', 'MFy']
    truncs = [71] #, 214]

    var = "MFx"
    trunc = 71

    calc_drag(task_id, var, trunc)


if __name__ == "__main__":
    main()
