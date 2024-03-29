from compute_fluxes_hp import *

# Allocate memory for .zarr/ files
# Need to know times, get these from original file or MF file

# Get var
nextgems_dir = "/work/bm1233/icon_for_ml/spherical/nextgems3/"
ds = xr.open_zarr(f'{nextgems_dir}/res51km_MFx_trunc71.zarr/')

times=ds.time.values


for v in ['Dx','Dy']:
    t = '71'
    filename=f"/work/bm1233/icon_for_ml/spherical/nextgems3/res51km_{v}_trunc{t}.zarr"
    xr.Dataset({
        v: (("time", "level_full", "ring_cell"), dask.array.empty((16080, 90, 196608), chunks=(1, 90, 196608), dtype="<f4")),
        }, 
        coords={
        "time": (("time",), times),
        "level_full": (("level_full",), np.arange(90)),
        "ring_cell" :  (("ring_cell",), np.arange(196608))
        }).to_zarr(filename, compute = False)

