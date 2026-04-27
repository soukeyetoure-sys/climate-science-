import xarray as xr
import os

BASE_DIR = os.path.expanduser("~/Bureau/assigment2:climate")

files = [f"{BASE_DIR}/cdd_output/cdd_{y}.nc" for y in range(1979, 1989)]

datasets = []
for year, fpath in zip(range(1979, 1989), files):
    ds = xr.open_dataset(fpath).expand_dims(year=[year])
    datasets.append(ds)

merged = xr.concat(datasets, dim='year')
merged.to_netcdf(f"{BASE_DIR}/cdd_final.nc")
print(f"cdd_final.nc cree avec {len(merged.year)} pas de temps")
print(merged)
