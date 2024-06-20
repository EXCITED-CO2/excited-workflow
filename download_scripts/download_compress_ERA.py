"""Script to download and compress (lossy!) ERA5 data."""
from itertools import product
from pathlib import Path

import h5py
import hdf5plugin
import cdsapi
import xarray as xr

from pathos.threading import ThreadPool as Pool

output_dir = Path("/data/volume_2/hourly_global_era5")

var_names = {
    "2m_temperature": "t2m", # K
    "2m_dewpoint_temperature": "d2m", # K
    "surface_net_solar_radiation": "ssr", # J m**-2
    "surface_net_thermal_radiation": "str", # J m**-2
    "mean_surface_sensible_heat_flux": "msshf",  # W m**-2
    "mean_surface_latent_heat_flux": "mslhf",  # W m**-2
    "surface_pressure": "sp",  # Pa
    "total_precipitation": "tp",  # m
}

compression = {
    "t2m": {**hdf5plugin.SZ3(absolute=0.1)}, # 0.1 K
    "d2m": {**hdf5plugin.SZ3(absolute=0.1)}, # 0.1 K
    "ssr": {**hdf5plugin.SZ3(absolute=1e3)},  # 1000 J/m2
    "str": {**hdf5plugin.SZ(absolute=250)},  # 250 J/m2 (sz) sz3 bugs out
    "msshf": {**hdf5plugin.SZ3(absolute=1)}, # 1 W/m2
    "mslhf": {**hdf5plugin.SZ3(absolute=1)}, # 1 W/m2
    "sp": {**hdf5plugin.SZ3(absolute=1)}, # 1 Pa
    "tp": {**hdf5plugin.SZ3(absolute=1e-6)}, # 0.01 mm
}

variables = [var for var in var_names]
years = [str(y) for y in range(2000,2020)]
months = [f"{m:02}" for m in range(1,13)]
days = [f"{d:02}" for d in range(32)]
hours = [f"{hr:02}:00" for hr in range(0,24)]

c = cdsapi.Client()

def get_data(file, var, year, month):
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": var,
            "year": year,
            "month": month,
            "day": days,
            "time": hours,
        },
        str(file),
    )

    ds = xr.load_dataset(file)
    ds.to_netcdf(
        file,
        engine="h5netcdf",
        encoding=compression,
    )

files = []
itervars = []
iteryears = []
itermonths = []

for var, year, month in product(variables, years, months):
    file = output_dir / f"{var}_{year}-{month:02}.nc"
    
    if not file.exists():
        files.append(file)
        itervars.append(var)
        iteryears.append(year)
        itermonths.append(month)
    else:
        print(f"File {file.name} already exists, skipping...")

pool = Pool(nodes=4)
pool.map(get_data, files, itervars, iteryears, itermonths)
