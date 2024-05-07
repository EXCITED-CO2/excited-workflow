"""Script to download and compress (lossy!) ERA5 data."""
from itertools import product
from pathlib import Path

import cdsapi
import xarray as xr
from enstools.io import write


output_dir = Path("/data/EXCITED/test")

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
    "t2m": "lossy,sz3,abs,0.1", # 0.1 K
    "d2m": "lossy,sz3,abs,0.1", # 0.1 K
    "ssr": "lossy,sz3,abs,0.00028",  # 1 J/hr/m2
    "str": "lossy,sz3,abs,0.00028",  # 1 J/hr/m2
    "msshf": "lossy,sz3,abs,1", # 1 W/m2
    "mslhf": "lossy,sz3,abs,1", # 1 W/m2
    "tp": "lossy,sz3,abs,1", # 1 Pa
    "sp": "lossy,sz3,abs,1e-6", # 0.01 mm
}

variables = [var for var in var_names]
years = [str(y) for y in range(2000,2020)]
months = [str(m) for m in range(1,13)]
days = [f"{d:02}" for d in range(32)]
hours = [f"{hr:02}:00" for hr in range(0,24)]

c = cdsapi.Client()

for var, year, month in product(variables, years, months):
    file = output_dir / f"{var}_{year}-{month}.nc"
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
    write(
        ds,
        file_path=file,
        compression=compression,
    )
