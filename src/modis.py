"""Ingest MODIS data."""

from pathlib import Path
import xarray as xr
import numpy as np


def load_modis_data(path: Path) -> xr.Dataset:
    """Load and ingest modis data.

    This function loads raw MODIS data files and merge them into one dataset.
    The given path should lead to the folder including raw `.nc` files. The file
    names must contain temporal information and follow the `NIRv_global_*_yyyy_mm.nc`
    format, e.g. "NIRv_global_0.5x0.5_v0.2_2015_12.nc"

    Args:
        path: path to the directory containing raw netcdf files.
    Returns:
        An aggregated xarray dataset
    """
    list_files = list(path.glob("NIRv_global_*.nc"))
    list_datasets = []
    for file in list_files:
        dataset = xr.open_dataset(file)
        time = "-".join(file.stem.split("_")[-2:])
        dataset["time"] = np.datetime64(time)
        dataset = dataset.rename_dims({"lat": "latitude", "lon": "longitude"})
        list_datasets.append(dataset)

    dataset_merge = (
        xr.concat(list_datasets, dim="time")
        .sortby("time")
        .chunk(chunks={"time": -1, "latitude": 1, "longitude": 1})
    )
    # raw data does not contain lat/lon coordinates, we need to add them
    lats = np.arange(0, 180, 0.5) - 89.75
    lons = np.arange(0, 360, 0.5) - 179.75
    dataset = dataset_merge.assign_coords(
        latitude=("latitude", lats), longitude=("longitude", lons)
    )

    return dataset
