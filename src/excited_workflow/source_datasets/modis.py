"""Ingest MODIS data."""
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
import xarray_regrid  # noqa: F401

from excited_workflow.source_datasets.protocol import DataSource
from excited_workflow.source_datasets.protocol import get_freq_kw


def load_modis_data(files: list[Path]) -> xr.Dataset:
    """Load and ingest modis data.

    This function loads raw MODIS data files and merge them into one dataset.
    The given path should lead to the folder including raw `.nc` files. The file
    names must contain temporal information and follow the `NIRv_global_*_yyyy_mm.nc`
    format, e.g. "NIRv_global_0.5x0.5_v0.2_2015_12.nc"

    Args:
        files: list of paths of the raw netcdf files.

    Returns:
        An aggregated xarray dataset
    """
    list_datasets = []
    for file in files:
        dataset = xr.open_dataset(file)
        time = "-".join(file.stem.split("_")[-2:])
        dataset["time"] = np.datetime64(time)
        dataset = dataset.rename_dims({"lat": "latitude", "lon": "longitude"})
        list_datasets.append(dataset)

    dataset_merge = (
        xr.concat(list_datasets, dim="time")
        .sortby("time")
        .chunk(chunks={"time": -1, "latitude": 60, "longitude": 60})
    )
    # raw data does not contain lat/lon coordinates, we need to add them
    lats = np.arange(0, 180, 0.5) - 89.75
    lons = np.arange(0, 360, 0.5) - 179.75
    dataset = dataset_merge.assign_coords(
        latitude=("latitude", lats), longitude=("longitude", lons)
    )

    return dataset


class Modis(DataSource):
    """MODIS dataset."""

    name: str = "modis"
    variable_names: list[str] = ["NDVI", "NIRv"]

    @classmethod
    def load(
        cls,
        freq: Literal["hourly", "monthly"],
        variables: list[str] | None = None,
        target_grid: xr.Dataset | None = None,
    ) -> xr.Dataset:
        """Load variables from this data source and regrid them to the target grid.

        Args:
            freq: Desired frequency of the dataset. Either "hourly" or "monthly".
            variables: List of variable names which should be downloaded.
            target_grid: Grid to which the data should be regridded to.

        Returns:
            Prepared dataset.
        """
        cls.validate_variables(cls, variables)

        files = list(cls.get_path(cls).glob("*.nc"))
        if len(files) == 0:
            msg = f"No netCDF files found at path '{cls.get_path(cls)}'"
            raise FileNotFoundError(msg)

        ds = load_modis_data(files)

        if freq == "hourly":
            freq_kw = get_freq_kw(freq)
            ds["time"] = ds["time"] + pd.Timedelta(days=14)
            ds = ds.resample(time=freq_kw).interpolate("linear")
        elif freq == "monthly":
            pass
        else:
            get_freq_kw(cls, freq)

        if variables is not None:
            ds = ds[variables]
        if target_grid is not None:
            ds = ds.regrid.linear(target_grid)

        return ds
