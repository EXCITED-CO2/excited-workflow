"""Ingest Copernicus land cover data."""
from typing import Literal

import numpy as np
import xarray as xr
import xarray_regrid  # noqa: F401

from excited_workflow.source_datasets.protocol import DataSource
from excited_workflow.source_datasets.protocol import get_freq_kw


def _cftime_to_datetime(data: xr.DataArray) -> np.ndarray:
    """Convert cftime dataarray values to a numpy datetime format.

    Args:
        data: DataArray containing the time values, e.g. ds["time"].

    Returns:
        Numpy array with a datetime64 dtype.
    """
    return np.array([np.datetime64(el) for el in data.to_numpy()])


class LandCover(DataSource):
    """Copernicus land cover dataset."""

    name: str = "copernicus_landcover"
    variable_names: list[str] = ["lccs_class"]

    def load(
        self,
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
        self.validate_variables(variables)

        files = list(self.get_path().glob("*.nc"))
        if len(files) == 0:
            msg = f"No netCDF files found at path '{self.get_path()}'"
            raise FileNotFoundError(msg)
            
        ds = xr.open_mfdataset(files, chunks={"lat": 2000, "lon": 2000})
        # Set time to middle of bounds.
        time_coords = _cftime_to_datetime(ds["time_bounds"].mean(dim="bounds"))
        ds = ds.drop("time_bounds")
        ds["time"] = time_coords

        ds = ds[["lccs_class"]]  # Only take the class variable.
        ds = ds.sortby(["lat", "lon"])
        ds = ds.rename({"lat": "latitude", "lon": "longitude"})
        
        if variables is not None:
            ds = ds[variables]
        if target_grid is not None:
            ds = ds.isel(time=slice(0, 2))
            ds = ds.regrid.most_common(target_grid, time_dim="time",  max_mem=1e9)
        
        freq_kw = get_freq_kw(freq)
        ds = ds.resample(time=freq_kw).interpolate("nearest")
        
        return ds
