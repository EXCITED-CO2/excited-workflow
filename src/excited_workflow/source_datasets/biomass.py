"""Global biomass dataset."""
from typing import Literal

import numpy as np
import pandas as pd
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


class Biomass(DataSource):
    """Global biomass dataset."""

    name: str = "biomass"
    variable_names: list[str] = ["biomass"]

    def load(
        self,
        freq: Literal["monthly", "hourly"] | None = None,
        variables: list[str] | None = None,
        target_grid: xr.Dataset | None = None,
    ) -> xr.Dataset:
        """Load variables from this data source and regrid them to the target grid.

        Args:
            freq: Desired frequency of the dataset. Either "monthly", "hourly", or None.
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

        freq_kw = get_freq_kw(freq)

        ds = xr.combine_by_coords(
            [xr.open_dataset(file, chunks={"lat": 240, "lon": 240, "time": 12}) for file in files],
            combine_attrs="drop_conflicts",
        )

        # Set time to middle of bounds.
        time_coords = _cftime_to_datetime(ds["time_bnds"].mean(dim="nv"))
        ds = ds.drop("time_bnds")
        ds["time"] = time_coords

        if freq_kw is None:
            freq_kw = "1YS"  # Dataset is yearly by default

        # Extend range to cover fluxnet data
        time_range = pd.date_range(start="1995-01-01", end="2022-01-01", freq=freq_kw)

        ds = ds.interp(time=time_range, method="linear")
        ds = ds.interpolate_na(
            dim="time",
            method="nearest",
            max_gap=np.timedelta64(5, "Y"),
            fill_value="extrapolate",
        )

        ds = ds.rename({"lat": "latitude", "lon": "longitude"})

        if variables is not None:
            ds = ds[variables]
        if target_grid is not None:
            ds = ds.regrid.linear(target_grid)

        return ds
