"""Ingest SPEI data."""

from pathlib import Path
from typing import Literal

from excited_workflow import utils

import numpy as np
import pandas as pd
import xarray as xr
import xarray_regrid  # noqa: F401

from excited_workflow.protocol import DataSource


class Spei(DataSource):
    """Global Standardised Precipitation Evapotranspiration Index dataset."""

    name: str = "spei"
    variable_names: list[str] = ["spei"]

    @classmethod
    def load(
        cls,
        freq: Literal["1H", "1M"],
        variables: list[str] | None = None,
        target_grid: xr.Dataset | None = None,
    ) -> xr.Dataset:
        """Load variables from this data source and regrid them to the target grid.

        Args:
            freq: Desired frequency of the dataset. Either "1H" for hourly, or "1M" for
                monthly.
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

        ds = xr.open_mfdataset(files)
        ds = utils.convert_timestamps(ds)
        ds = ds.sel(time=slice("1990-01-01T00:00:00", None))  # Drop pre-90s data.
        ds = ds.rename({"lat": "latitude", "lon": "longitude"})

        if freq == "monthly":
            ds = ds.resample(time="1MS").mean()
        else:
            freq_kw = cls.get_freq_kw(cls, freq)
            ds = ds.resample(time=freq_kw).interpolate("linear")

        if variables is not None:
            ds = ds[variables]
        if target_grid is not None:
            ds = ds.regrid.linear(target_grid)

        return ds
