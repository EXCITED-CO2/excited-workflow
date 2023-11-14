"""Ingest Copernicus land cover data."""
from typing import Literal

import xarray as xr
import xarray_regrid  # noqa: F401

from excited_workflow.source_datasets.protocol import DataSource
from excited_workflow.source_datasets.protocol import get_freq_kw


class LandCover(DataSource):
    """Copernicus land cover dataset."""

    name: str = "copernicus_landcover"
    variable_names: list[str] = ["lccs_class"]

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

        ds = xr.open_mfdataset(files)

        freq_kw = get_freq_kw(freq)
        ds = ds.resample(time=freq_kw).interpolate("linear")

        if variables is not None:
            ds = ds[variables]
        if target_grid is not None:
            ds = ds.regrid.most_common(target_grid)

        return ds
