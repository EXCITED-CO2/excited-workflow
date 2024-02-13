"""Ingest SPEI data."""

from typing import Literal

import xarray as xr
import xarray_regrid  # noqa: F401

from excited_workflow import utils
from excited_workflow.source_datasets.protocol import DataSource
from excited_workflow.source_datasets.protocol import get_freq_kw


class Spei(DataSource):
    """Global Standardised Precipitation Evapotranspiration Index dataset."""

    name: str = "spei"
    variable_names: list[str] = ["spei"]

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

        ds = xr.combine_by_coords(
            [xr.open_dataset(file, chunks={"time": 24}) for file in files],
            combine_attrs="drop_conflicts",
        )
        assert isinstance(ds, xr.Dataset)  # combine_by_coords misses typing overload
        ds = utils.convert_timestamps(ds)
        ds = ds.sel(time=slice("1990-01-01T00:00:00", None))  # Drop pre-90s data.
        ds = ds.rename({"lat": "latitude", "lon": "longitude"})

        if freq == "monthly":
            ds = ds.resample(time="1MS").mean()
        elif freq == "hourly":
            freq_kw = get_freq_kw(freq)
            ds = ds.resample(time=freq_kw).interpolate("linear")
        else:
            pass

        if variables is not None:
            ds = ds[variables]
        if target_grid is not None:
            ds = ds.regrid.linear(target_grid)

        return ds  # type: ignore
