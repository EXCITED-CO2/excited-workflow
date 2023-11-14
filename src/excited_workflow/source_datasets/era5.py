"""Ingest ERA5 data."""
from typing import Literal

import xarray as xr
import xarray_regrid  # noqa: F401

from excited_workflow.source_datasets.protocol import DataSource


def shift_era5_longitude(dataset: xr.Dataset) -> xr.Dataset:
    """Shift the longitude of a dataset from [0, 360] to [-180, 180].

    Args:
        dataset: Dataset to shift the longitude value of.

    Returns:
        Dataset with the longitude values shifted and resorted.
    """
    dataset["longitude"] = xr.where(
        dataset["longitude"] > 180, dataset["longitude"] - 360, dataset["longitude"]
    )
    return dataset.sortby(["latitude", "longitude"])


class ERA5DataSource(DataSource):
    """ERA5 data source superclass."""
    name: str
    variable_names: list[str]
    data_freq: Literal["hourly"] | Literal["monthly"]

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
        if freq != self.data_freq:
            msg = (
                "For ERA5 data use the correct dataset source for monthly/hourly "
                "datasets."
            )
            raise ValueError(msg)

        self.validate_variables(variables)

        files = list(self.get_path().glob("*.nc"))
        if len(files) == 0:
            msg = f"No netCDF files found at path '{self.get_path()}'"
            raise FileNotFoundError(msg)

        ds = xr.open_mfdataset(files)
        ds = shift_era5_longitude(ds)

        if variables is not None:
            ds = ds[variables]
        if target_grid is not None:
            ds = ds.regrid.linear(target_grid)

        return ds


class ERA5Hourly(ERA5DataSource):
    """Hourly ERA5 dataset."""

    name: str = "era5_hourly"
    variable_names: list[str] = [
        "d2m",
        "mslhf",
        "msshf",
        "sp",
        "ssr",
        "str",
        "t2m",
        "tp",
    ]
    data_freq = "hourly"


class ERA5Monthly(ERA5DataSource):
    """Monthly ERA5 dataset."""

    name: str = "era5_monthly"
    variable_names: list[str] = [
        "d2m",
        "mslhf",
        "msshf",
        "sp",
        "ssr",
        "str",
        "t2m",
        "tp",
        "tvh",
        "tvl",
    ]
    data_freq = "monthly"


class ERA5LandMonthly(ERA5DataSource):
    """Monthly ERA5-land dataset."""

    name: str = "era5_land_monthly"
    variable_names: list[str] = [
        "skt",
        "stl1",
        "stl2",
        "stl3",
        "stl4",
        "swvl1",
        "swvl2",
        "swvl3",
        "swvl4",
    ]
    data_freq = "monthly"
