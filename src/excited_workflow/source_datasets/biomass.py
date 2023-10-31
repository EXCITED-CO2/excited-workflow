"""Global biomass dataset."""
import numpy as np
import pandas as pd
import xarray as xr
import xarray_regrid  # noqa: F401

from excited_workflow.protocol import DataSource


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

    name = "biomass"
    variable_names = ["biomass"]

    @classmethod
    def load(
        cls, variables: list[str] | None = None, target_grid: xr.Dataset | None = None
    ) -> xr.Dataset:
        """Load variables from this data source and regrid them to the target grid.

        Args:
            variables: List of variable names which should be downloaded.
            target_grid: Grid to which the data should be regridded to.

        Returns:
            Prepared dataset.
        """
        cls.validate_variables(variables)
        ds = xr.open_dataset(cls.get_path().glob("*.nc"))

        ds["time"] = _cftime_to_datetime(ds["time"])
        ds = ds.drop("time_bnds")
        time_range = pd.date_range(start="1995-01-01", end="2022-01-01", freq="1H")
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
