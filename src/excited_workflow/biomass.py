from pathlib import Path

import numpy as np
import xarray as xr


def cftime_to_datetime(data: xr.DataArray) -> np.ndarray:
    """Convert cftime dataarray values to a numpy datetime format.

    Args:
        data: DataArray containing the time values, e.g. ds["time"].

    Returns:
        Numpy array with a datetime64 dtype.
    """
    return np.array([np.datetime64(el) for el in data.to_numpy()])


def load_biomass_data(file: Path | str) -> xr.Dataset:
    ds = xr.open_dataset(file)
    ds["time"] = cftime_to_datetime(ds["time"])
    ds = ds.drop("time_bnds")
    return ds


def load_and_resample_biomass_data(
    file: Path | str, target_dataset: xr.Dataset
) -> xr.Dataset:
    """Load in and resample the biomass dataset to the target dataset.

    Args:
        file: Filename or path to the biomass netCDF file.
        target_dataset: Dataset with a "time" dimension, to which the biomass dataset
            should be resampled to.

    Returns:
        Loaded in and resampled biomass dataset.
    """
    ds = load_biomass_data(file)
    ds = ds.interp(time=target_dataset["time"], method="linear")
    ds = ds.interpolate_na(
        dim="time",
        method="nearest",
        max_gap=np.timedelta64(5, "Y"),
        fill_value="extrapolate",
    )
    ds = ds.rename({"lat": "latitude", "lon": "longitude"})
    return ds
