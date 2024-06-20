"""Utils shared by all data sets."""

import warnings

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path


warnings.filterwarnings("ignore")  # suppress nanosecond warning for xarray


def convert_timestamps(dataset: xr.Dataset) -> xr.Dataset:
    """Convert monthly timestamps to the standard (1st day of the month).

    Args:
        dataset: Input dataset that needs to be converted.

    Returns:
        Input dataset, but with the timestamps converted.
    """
    ds = dataset.copy()
    ds["time"] = np.array(
        [
            pd.Timestamp(year=year, month=month, day=1).to_datetime64()
            for year, month in zip(
                ds["time"].dt.year.values,
                ds["time"].dt.month.values,
                strict=True,
            )
        ]
    )
    return ds


def h5encoded(files: list[Path]) -> bool:
    """Figure out of any of the netCDF files use a hdf5 compression filter."""
    if len(files) == 0:
        msg = "There are no files in the list."
        raise ValueError(msg)
    for file in files:
        try:
            xr.open_dataset(file, chunks="auto")
        except RuntimeError as err:
            if "NetCDF: Filter error: undefined filter" in str(err):
                return True
            raise err
    return False
