"""Ingest SPEI data."""

from pathlib import Path
import xarray as xr
import numpy as np
from src import utils


def load_spei_data(path: Path | str) -> xr.Dataset:
    ds = xr.open_dataset(path, chunks={"time": -1, "lat": 1, "lon": 1})
    ds = utils.convert_timestamps(ds)
    ds = ds.rename({"lat": "latitude", "lon": "longitude"})

    return ds
