"""Ingest SPEI data."""

from pathlib import Path
import xarray as xr
import numpy as np
from src import utils


def load_spei_data(path: Path | str) -> xr.Dataset:
    ds = xr.open_dataset(path)
    ds = utils.convert_timestamps(ds)
    ds = ds.rename({"lat": "latitude", "lon": "longitude"})

    return ds
