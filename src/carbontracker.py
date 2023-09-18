import numpy as np
import xarray as xr


def round_half_up(n: float) -> float:
    """Round a number using the 'round upward' strategy.

    Functions like round() or np.round() use the 'even' strategy, where .5 values are
    rounded up to the nearest even integer. E.g. 1.5 -> 2, and 2.5 -> 2.
    This function rounds 1.5 -> 2, and 2.5 -> 3.

    Args:
        n: Floating point number

    Returns:
        Input number rounded using the 'round upward' strategy.
    """
    if n - np.floor(n) < 0.5:
        return np.floor(n)
    return np.ceil(n)


def round_to_half(a: float) -> float:
    return round_half_up(a - 0.5) + 0.5


def coarsen_era5(dataset: xr.Dataset) -> xr.Dataset:
    """Coarsen ERA5 data to a CarbonTracker-like grid.

    Args:
        dataset: Input data at native resolution.

    Returns:
        The input data resampled to a 1-degree grid.
    """
    lat_min = dataset["latitude"].min().to_numpy()
    lat_max = dataset["latitude"].max().to_numpy()
    lon_min = dataset["longitude"].min().to_numpy()
    lon_max = dataset["longitude"].max().to_numpy()

    dsr = dataset.rolling(latitude=5, center=True).mean()
    dsr = dsr.rolling(longitude=5, center=True).mean()

    lats = np.arange(round_to_half(lat_min), np.floor(lat_max) + 0.5, step=1)
    lons = np.arange(round_to_half(lon_min), np.floor(lon_max) + 0.5, step=1)

    return dsr.sel(latitude=lats, longitude=lons)


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
