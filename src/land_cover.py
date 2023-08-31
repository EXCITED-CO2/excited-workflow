"""Ingest land cover data."""

import numpy as np
import numpy_groupies as npg
import pandas as pd
import xarray as xr
import xarray_regrid  # Importing this will make Dataset.regrid accessible.
from flox import Aggregation
import flox.xarray


def regrid(ds_land_cover: xr.Dataset, ds_target: xr.Dataset) -> xr.Dataset:
    """Regrid land cover data.

    For downscaling, the regridding is performed with nearest neighbour method.
    For upscaling, the regridding takes the most common class from the neighbouring
    pixels.

    Note that currently this approach only supports structured grids. And the target
    grid should not exceed the boundary of input land cover dataset.

    For both datasets, latitude and longitude coordinates must be sorted as
    monotonically increase.

    Args:
        ds_land_cover: land cover dataset with latitude (`lat`) and longitude (`lon`)
                       coordinates.
        ds_target: dataset containing target grid for regridding, must contain
                   latitude (`lat`) and longitude (`lon`) coordinates.

    Returns:
        xarray.dataset with regridded land cover categorical data.
    """
    _boundary_check(ds_land_cover["lat"].values, ds_target["lat"].values)
    _boundary_check(ds_land_cover["lon"].values, ds_target["lon"].values)
    # get resolution
    (cell_lat_land_cover, cell_lon_land_cover) = infer_resolution(ds_land_cover)
    (cell_lat_target, cell_lon_target) = infer_resolution(ds_target)
    # upscaling case - zonal statistics of most common class
    if cell_lat_land_cover < cell_lat_target and cell_lon_land_cover < cell_lon_target:
        ds_land_cover_regrid = upsample(
            ds_land_cover, ds_target, cell_lat_target, cell_lon_target
        )
    # downscaling case - nearest neighbour
    else:
        ds_land_cover_regrid = ds_land_cover.regrid.regrid(ds_target, method="nearest")

    return ds_land_cover_regrid


def upsample(
    ds_land_cover: xr.Dataset,
    ds_target: xr.Dataset,
    cell_lat_target: float,
    cell_lon_target: float,
) -> xr.Dataset:
    """Upsampling of land cover with most common label approach.

    The implementation includes two steps:
    - "groupby" lat/lon
    - select most common label

    We use flox to perform "groupby" multiple dimensions. Here is an example:
    https://flox.readthedocs.io/en/latest/intro.html#histogramming-binning-by-multiple-variables

    To embed our customized function for most common label selection, we need to
    create our `flox.Aggregation`, for instance:
    https://flox.readthedocs.io/en/latest/aggregations.html

    `flox.Aggregation` function works with `numpy_groupies.aggregate_numpy.aggregate
    API. Therefore this function also depends on `numpy_groupies`. For more information,
    check the following example:
    https://flox.readthedocs.io/en/latest/user-stories/custom-aggregations.html

    Args:
        ds_land_cover: land cover dataset with latitude (`lat`) and longitude (`lon`)
                       coordinates.
        ds_target: dataset containing target grid for regridding, must contain
                   latitude (`lat`) and longitude (`lon`) coordinates.
        cell_lat_target: resolution of latitude from target grid.
        cell_lon_target: resolution of lontitude from target grid.

    Returns:
        xarray.dataset with regridded land cover categorical data.
    """
    # create bounds for lat/lon
    lat_bounds = _construct_intervals(ds_target["lat"].values, cell_lat_target)
    lon_bounds = _construct_intervals(ds_target["lon"].values, cell_lon_target)
    # groupby
    most_common = Aggregation(
        name="most_common", numpy=_custom_grouped_reduction, chunk=None, combine=None
    )
    ds_regrid = flox.xarray.xarray_reduce(
        ds_land_cover,
        "lat",
        "lon",
        func=most_common,
        expected_groups=(lat_bounds, lon_bounds),
    )
    # create regridding dataset
    ds_land_cover_regrid = _create_ds_grid(ds_land_cover, ds_target)
    ds_land_cover_regrid["lccs_class"].values = ds_regrid["lccs_class"].values

    return ds_land_cover_regrid


def infer_resolution(dataset: xr.Dataset) -> tuple[float, float]:
    """Infer the resolution of a dataset's latitude and longitude coordinates.
    (zampy.utils.regrid)

    Args:
        dataset: dataset with latitude and longitude coordinates.

    Returns:
        The latitude and longitude resolution.
    """
    resolution_lat = np.median(
        np.diff(
            dataset["lat"].to_numpy(),
            n=1,
        )
    )
    resolution_lon = np.median(
        np.diff(
            dataset["lon"].to_numpy(),
            n=1,
        )
    )

    return (resolution_lat, resolution_lon)


def _construct_intervals(coord: np.ndarray, step_size: float) -> pd.IntervalIndex:
    """Create pandas.intervals with given coordinates."""
    breaks = np.append(coord, coord[-1] + step_size) - step_size / 2

    # Note: closed="both" triggers an `NotImplementedError`
    return pd.IntervalIndex.from_breaks(breaks, closed="left")


def _most_common_label(neighbors: np.ndarray) -> int:
    """Find the most common label in a neighborhood.

    Note that if more than one labels have the same frequency which is the highest,
    then the first label in the list will be picked.
    """
    unique_labels, counts = np.unique(neighbors, return_counts=True)
    return unique_labels[np.argmax(counts)]


def _custom_grouped_reduction(
    group_idx: np.ndarray,
    array: np.ndarray,
    *,
    axis: int = -1,
    size: int = None,
    fill_value=None,
    dtype=None
) -> np.ndarray:
    """Custom grouped reduction for flox.Aggregation to get most common label.

    Args:
        group_idx : integer codes for group labels (1D)
        array : values to reduce (nD)
        axis : axis of array along which to reduce.
            Requires array.shape[axis] == len(group_idx)
        size : expected number of groups. If none,
            output.shape[-1] == number of uniques in group_idx
        fill_value : fill_value for when number groups in group_idx is less than size
        dtype : dtype of output

    Returns:
        np.ndarray with array.shape[-1] == size, containing a single value per group
    """
    return npg.aggregate_numpy.aggregate(
        group_idx,
        array,
        func=_most_common_label,
        axis=axis,
        size=size,
        fill_value=fill_value,
        dtype=dtype,
    )


def _create_ds_grid(ds_land_cover: xr.Dataset, ds_target: xr.Dataset) -> xr.Dataset:
    """Create empty/dummy dataset with target grid.

    Also copy the relevant attributes from given land cover dataset.
    The variable name of land cover must be "lccs_class".


    Args:
        ds_land_cover: land cover dataset with latitude (`lat`) and longitude (`lon`)
                       coordinates. 
        ds_target: dataset containing target grid for regridding, must contain
                   latitude (`lat`) and longitude (`lon`) coordinates.

    Returns:
        xarray.dataset with target grid and dummy land cover variable.
    """
    lat = ds_target["lat"].values
    lon = ds_target["lon"].values
    time = ds_land_cover["time"].values
    dummy_data = np.ones((len(time), len(lat), len(lon)))

    ds_grid = xr.Dataset(
        data_vars=dict(
            lccs_class=(["time", "lat", "lon"], dummy_data),
        ),
        coords=dict(
            time=time,
            lat=(["lat"], lat),
            lon=(["lon"], lon),
        ),
        attrs=dict(description="Regridded land cover dataset."),
    )
    ds_grid.attrs.update(ds_land_cover.attrs)
    ds_grid["lccs_class"].attrs.update(ds_land_cover["lccs_class"].attrs)

    return ds_grid


def _boundary_check(ds_coord: np.ndarray, ds_target_coord: np.ndarray) -> None:
    """Check if target grid can be covered by grid of input dataset."""
    if ds_target_coord.min() < ds_coord.min() or ds_target_coord.max() > ds_coord.max():
        raise ValueError("The original grid can not cover the target grid!")
