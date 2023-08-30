"""Ingest land cover data."""

from pathlib import Path
import numpy as np
import numpy_groupies as npg
import pandas as pd
import xarray as xr
from flox import Aggregation
import flox.xarray
import xarray_regrid.utils


def regrid(ds_land_cover, ds_target):
    """Regrid land cover data.

    For downscaling, the regridding is performed with nearest neighbour method.
    For upscaling, the regridding takes the most common class from the neighbouring
    pixels.

    Note that currently this approach only supports structured grids. And the target
    grid should not exceed the boundary of input land cover dataset.
    """
    # TODO: boundary check
    # get resolution
    (cell_lat_land_cover, cell_lon_land_cover) = infer_resolution(ds_land_cover)
    (cell_lat_target, cell_lon_target) = infer_resolution(ds_target)
    # upscaling case
    if cell_lat_land_cover < cell_lat_target and cell_lon_land_cover < cell_lon_target:
        ds_land_cover_regrid = upsample(
            ds_land_cover, ds_target, cell_lat_target, cell_lon_target
        )
    # downscaling case - nearest neighbour
    else:
        raise NotImplementedError

    return ds_land_cover_regrid


def upsample(ds_land_cover, ds_target, cell_lat_target, cell_lon_target):
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
        ds_land_cover: land cover dataset with latitude and longitude coordinates.

    Returns:
        xarray.dataset with regridded land cover categorical data.
    """
    # create bounds for lat/lon
    lat_bounds = pd.IntervalIndex.from_breaks(
        ds_target["lat"].values - cell_lat_target / 2, closed="right"
    )
    lon_bounds = pd.IntervalIndex.from_breaks(
        ds_target["lon"].values - cell_lon_target / 2, closed="right"
    )
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
    ds_land_cover_regrid = xarray_regrid.utils.create_regridding_dataset(ds_target)
    ds_land_cover_regrid["land_cover"] = ds_regrid["land_cover"].values

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
            dataset["latitude"].to_numpy(),
            n=1,
        )
    )
    resolution_lon = np.median(
        np.diff(
            dataset["longitude"].to_numpy(),
            n=1,
        )
    )

    return (resolution_lat, resolution_lon)


def _most_common_label(neighbors):
    """Find the most common label in a neighborhood."""
    unique_labels, counts = np.unique(neighbors, return_counts=True)
    return unique_labels[np.argmax(counts)]


def _custom_grouped_reduction(
    group_idx, array, *, axis=-1, size=None, fill_value=None, dtype=None
):
    """Custom grouped reduction for flox.Aggregation to get most common label.
    Parameters
    ----------

    group_idx : np.ndarray, 1D
        integer codes for group labels (1D)
    array : np.ndarray, nD
        values to reduce (nD)
    axis : int
        axis of array along which to reduce.
        Requires array.shape[axis] == len(group_idx)
    size : int, optional
        expected number of groups. If none,
        output.shape[-1] == number of uniques in group_idx
    fill_value : optional
        fill_value for when number groups in group_idx is less than size
    dtype : optional
        dtype of output

    Returns
    -------

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
