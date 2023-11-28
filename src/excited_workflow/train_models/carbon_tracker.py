"""Train carbon tracker datasets."""

from pathlib import Path

import numpy as np
import pandas as pd
import pycaret.regression
import xarray as xr
import xarray_regrid  # Importing this will make Dataset.regrid accessible.
from dask.distributed import Client

import excited_workflow
from excited_workflow.source_datasets import datasets


def merge_datasets(desired_data: list[str], target: Path) -> xr.Dataset:
    """Merge datasets onto one common grid.
    
    Args:
        desired_data: list of desired datasets
        freq: time frequency that is desired
        target: target grid dataset

    Returns:
        Dataset of input variables.
    """
    ds_cb = xr.open_dataset(target)
    ds_cb = excited_workflow.utils.convert_timestamps(ds_cb)
    ds_input = xr.merge([datasets[name].load(freq="monthly", target_grid=ds_cb) 
                         for name in desired_data])
    
    return ds_input


def mask_region(regions: Path, 
                region_name: str, 
                ds_cb: xr.Dataset, 
                ds_input: xr.Dataset
                ) -> xr.Dataset:
    """Limit data to a region and time slice."""
    ds_regions = xr.open_dataset(regions)
    ds_merged = xr.merge([ds_cb[["bio_flux_opt"]], 
                          ds_regions[region_name], 
                          ds_input,
                          ])
    time_region_na = {"time": slice("2000-01", "2019-12"), 
                      "latitude": slice(15, 60), 
                      "longitude": slice(-140, -55),
                      }
    ds_na = ds_merged.sel(time_region_na)
    ds_na = ds_na.compute()

    return ds_na


def create_bins(ds: xr.Dataset, bin_no: int) -> pd.DataFrame:
    """Create dataframe with different groups."""
    df_train = ds.to_dataframe().dropna()
    bins = bin_no
    splits = np.array_split(df_train, bins)
    for i in range(len(splits)):
        splits[i]['group'] = i + 1
        
    df_train = pd.concat(splits)

    return df_train


def train_model(df: pd.DataFrame, x_keys: list[str], y_key: str):
    """Train model on input datasets."""
    df_pycaret = df[x_keys + [y_key]]
    df_reduced = df_pycaret[::10]

    df_reduced[y_key] = df_reduced[y_key]*1e6  # So RMSE etc. are easier to interpret.

    pycs = pycaret.regression.setup(df_reduced, target=y_key)

    return pycs


def validation_model(df_train: pd.DataFrame, 
                     bin_no: int, 
                     x_keys: list[str], 
                     y_key: str
                     ):
    """Validate data."""
    df = df_train[df_train["group"] != bin_no]
    model = train_model(df, x_keys, y_key)

    return model


def model_statistics(prediction, target):
    """Calculate RMSE and scatterplot."""
    #ds = xr.Dataset.from_dataframe(df)
    rmse = np.sqrt(((prediction - target) ** 2).mean())
    return rmse

#if __name__ == "__main__":
#    client = Client()
#
#    ds_cb = "/data/volume_2/EXCITED_prepped_data/CT2022.flux1x1-monthly.nc"
#    ds_regions = "/data/volume_2/EXCITED_prepped_data/regions.nc"
#
#    desired_data = [
#        "biomass",
#        "spei",
#        "modis",
#        "era5_monthly",
#        "era5_land_monthly",
#        "copernicus_landcover"
#    ]
#
#    x_keys = ["d2m", "mslhf", "msshf", "ssr", "str", "t2m", "spei", "NIRv", "skt", "stl1", "swvl1", "lccs_class"]
#    y_key = "bio_flux_opt"
#
#    ds_input = merge_datasets(desired_data, ds_cb)
#    ds_na = mask_region()
#    df_train = create_bins()
#    for i in bins:
#        m = validation_model()

