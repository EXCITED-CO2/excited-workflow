"""Train carbon tracker datasets."""

from pathlib import Path

import numpy as np
import pandas as pd
import pycaret.regression
import xarray as xr
import xarray_regrid  # Importing this will make Dataset.regrid accessible.
from dask.distributed import Client
from pycaret.classification import predict_model

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
                target: Path, 
                ds_input: xr.Dataset
                ) -> xr.Dataset:
    """Limit data to a region and time slice."""
    ds_regions = xr.open_dataset(regions)
    ds_cb = xr.open_dataset(target)
    ds_cb = excited_workflow.utils.convert_timestamps(ds_cb)
    ds_merged = xr.merge([
        ds_cb[["bio_flux_opt"]], 
        ds_regions["transcom_regions"],
        ds_input,
        ])
    time_region_na = {"time": slice("2000-01", "2019-12"), 
                      "latitude": slice(15, 60), 
                      "longitude": slice(-140, -55),
                      }
    ds_na = ds_merged.sel(time_region_na)
    ds_na = ds_na.compute()
    ds_na = ds_na.where(ds_merged["transcom_regions"]==2)

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


def create_df(df: pd.DataFrame, x_keys: list[str], y_key: str) -> pd.DataFrame:
    """Create a dataframe for training."""
    df_pycaret = df[x_keys + [y_key]]
    df_reduced = df_pycaret[::10]

    df_reduced[y_key] = df_reduced[y_key]*1e6  # So RMSE etc. are easier to interpret.

    return df_reduced


def train_model(df: pd.DataFrame, x_keys: list[str], y_key: str) -> xr.Dataset:
    """Train a model on training data and create prediction."""
    df_reduced = create_df(df, x_keys, y_key)

    pycs = pycaret.regression.setup(df_reduced, target=y_key)
    best = pycs.compare_models(n_select=5, round=2)

    data2 = create_df(df, x_keys, y_key)
    data2.drop(y_key, axis=1, inplace=True)

    prediction = pycs.predict_model(best[1], data=data2)
    ds_prediction = xr.Dataset.from_dataframe(prediction)

    return ds_prediction


def create_validation_data(df_train: pd.DataFrame, 
                     bin_no: int, 
                     x_keys: list[str], 
                     y_key: str
                     ):
    """Validate data."""
    df = df_train[df_train["group"] != bin_no]
    prediction = train_model(df, x_keys, y_key)

    return prediction


def calculate_rmse(prediction, target):
    """Calculate RMSE and scatterplot."""
    rmse = np.sqrt(((prediction - target) ** 2).mean(dim="time"))
    return rmse


def validate_model(ds, bins, x_keys, y_key, target):
    """Validate the trained model."""
    df_group = create_bins(ds, bins)

    for i in range(bins):
        prediction = create_validation_data(df_group, i, x_keys, y_key)
        rmse = calculate_rmse(prediction["prediction_label"], target["y_key"])
        print(rmse)
 

if __name__ == "__main__":
    client = Client()

    ds_cb = Path("/data/volume_2/EXCITED_prepped_data/CT2022.flux1x1-monthly.nc")
    ds_regions = Path("/data/volume_2/EXCITED_prepped_data/regions.nc")

    desired_data = [
        "biomass",
        "spei",
        "modis",
        "era5_monthly",
        "era5_land_monthly",
        "copernicus_landcover"
    ]

    x_keys = ["d2m", "mslhf", "msshf", "ssr", "str", "t2m", "spei", "NIRv", "skt",
               "stl1", "swvl1", "lccs_class"]
    y_key = "bio_flux_opt"

    print("I am attempting to run things")
    ds_input = merge_datasets(desired_data, ds_cb)
    ds_na = mask_region(ds_regions, ds_cb, ds_input)
    validate_model(ds_na, 5, x_keys, y_key, ds_cb)
