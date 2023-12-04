"""Train carbon tracker datasets."""

from pathlib import Path

import matplotlib.pyplot as plt
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
    """Limit data to a region and time slice.
    
    Args:
        regions: path to regions file.
        target: path to target dataset.
        ds_input: input dataset.
    
    Returns:
        Dataset of masked to North America.
    """
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
    """Create dataframe with different groups.
    
    Args:
        ds: Dataset to split into groups.
        bin_no: number of groups.
    
    Returns:
        Dataframe with groups column.
    """
    df_train = ds.to_dataframe().dropna()
    bins = bin_no
    splits = np.array_split(df_train, bins)
    for i in range(len(splits)):
        splits[i]["group"] = i + 1 

    df_train = pd.concat(splits)
    return df_train


def create_df(df: pd.DataFrame, x_keys: list[str], y_key: str) -> pd.DataFrame:
    """Create a dataframe for training.
    
    Args:
        df: dataframe which needs to be converted for training.
        x_keys: list of input variables.
        y_key: target variable name.
    
    Returns:
        Dataframe for training.
    """
    df_pycaret = df[x_keys + [y_key]]
    df_reduced = df_pycaret[::10]

    df_reduced[y_key] = df_reduced[y_key]*1e6  # So RMSE etc. are easier to interpret.

    return df_reduced


def train_model(df: pd.DataFrame, bin_no: int, x_keys: list[str], y_key: str
                ):
    """Train a model on training data and create prediction.
    
    Args:
        df: dataframe which needs to be converted for training.
        bin_no: number of groups.
        x_keys: list of input variables.
        y_key: target variable name.
    
    Returns:
        Dataset for training and prediction dataset.
    """
    mask = df["group"].values != bin_no
    df_train = df[mask]
    df_reduced = create_df(df_train, x_keys, y_key)
    ds_reduced = xr.Dataset.from_dataframe(df_reduced)

    pycs = pycaret.regression.setup(df_reduced, target=y_key)
    model = pycs.compare_models(include=["lightgbm"], n_select=1, round=1)

    df_prediction = df[~mask]
    data = create_df(df_prediction, x_keys, y_key)
    data.drop(y_key, axis=1, inplace=True)

    prediction = pycs.predict_model(model, data=data)
    ds_prediction = xr.Dataset.from_dataframe(prediction)

    return ds_reduced, ds_prediction


def calculate_rmse(prediction, target):
    """Calculate RMSE.
    
    Args:
        prediction: column of prediction.
        target: column of target.
    
    Returns:
        RMSE 
    """
    rmse = np.sqrt(((prediction - target) ** 2).mean(dim="time", skipna=True))
    return rmse
    

def validate_model(ds, bins, x_keys, y_key):
    """Validate the trained model."""
    df_group = create_bins(ds, bins)

    for i in range(1,bins):
        target_ds, prediction = train_model(df_group, i, x_keys, y_key)
        rmse = calculate_rmse(prediction["prediction_label"], target_ds[y_key])
        plt.scatter(prediction["prediction_label"], target_ds[y_key])
        plt.savefig("scatter" + str(i) + ".png")
        plt.close()
    
    return rmse, prediction
 

if __name__ == "__main__":
    client = Client()

    ct_path = Path("/data/volume_2/EXCITED_prepped_data/CT2022.flux1x1-monthly.nc")
    regions_path = Path("/data/volume_2/EXCITED_prepped_data/regions.nc")

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

    ds_input = merge_datasets(desired_data, ct_path)
    ds_na = mask_region(regions_path, ct_path, ds_input)
    validate_model(ds_na, 5, x_keys, y_key)
