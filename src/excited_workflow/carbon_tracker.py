"""Train carbon tracker datasets."""

import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import onnxmltools
import pandas as pd
import pycaret.regression
import xarray as xr
from dask.distributed import Client
from skl2onnx.common.data_types import DoubleTensorType

import excited_workflow
from excited_workflow.source_datasets import datasets


def merge_datasets(desired_data: list[str], target: Path) -> xr.Dataset:
    """Merge datasets onto one common grid.

    Args:
        desired_data: list of desired datasets
        target: target grid dataset

    Returns:
        Dataset of input variables.
    """
    ds_cb = xr.open_dataset(target)
    ds_cb = excited_workflow.utils.convert_timestamps(ds_cb)
    ds_input = xr.merge(
        [
            datasets[name].load(freq="monthly", target_grid=ds_cb)
            for name in desired_data
        ]
    )

    return ds_input


def mask_region(
    regions: Path, target: Path, mask: str, ds_input: xr.Dataset
) -> xr.Dataset:
    """Limit data to a region and time slice.

    Args:
        regions: path to regions file.
        target: path to target dataset.
        mask: name of region to mask to.
        ds_input: input dataset.

    Returns:
        Masked dataset.
    """
    ds_regions = xr.open_dataset(regions)
    ds_cb = xr.open_dataset(target)
    ds_cb = excited_workflow.utils.convert_timestamps(ds_cb)
    ds_merged = xr.merge(
        [
            ds_cb[["bio_flux_opt"]],
            ds_regions[mask],
            ds_input,
        ]
    )

    ds_sel = ds_merged.sel({"time": slice("2000-01", "2019-12")})
    ds_sel = ds_sel.compute()
    ds_sel = ds_sel.where(ds_merged[mask] == 2)

    return ds_sel


def create_groups(ds: xr.Dataset, number: int) -> pd.DataFrame:
    """Create dataframe with different groups.

    Args:
        ds: Dataset to split into groups.
        number: number of groups.

    Returns:
        Dataframe with groups column.
    """
    df_train = ds.to_dataframe().dropna()
    splits = np.array_split(df_train, number)
    for i in range(len(splits)):
        splits[i]["group"] = i
    df_train = pd.concat(splits)  # type: ignore
    return df_train


def train_model(df: pd.DataFrame, x_keys: list[str], y_key: str) -> tuple[Any, Any]:
    """Train model."""
    df_reduced = df[x_keys + [y_key]]

    pycs = pycaret.regression.setup(df_reduced, target=y_key, verbose=False)
    model = pycs.compare_models(include=["lightgbm"], n_select=1, round=7)

    return pycs, model


def groupwise_cross_validation(
    df: pd.DataFrame, number: int, x_keys: list[str], y_key: str
) -> tuple[xr.Dataset, xr.Dataset]:
    """Train a model on training data and create prediction.

    Args:
        df: dataframe which needs to be converted for training.
        number: number of groups.
        x_keys: list of input variables.
        y_key: target variable name.

    Returns:
        Dataset for training and prediction dataset.
    """
    mask = df["group"] != number
    df_train = df[mask]
    pycs, model = train_model(df_train, x_keys, y_key)

    df_prediction = df[~mask]
    data = df_prediction[x_keys + [y_key]]
    ds_target = xr.Dataset.from_dataframe(data)
    data.drop(y_key, axis=1, inplace=True)

    prediction = pycs.predict_model(model, data=data)
    ds_prediction = xr.Dataset.from_dataframe(prediction)

    return ds_target, ds_prediction


def calculate_rmse(prediction: xr.DataArray, target: xr.DataArray) -> xr.DataArray:
    """Calculate RMSE.

    Args:
        prediction: column of prediction.
        target: column of target.

    Returns:
        RMSE
    """
    rmse = np.sqrt(((prediction - target) ** 2).mean(dim="time", skipna=True))
    rmse_da = xr.DataArray(rmse)
    return rmse_da


def create_rmseplot(rmse: xr.DataArray) -> None:
    """Create map plot for rmse.

    Args:
        rmse: rmse dataarray
    """
    plt.figure(figsize=(5, 3))
    rmse.plot()
    plt.tight_layout()


def create_scatterplot(prediction: xr.DataArray, target: xr.DataArray) -> None:
    """Create scatterplot of prediction vs. target.

    Args:
        prediction: dataframe column of prediction.
        target: dataframe column of target variable.
    """
    plt.figure(figsize=(5, 5))
    plt.scatter(prediction, target, s=25)
    plt.xlabel("Prediction")
    plt.ylabel("Target")
    plt.tight_layout()


def validate_model(
    ds: xr.Dataset, groups: int, x_keys: list[str], y_key: str, output_path: Path
) -> None:
    """Validate the trained model by calculating rmse and scatterplots.

    Args:
        ds: dataset for training.
        groups: number of groups.
        x_keys: list of input variables.
        y_key: target variable name.
        output_path: directory to output rmse and scatterplots.
    """
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H")
    output_dir = output_path / f"carbon_tracker-{time}"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_group = create_groups(ds, groups)

    model_vars = [ds[var].attrs["long_name"] for var in x_keys]

    text = "##Carbon tracker model \n" + f"Model variables: \n {model_vars} \n"

    for group in range(groups):
        target_ds, prediction = groupwise_cross_validation(
            df_group, group, x_keys, y_key
        )
        rmse = calculate_rmse(prediction["prediction_label"], target_ds[y_key])
        rmse.to_netcdf(output_dir / f"rmse{group}.nc")
        create_rmseplot(rmse)
        plt.savefig(output_dir / f"rmseplot{group}.png")
        plt.close()
        create_scatterplot(prediction["prediction_label"], target_ds[y_key])
        plt.savefig(output_dir / f"scatter{group}.png")
        plt.close()
        text = (
            text
            + f"**Validation plots for group {group}"
            + f"**RMSE map** \n ![image]({output_dir}/rmseplot{group}.png) \n"
            + f"**Scatter plot** \n ![image]({output_dir}/scatter{group}.png) \n"
        )

    with open(output_dir / "model_description.md", "w") as file:
        file.write(text)


def save_model(
    ds: xr.Dataset, x_keys: list[str], y_key: str, output_path: Path
) -> None:
    """Create lightgbm model for whole dataset and save with ONNX.

    Args:
        ds: dataset used for model creation.
        x_keys: list of input variables.
        y_key: target variable name.
        output_path: path to output directory.
    """
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H")
    output_dir = output_path / f"carbon_tracker-{time}"
    df = ds.to_dataframe().dropna()

    pycs, model = train_model(df, x_keys, y_key)
    x_test = pycs.get_config("X_test").to_numpy()

    lightgbm_onnx = onnxmltools.convert_lightgbm(
        model, initial_types=[("X", DoubleTensorType([None, x_test.shape[1]]))]
    )

    with open(output_dir / "lightgbm.onnx", "wb") as f:
        f.write(lightgbm_onnx.SerializeToString())

    # with open(output_dir / "model_description.md", "a") as file:
    #    file.write(text)


if __name__ == "__main__":
    client = Client()

    ct_path = Path("/data/volume_2/EXCITED_prepped_data/CT2022.flux1x1-monthly.nc")
    regions_path = Path("/data/volume_2/EXCITED_prepped_data/regions.nc")
    output_dir = Path("/home/cdonnelly")

    desired_data = [
        "biomass",
        "spei",
        "modis",
        "era5_monthly",
        "era5_land_monthly",
        "copernicus_landcover",
    ]

    x_keys = [
        "d2m",
        "mslhf",
        "msshf",
        "ssr",
        "str",
        "t2m",
        "spei",
        "NIRv",
        "skt",
        "stl1",
        "swvl1",
        "lccs_class",
    ]
    y_key = "bio_flux_opt"

    ds_input = merge_datasets(desired_data, ct_path)
    ds_na = mask_region(regions_path, ct_path, "transcom_regions", ds_input)
    validate_model(ds_na, 5, x_keys, y_key, output_dir)
    save_model(ds_na, x_keys, y_key, output_dir)
