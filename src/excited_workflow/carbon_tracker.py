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


def mask_region(
    regions: Path,
    target: Path,
    ds_input: xr.Dataset,
    mask: int,
) -> tuple[xr.Dataset, pd.DataFrame]:
    """Limit data to a region and time slice.

    Args:
        regions: path to regions file.
        target: path to target dataset.
        mask: name of region to mask to.
        ds_input: input dataset.

    Returns:
        Masked dataset and dataframe.
    """
    ds_regions = xr.open_dataset(regions)
    ds_cb = xr.open_dataset(target)
    ds_cb = excited_workflow.utils.convert_timestamps(ds_cb)
    ds_merged = xr.merge(
        [
            ds_cb[["bio_flux_opt"]],
            ds_regions["transcom_regions"],
            ds_input,
        ]
    )

    ds_sel = ds_merged.sel({"time": slice("2000-01", "2019-12")})
    ds_sel = ds_sel.compute()
    ds_sel = ds_sel.where(ds_merged["transcom_regions"] == mask)
    df_sel = ds_sel.to_dataframe().dropna()

    return ds_sel, df_sel


def create_groups(ds: xr.Dataset, number: int) -> pd.DataFrame:
    """Create dataframe with different groups.

    Args:
        ds: Dataset to split into groups.
        number: number of groups.

    Returns:
        Dataframe with groups column.
    """
    df = ds.to_dataframe().dropna()
    splits = np.array_split(df, number)
    for i in range(len(splits)):
        splits[i]["group"] = i
    df_train = pd.concat(splits)
    return df_train


def train_model(df: pd.DataFrame, x_keys: list[str], y_key: str) -> tuple[Any, Any]:
    """Train model.

    Args:
        df: dataframe for training data.
        x_keys: list of input variables.
        y_key: target variable name.

    Returns:
        trained model.
    """
    df_reduced = df[x_keys + [y_key]]

    pycs = pycaret.regression.setup(df_reduced, target=y_key, verbose=False)
    model = pycs.compare_models(
        include=["lightgbm"], n_select=1, round=5, cross_validation=False
    )

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
    ds: xr.Dataset, groups: int, x_keys: list[str], y_key: str, output_dir: Path
) -> None:
    """Validate the trained model by calculating rmse and scatterplots.

    Args:
        ds: dataset for training.
        groups: number of groups.
        x_keys: list of input variables.
        y_key: target variable name.
        output_dir: directory to output rmse and scatterplots.
    """
    df_group = create_groups(ds, groups)

    long_names = [
        ds[var].attrs["long_name"] if "long_name" in ds[var].attrs else "undefined"
        for var in x_keys
    ]
    units = [
        ds[var].attrs["units"] if "units" in ds[var].attrs else "?" for var in x_keys
    ]

    text = "## Carbon tracker model \n \n### Model variables (in order): \n\n"

    for var, long_name, unit in zip(x_keys, long_names, units, strict=True):
        text += f"- {var}: {long_name} ({unit})\n"

    text += "\n### Validation plots \n\n"
    rmses = []
    for group in range(groups):
        target_ds, prediction = groupwise_cross_validation(
            df_group, group, x_keys, y_key
        )
        rmse = calculate_rmse(prediction["prediction_label"], target_ds[y_key])
        rmses.append(rmse)
        create_scatterplot(prediction["prediction_label"], target_ds[y_key])
        plt.savefig(output_dir / f"scatter{group}.png")
        plt.close()
        text += (
            f"**Scatter plot {group}** \n ![image]({output_dir}/scatter{group}.png) \n"
        )

    for idx, rmse in enumerate(rmses):
        rmse.to_netcdf(output_dir / f"rmse{idx}.nc")
        create_rmseplot(rmse)
        plt.savefig(output_dir / f"rmseplot{idx}.png")
        plt.close()
        text += f"**RMSE map {idx}** \n ![image]({output_dir}/rmseplot{idx}.png) \n"

    with open(output_dir / "model_description.md", "w") as file:
        file.write(text)


def save_model(pycs: Any, model: Any, output_dir: Path) -> None:
    """Create lightgbm model for whole dataset and save with ONNX.

    Args:
        pycs: model variables
        model: trained model
        output_dir: path to output directory.
    """
    x_test = pycs.get_config("X_test").to_numpy()

    lightgbm_onnx = onnxmltools.convert_lightgbm(
        model, initial_types=[("X", DoubleTensorType([None, x_test.shape[1]]))]
    )

    with open(output_dir / "lightgbm.onnx", "wb") as f:
        f.write(lightgbm_onnx.SerializeToString())


if __name__ == "__main__":
    client = Client()

    cb_file = Path("/data/volume_2/EXCITED_prepped_data/CT2022.flux1x1-monthly.nc")
    regions_file = Path("/data/volume_2/EXCITED_prepped_data/regions.nc")
    output_path = Path.home()

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

    time = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    output_dir = output_path / f"carbon_tracker-{time}"
    output_dir.mkdir(parents=True, exist_ok=True)

    ds_cb = xr.open_dataset(cb_file)
    ds_cb = excited_workflow.utils.convert_timestamps(ds_cb)
    ds_input = xr.merge(
        [
            datasets[name].load(freq="monthly", target_grid=ds_cb)
            for name in desired_data
        ]
    )
    ds_na, df = mask_region(regions_file, cb_file, ds_input, 2)
    validate_model(ds_na, 5, x_keys, y_key, output_dir)
    pycs, model = train_model(df, x_keys, y_key)
    save_model(pycs, model, output_dir)
