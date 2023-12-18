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
    y_key: str,
) -> tuple[xr.Dataset, pd.DataFrame]:
    """Limit data to a region and time slice.

    Args:
        regions: path to regions file.
        target: path to target dataset.
        ds_input: input dataset.
        mask: name of region to mask to.
        y_key: target variable name.

    Returns:
        Masked dataset and dataframe.
    """
    ds_regions = xr.open_dataset(regions)
    ds_cb = xr.open_dataset(target)
    ds_cb = excited_workflow.utils.convert_timestamps(ds_cb)
    ds_merged = xr.merge(
        [
            ds_cb[[y_key]],
            ds_regions["transcom_regions"],
            ds_input,
        ]
    )

    ds_sel = ds_merged.sel({"time": slice("2000-01", "2019-12")})
    ds_sel = ds_sel.compute()
    ds_sel = ds_sel.where(ds_merged["transcom_regions"] == mask)
    ds_sel[y_key].attrs["long_name"] = "terrestrial biosphere CO2 flux"
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


def create_scatterplots(
    predictions: list[xr.Dataset],
    targets: list[xr.Dataset],
    y_key: str,
    target_units: str,
    output_dir: Path,
) -> str:
    """Create scatterplots of prediction vs. target.

    Args:
        predictions: list of predictions datasets per group.
        targets: list of target datasets per group.
        y_key: name of target variable.
        target_units: units for target variable.
        output_dir: Path to output directory
    """
    text = "Scatter plots for groupwise cross validation. \n\n"
    for idx, (target, prediction) in enumerate(zip(targets, predictions, strict=True)):
        plt.figure(figsize=(5, 5))
        plt.scatter(prediction["prediction_label"], target[y_key], s=20)
        plt.axline((0, 0), slope=1, color="black")
        plt.xlabel(f"{target[y_key].name} ({target_units})")
        plt.ylabel(f"{prediction['prediction_label'].name} ({target_units})")
        plt.tight_layout()
        plt.savefig(output_dir / f"scatter{idx}.png")
        plt.close()
        text += f"**Scatter plot {idx}**\n ![image]({output_dir}/scatter{idx}.png)\n\n"

    return text


def make_full_plot(
    predictions: list[xr.Dataset],
    targets: list[xr.Dataset],
    y_key: str,
    target_units: str,
    output_dir: Path,
) -> str:
    """Make scatterplot for all groups.

    Args:
        predictions: list of predictions datasets per group.
        targets: list of target datasets per group.
        y_key: name of target variable.
        target_units: units for target variable.
        output_dir: Path to output directory
    """
    text = "Scatterplot for all groups. \n\n"
    fig = plt.figure(figsize=(6, 6))
    ax = plt.axes()
    for target, prediction in zip(targets, predictions, strict=True):
        ax.scatter(target[y_key], prediction["prediction_label"], s=12, alpha=0.5)
        ax.axline((0, 0), slope=1, color="black")
    ax.set_xlabel(f"{target[y_key].name} ({target_units})")
    ax.set_ylabel(f"{prediction['prediction_label'].name} ({target_units})")
    fig.tight_layout()
    fig.savefig(output_dir / "all_scatter.png")
    fig.clf()
    text += f"**Scatter plot all**\n ![image]({output_dir}/all_scatter.png)\n\n"

    return text


def create_rmseplots(rmses: list[xr.DataArray], output_dir: Path) -> str:
    """Create map plot for rmse.

    Args:
        rmses: list of rmse dataarrays
        output_dir: Path to output directory
    """
    text = "Rmse plots for groupwise cross validation. \n\n"
    for idx, rmse in enumerate(rmses):
        rmse.to_netcdf(output_dir / f"rmse{idx}.nc")
        plt.figure(figsize=(5, 3))
        rmse.plot()
        plt.tight_layout()
        plt.savefig(output_dir / f"rmseplot{idx}.png")
        plt.close()
        text += f"**RMSE map {idx}**\n ![image]({output_dir}/rmseplot{idx}.png)\n\n"

    return text


def create_markdown_file(
    ds: xr.Dataset,
    predictions: list[xr.Dataset],
    targets: list[xr.Dataset],
    rmses: list[xr.DataArray],
    x_keys: list[str],
    y_key: str,
    output_dir: Path,
) -> None:
    """Create a markdown file with a model description and validation plots.

    Args:
        ds: dataset for training.
        predictions: list of predictions datasets per group.
        targets: list of target datasets per group.
        rmses: list of rmse dataarrays.
        x_keys: list of input variables.
        y_key: target variable name.
        output_dir: directory to output rmse and scatterplots.
    """
    long_names = [
        ds[var].attrs["long_name"] if "long_name" in ds[var].attrs else "undefined"
        for var in x_keys
    ]
    units = [
        ds[var].attrs["units"] if "units" in ds[var].attrs else "?" for var in x_keys
    ]

    text = """## Carbon tracker model 
    
Description of model for carbon tracker workflow.

    ### Model variables (in order): \n\n"""

    for var, long_name, unit in zip(x_keys, long_names, units, strict=True):
        text += f"- **{var}**: {long_name} (`{unit}`)\n"

    text += "\n\n ###Target variable:\n\n"
    target_name = ds[y_key].attrs["long_name"]
    target_units = ds[y_key].attrs["units"]
    text += f"{y_key}: {target_name} ({target_units})\n"

    text += """\n### Validation plots

Dataset is split into 5 equal groups for crosswise validation, one group is left out of 
training iteratively and used for validation. Then rmse is calculated and mapped and 
scatterplots for each group are created. \n\n"""

    text += create_scatterplots(predictions, targets, y_key, target_units, output_dir)
    text += make_full_plot(predictions, targets, y_key, target_units, output_dir)
    text += create_rmseplots(rmses, output_dir)

    with open(output_dir / "model_description.md", "w") as file:
        file.write(text)


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

    predictions = []
    targets = []
    rmses = []
    for group in range(groups):
        target_ds, prediction_ds = groupwise_cross_validation(
            df_group, group, x_keys, y_key
        )
        rmse = calculate_rmse(prediction_ds["prediction_label"], target_ds[y_key])
        predictions.append(prediction_ds)
        targets.append(target_ds)
        rmses.append(rmse)

    create_markdown_file(ds, predictions, targets, rmses, x_keys, y_key, output_dir)


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
    ds_na, df = mask_region(regions_file, cb_file, ds_input, 2, y_key)
    validate_model(ds_na, 5, x_keys, y_key, output_dir)
    pycs, model = train_model(df, x_keys, y_key)
    save_model(pycs, model, output_dir)
