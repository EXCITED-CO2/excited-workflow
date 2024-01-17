"""Workflow to produce the Fluxnet-based ML model(s)."""
import shutil
from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnxmltools
import pycaret.regression
import xarray as xr
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import DoubleTensorType
from sklearn.model_selection import BaseCrossValidator

from excited_workflow import fluxnet_site_extraction
from excited_workflow.common_utils import write_model_variables
from excited_workflow.config import PATHS_CFG
from excited_workflow.source_datasets import datasets


def calculate_era5_derived_vars(era5_dataset: xr.Dataset) -> xr.Dataset:
    """Calculate variables, based on the ERA5 dataset.

    Variables calculated:
        ssr_6hr: six hour rolling mean of the incoming shortwave solar radiation.
        day_of_year: The day of year.
        hour: The hour of day.
        t2m_1w_rolling: 1-week rolling mean of the air temperature.
        dewpoint_depression_1w_rolling: 1-week rolling mean of the dewpoint depression.
        mean_temp: Mean temperature over time (over the entire time period available).
        mean_dewpoint_depression: Mean dewpoint depression.
    """
    era5_dataset = era5_dataset.copy(deep=True)
    era5_dataset["ssr_6hr"] = (
        era5_dataset["ssr"].rolling(time=6, center=True).mean()
    )

    era5_dataset["day_of_year"] = era5_dataset["time"].dt.dayofyear
    era5_dataset["hour"] = era5_dataset["time"].dt.hour

    era5_dataset["t2m_1w_rolling"] = (
        era5_dataset["t2m"].rolling(time=7 * 24, center=True).mean()
    )
    era5_dataset["mean_air_temperature"] = era5_dataset["t2m"].mean(dim="time")

    era5_dataset["mean_dewpoint_depression"] = era5_dataset["t2m"].mean(
        dim="time"
    ) - era5_dataset["d2m"].mean(dim="time")

    era5_dataset["dewpoint_depression_1w_rolling"] = (
        era5_dataset["t2m"].rolling(time=7 * 24, center=True).mean()
        - era5_dataset["d2m"].rolling(time=7 * 24, center=True).mean()
    )
    return era5_dataset


def compute_respiration(fluxnet_data: xr.Dataset) -> xr.Dataset:
    """Compute respiration from fluxnet GPP and NEE."""
    fluxnet_data = fluxnet_data.copy(deep=True)
    fluxnet_data["resp"] = fluxnet_data["GPP_NT_VUT_REF"] + fluxnet_data["NEE_VUT_REF"]
    fluxnet_data["resp"].attrs = {
        "units": "umolCO2 m-2 s-1",
        "long_name": "ecosystem_respiration",
    }
    fluxnet_data["GPP_NT_VUT_REF"].attrs = {
        "units": "umolCO2 m-2 s-1",
        "long_name": "gross_primary_production",
    }
    fluxnet_data["NEE_VUT_REF"].attrs = {
        "units": "umolCO2 m-2 s-1",
        "long_name": "net_ecosystem_exchange",
    }
    return fluxnet_data


def extract_sites_from_datasets(
    input_data: xr.Dataset, fluxnet_data: xr.Dataset
) -> xr.Dataset:
    """Extract Fluxnet site locations from input dataset."""
    # Shift to center of month to allow for more accurate interpolation.
    input_data["time"] = input_data["time"].copy(deep=True) + np.timedelta64(14, "D")

    additional_data = []
    for i_site in range(fluxnet_data["site"].size):
        ds_site = fluxnet_data.isel(site=i_site)
        site_data = input_data.sel(
            latitude=ds_site["latitude"],
            longitude=ds_site["longitude"],
            time=ds_site["time"],
            method="nearest",
        ).drop(["latitude", "longitude"])

        # Otherwise the original (monthly) dates are kept:
        site_data["time"] = fluxnet_data["time"]
        additional_data.append(site_data)

    additional_site_data = xr.concat(additional_data, dim="site")
    additional_site_data = additional_site_data.compute()
    return additional_site_data


def write_onnx(model: Any, n_predictors: int, fname: str | Path) -> None:
    """Convert the provided model to ONNX and write it to file."""
    fname = Path(fname)

    if "LGBM" in str(model):
        model_onnx = onnxmltools.convert_lightgbm(
            model, initial_types=[("X", DoubleTensorType([None, n_predictors]))]
        )
    else:
        model_onnx = convert_sklearn(
            model, initial_types=[("X", DoubleTensorType([None, n_predictors]))]
        )

    with fname.open(mode="wb") as f:
        f.write(model_onnx.SerializeToString())


def make_validation_plots(
    pycaret_setup: pycaret.regression.RegressionExperiment,
    pycaret_model: Any,
    output_dir: Path,
    plots: Iterable[str] = ("error", "feature", "residuals", "learning"),
) -> None:
    """Make pycaret model analysis plots."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for plotname in plots:
        dest = plots_dir / (plotname + ".png")
        file = Path(pycaret_setup.plot_model(pycaret_model, plot=plotname, save=True))
        file = file.resolve()
        shutil.move(file, dest)


def create_markdown_file(
    ds: xr.Dataset,
    x_keys: list[str],
    y_key: str,
    output_dir: Path,
) -> None:
    """Create a markdown file with a model description and validation plots.

    Args:
        ds: dataset containing all variables for training the model.
        predictions: list of predictions datasets per group.
        targets: list of target datasets per group.
        rmses: list of rmse dataarrays.
        x_keys: list of input variables.
        y_key: target variable name.
        output_dir: directory to write files to.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    long_names = [
        ds[var].attrs["long_name"] if "long_name" in ds[var].attrs else "undefined"
        for var in x_keys
    ]
    units = [
        ds[var].attrs["units"] if "units" in ds[var].attrs else "?" for var in x_keys
    ]

    text = """## Fluxnet model 
    
Description of model for fluxnet workflow.

### Model variables (in order):
"""

    for var, long_name, unit in zip(x_keys, long_names, units, strict=True):
        text += f"- **{var}**: {long_name} (`{unit}`)\n"

    text += "\n### Target variable:\n\n"
    target_name = ds[y_key].attrs["long_name"]
    target_units = ds[y_key].attrs["units"]
    text += f"- **{y_key}**: {target_name} ({target_units})\n"

    plots_dir = output_dir / "plots"
    if plots_dir.exists():
        plot_files = plots_dir.glob("*.png")

        text += "\n### Validation plots\n"
        for plot in plot_files:
            text += f"![]({str(plot.relative_to(output_dir))})\n"

    with open(output_dir / "model_description.md", "w") as file:
        file.write(text)


@dataclass
class FluxnetExperiment:
    """Parameters to train a ML model on Fluxnet data."""

    name: str
    X_keys: list[str]
    y_key: str
    ml_model_name: str
    cv_method: BaseCrossValidator
    cv_group_key: str
    output_dir: Path

    def __post_init__(self) -> None:
        """Set up the model's output directory."""
        self._init_time = datetime.now().strftime("%Y-%m-%d_%H_%M")
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_dir(self) -> Path:
        """Get the model's output directory."""
        return (
            self.output_dir
            / f"fluxnet_{self.name}-{self.ml_model_name}-{self._init_time}"
        )


def collect_training_data(
    fluxnet_file: str | Path,
    preprocessing_dir: str | Path,
    additional_datasets: list[str],
    variable_derivation: Callable[[xr.Dataset], xr.Dataset] | None = None,
) -> xr.Dataset:
    """Collect the training data into a single xarray Dataset.

    Args:
        fluxnet_file: Path to the preprocessed fluxnet netCDF file.
        preprocessing_dir: Directory to store preprocessing data in.
        additional_datasets: Which additional datasets you want to load.
        variable_derivation: (optional) function which derives additional variables from
            the combined dataset.

    Returns:
        A single dataset with all data merged together.
    """
    era5_files = list(PATHS_CFG["era5_hourly"].glob("*.nc"))
    ds_era5 = xr.open_mfdataset(era5_files)
    ds_grid = ds_era5[["latitude", "longitude"]].sortby(["latitude", "longitude"])

    ds_fluxnet = xr.open_dataset(fluxnet_file).compute()
    ds_fluxnet = compute_respiration(ds_fluxnet)

    fluxnet_site_extraction.preprocess_site_data(
        input_files=era5_files,
        out_dir=preprocessing_dir,
        fluxnet_file=fluxnet_file,
    )

    ds_era5_sites = xr.open_mfdataset(
        list(Path(preprocessing_dir).glob("fluxnet-sites_era5*.nc"))
    )

    # Load monthly data: hourly data will lead to memory issues
    input_data = xr.merge(
        [
            datasets[name].load(freq="monthly", target_grid=ds_grid)
            for name in additional_datasets
        ]
    )

    additional_data = extract_sites_from_datasets(input_data, ds_fluxnet)

    ds = xr.merge([ds_fluxnet, ds_era5_sites, additional_data])

    if variable_derivation is not None:
        ds = variable_derivation(ds)

    return ds


def run_workflow(
    fluxnet_file: str | Path,
    preprocessing_dir: str | Path,
    additional_datasets: list[str],
    models: Iterable[FluxnetExperiment] | FluxnetExperiment,
    variable_derivation: Callable[[xr.Dataset], xr.Dataset] | None = None,
    validation_plots: Iterable[str] = ("error", "feature", "residuals", "learning"),
) -> None:
    """Run the fluxnet workflow.

    Args:
        fluxnet_file: Path to the preprocessed fluxnet netCDF file.
        preprocessing_dir: Directory to store preprocessing data in.
        additional_datasets: Which additional datasets you want to load.
        models: The Fluxnet experiment(s) in the format of the FluxnetExperiment object.
        variable_derivation: (optional) function which derives additional variables from
            the combined dataset.
        validation_plots: Which pycaret validation plots to add to the markdown file.
    """
    if isinstance(models, FluxnetExperiment):
        models = [models]

    ds = collect_training_data(
        fluxnet_file,
        preprocessing_dir,
        additional_datasets,
        variable_derivation,
    )

    df_train = ds.to_dataframe().dropna().reset_index()

    for model in models:
        model.model_dir.mkdir(parents=True, exist_ok=True)

        pycs = pycaret.regression.setup(
            experiment_name=model.name,
            data=df_train[model.X_keys + [model.y_key, model.cv_group_key]],
            target=model.y_key,
            fold_strategy=model.cv_method,
            fold=model.cv_method.get_n_splits(),
            fold_groups=model.cv_group_key,
            ignore_features=[model.cv_group_key],
            verbose=False,
        )

        ml_model = pycs.compare_models(
            include=[model.ml_model_name],
            n_select=1,
            round=5,
        )

        write_onnx(
            ml_model,
            n_predictors=len(model.X_keys),
            fname=model.model_dir / f"{model.name}_{model.ml_model_name}.onnx",
        )
        write_model_variables(
            model.model_dir, model.X_keys, model.y_key, additional_datasets
        )
        make_validation_plots(
            pycs, ml_model, output_dir=model.model_dir, plots=validation_plots
        )
        create_markdown_file(
            ds,
            model.X_keys,
            model.y_key,
            model.model_dir,
        )
