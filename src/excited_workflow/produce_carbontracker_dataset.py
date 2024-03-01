"""Create NEE datasets from CarbonTracker model."""

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr
from onnxruntime import InferenceSession

import excited_workflow
from excited_workflow.common_utils import read_model_variables
from excited_workflow.source_datasets import datasets


def run_model(onnx_model: Path, df: pd.DataFrame, X_keys: list[str]) -> Any:
    """Open model and run it.

    Args:
        onnx_model: path to model file.
        df: input dataframe for running model.
        X_keys: list of variables required for model.

    Returns:
        Array of predictions.
    """
    with onnx_model.open(mode="rb") as f:
        model = f.read()

    sess = InferenceSession(model)
    predictions_onnx = sess.run(None, {"X": df[X_keys].to_numpy()})[0]

    return predictions_onnx


def predict(
    ds_merge: xr.Dataset,
    onnx_model: Path,
    X_keys: list[str],
) -> list[pd.DataFrame]:
    """Create NEE monthly dataset.

    Args:
        ds_merge: input dataset.
        onnx_model: path to model file.
        X_keys: list of variables required for model.

    Returns:
        list of dataframes with model predictions.
    """
    allnan = ds_merge.isnull().all(dim=["latitude", "longitude"]).compute()

    dfs = []
    for idx, _dtime in enumerate(ds_merge["time"]):
        ds_sel = ds_merge.isel(time=[idx])
        if not any([allnan.isel(time=idx)[var] for var in allnan.data_vars]):
            ds_sel = ds_sel.compute()
            ds_na = ds_sel.where(ds_merge["transcom_regions"] == 2)
            ds_land = ds_na.where(ds_na["lccs_class"] != 210)
            df_land = ds_land.to_dataframe().dropna()
            prediction = run_model(onnx_model, df_land, X_keys)
            dfs.append(
                pd.DataFrame(data=prediction, index=df_land.index, columns=["bio_flux"])
            )

    return dfs  #


def produce_dataset(
    model_dir: Path,
    model_name: str,
    carbontracker_file: Path,
    regions_file: Path,
    data_dir: Path,
) -> Any:
    """Create dataset for predictions over entire time period.

    Args:
        model_dir: path to directory containing onnx model.
        model_name: name of onnx model.
        carbontracker_file: path to carbontracker file.
        regions_file: path to regions file.
        data_dir: path to data directory.
    """
    predictors, target, required_datasets = read_model_variables(model_dir)
    X_keys = list(predictors)  # noqa: N806  (Get the var names only)

    ds_cb = xr.open_dataset(carbontracker_file)
    ds_cb = excited_workflow.utils.convert_timestamps(ds_cb)
    ds_regions = xr.open_dataset(regions_file)
    ds_input = xr.merge(
        [
            datasets[name].load(freq="monthly", target_grid=ds_cb)
            for name in required_datasets
        ]
    )

    dsx = ds_input[X_keys]
    ds_merge = xr.merge([dsx, ds_regions["transcom_regions"]])
    onnx_model = model_dir / model_name

    dfs = predict(ds_merge, onnx_model, X_keys)

    dss = [df.to_xarray().sortby(["latitude", "longitude"]) for df in dfs]
    ds_out = xr.concat(dss, dim="time")
    ds_out["bio_flux"].attrs = {
        "long_name": "terrestrial biosphere CO2 flux",
        "units": "mol m-2 s-1",
    }
    ds_out.attrs = {
        "title": "CarbonTracker model monthly dataset",
        "version": "1",
        "institution": "Utrecht University, Netherlands eScience Center",
        "history": f"Date created: {datetime.now().strftime('%Y-%m-%d')}",
        "workflow_version": excited_workflow.version,
        "workflow_source": "https://github.com/EXCITED-CO2/excited-workflow/",
        "predictor_variables": X_keys,
    }
    ds_out = xr.Dataset(ds_out)

    output_dir = data_dir / "NEE/CarbonTracker"
    output_dir.mkdir(parents=True, exist_ok=True)

    comp = dict(zlib=True, complevel=8)
    encoding = {var: comp for var in ds_out.data_vars}
    ds_out.to_netcdf(output_dir / "NEE_monthly.nc", encoding=encoding)
