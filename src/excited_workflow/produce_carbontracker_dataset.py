"""Create NEE datasets from CarbonTracker model."""
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr
from onnxruntime import InferenceSession

import excited_workflow


def run_model(onnx_model: Path, df: pd.DataFrame, x_keys: list[str]) -> Any:
    """Open model and run it.

    Args:
        onnx_model: path to model file.
        df: input dataframe for running model.
        x_keys: list of variables required for model.

    Returns:
        Array of predictions.
    """
    with onnx_model.open(mode="rb") as f:
        model = f.read()

    sess = InferenceSession(model)
    predictions_onnx = sess.run(None, {"X": df[x_keys].to_numpy()})[0]

    return predictions_onnx


def get_predictions(
    ds_input: xr.Dataset, x_keys: list[str], ds_regions: xr.Dataset, model_dir: Path
) -> list[pd.DataFrame]:
    """Create NEE monthly dataset.

    Args:
        ds_input: input dataset.
        x_keys: list of variables required for model.
        ds_regions: dataset of regions.
        model_dir: path to model directory.
    """
    dsx = ds_input[x_keys]
    ds_merge = xr.merge([dsx, ds_regions["transcom_regions"]])
    allnan = ds_merge.isnull().all(dim=["latitude", "longitude"]).compute()

    dfs = []
    for idx, _dtime in enumerate(ds_merge["time"]):
        ds_sel = ds_merge.isel(time=[idx])
        if not any([allnan.isel(time=idx)[var] for var in allnan.data_vars]):
            ds_sel = ds_sel.compute()
            ds_na = ds_sel.where(ds_merge["transcom_regions"] == 2)
            ds_land = ds_na.where(ds_na["lccs_class"] != 210)
            df_land = ds_land.to_dataframe().dropna()
            prediction = run_model(model_dir, df_land, x_keys)
            dfs.append(
                pd.DataFrame(data=prediction, index=df_land.index, columns=["bio_flux"])
            )

    return dfs  #


def create_dataset(dfs: list[pd.DataFrame], data_dir: Path, x_keys: list[str]) -> Any:
    """Create dataset for predictions over entire time period.

    Args:
        dfs: list of prediction dataframes.
        data_dir: path to data directory.
        x_keys: list of variables required for model.

    Returns:
        Output NEE dataset.
    """
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
        "predictor_variables": x_keys,
    }
    ds_out = xr.Dataset(ds_out)

    output_dir = data_dir / "NEE/CarbonTracker"
    output_dir.mkdir(parents=True, exist_ok=True)

    comp = dict(zlib=True, complevel=8)
    encoding = {var: comp for var in ds_out.data_vars}
    ds_out.to_netcdf(output_dir / "NEE_monthly.nc", encoding=encoding)

    return ds_out
