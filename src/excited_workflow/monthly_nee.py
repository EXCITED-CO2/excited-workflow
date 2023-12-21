"""Create NEE datasets from CarbonTracker model."""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import Client
from onnxruntime import InferenceSession


client = Client()


def run_model(model_dir: Path, df: pd.DataFrame, x_keys: list[str]) -> xr.DataArray:
    """Open model and run it.

    Args:
        model_dir: path to model directory.
        df: input dataframe for running model.
        x_keys: list of variables required for model.

    Returns:
        Array of predictions.
    """
    with open(model_dir / "lightgbm.onnx", "rb") as f:
        model = f.read()

    sess = InferenceSession(model)
    predictions_onnx = sess.run(None, {"X": df[x_keys].to_numpy()})[0]

    return predictions_onnx


def get_predictions(
    ds_input: xr.Dataset, x_keys: list[str], ds_regions: xr.Dataset, model_dir: Path
):
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

    predictions = []
    times = []
    for idx, dtime in enumerate(ds_merge["time"]):
        ds_sel = ds_merge.isel(time=idx)
        if not any([allnan.isel(time=idx)[var] for var in allnan.data_vars]):
            ds_sel = ds_sel.compute()
            ds_na = ds_sel.where(ds_merge["transcom_regions"] == 2)
            df_sel = ds_na.to_dataframe().dropna()

            predictions.append(run_model(model_dir, df_sel, x_keys))
            times.append(dtime.to_numpy())

    dfs = [
        pd.DataFrame(data=pred, index=df_sel.index, columns=["bio_flux"])
        for pred in predictions
    ]

    return dfs, times


def create_dataset(dfs: list[pd.DataFrame], times: list[str], data_dir: Path) -> None:
    """Create dataset for predictions over entire time period.

    Args:
        dfs: list of prediction dataframes.
        times: list of times.
        data_dir: path to data directory.

    Returns:
        Output NEE dataset.
    """
    dss = [df.to_xarray().sortby(["latitude", "longitude"]) for df in dfs]
    ds_out = xr.concat(dss, dim="time")
    ds_out["time"] = np.array(times)
    ds_out["bio_flux"].attrs = {
        "long_name": "terrestrial biosphere CO2 flux",
        "units": "mol m-2 s-1",
    }
    ds_out.attrs = {
        "title": "CarbonTracker model monthly dataset",
        "version": "1",
        "institution": "Netherlands eScience Center",
        "history": f"Date created: {datetime.now().strftime('%Y-%m-%d')}",
    }

    output_dir = data_dir / "NEE/CarbonTracker"
    output_dir.mkdir(parents=True, exist_ok=True)

    comp = dict(zlib=True, complevel=8)
    encoding = {var: comp for var in ds_out.data_vars}
    ds_out.to_netcdf(output_dir / "NEE_monthly.nc", encoding=encoding)
