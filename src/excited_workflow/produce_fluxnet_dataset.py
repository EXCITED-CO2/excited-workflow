"""Functions for producing a dataset from a Fluxnet-based ML model."""
from copy import copy
from pathlib import Path
from time import time

import numpy as np
import onnxruntime
import pandas as pd
import xarray as xr

from excited_workflow.common_utils import read_model_variables
from excited_workflow.config import PATHS_CFG
from excited_workflow.source_datasets import datasets
from excited_workflow.train_fluxnet_models import calculate_era5_derived_vars


def load_era5(X_keys: list[str]) -> xr.Dataset:  # noqa: N803
    """Load the specified ERA5 variables into a dataset."""
    era5_files = list(PATHS_CFG["era5_hourly"].glob("*.nc"))

    ds_seperate = [xr.open_dataset(file, chunks={"time": 1440}) for file in era5_files]
    ds_era5 = xr.combine_by_coords(ds_seperate, combine_attrs="drop")
    assert isinstance(ds_era5, xr.Dataset)  # combine_by_coords misses typing overload

    keys = list(ds_era5.data_vars)

    # these vars don't have all years available which messes up the chunking.
    keys.remove("ssrd")
    keys.remove("strd")
    ds_era5 = ds_era5[keys]

    ds_era5 = calculate_era5_derived_vars(ds_era5)

    era5_keys = set(ds_era5.data_vars).intersection(X_keys)
    ds_era5 = ds_era5[era5_keys]

    return ds_era5


def infer_start_end_time(
    ds_era5: xr.Dataset, additional_data: list[xr.Dataset]
) -> tuple[np.datetime64, np.datetime64]:
    """Infer the start end end times from the input data."""
    all_data = copy(additional_data)
    all_data.append(ds_era5)
    start_time = np.max([ds.isel(time=0)["time"].to_numpy() for ds in all_data])
    end_time = np.min([ds.isel(time=-1)["time"].to_numpy() for ds in all_data])
    return start_time, end_time


def predict(onnx_file: Path, input_data: np.ndarray) -> np.ndarray:
    """Make a prediction using an ONNX model."""
    session = onnxruntime.InferenceSession(onnx_file)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    onnx_input = {input_name: input_data}
    pred_onnx = session.run([output_name], onnx_input)[0]
    return pred_onnx  # type: ignore


def produce_dataset(
    model_dir: Path | str,
    output_dir: Path | str,
) -> None:
    """Predict using the specified trained model, and write results to the output_dir.

    Args:
        model_dir: Directory of a previously trained model. Must contain the ONNX file
            as well as the model variables .json file.
        output_dir: The directory to which the dataset files should be written to.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_keys, y_key, required_datasets = read_model_variables(model_dir)  # noqa: N806
    onnx_file = next(Path(model_dir).glob("*.onnx"))

    ds_era5 = load_era5(X_keys)

    ds_grid = ds_era5[["latitude", "longitude"]].sortby(["latitude", "longitude"])

    additional_data = [
        datasets[name].load(freq="monthly", target_grid=ds_grid)
        for name in required_datasets
    ]

    start, end = infer_start_end_time(ds_era5, additional_data)
    dates = pd.date_range(start, end, freq="1MS")
    for start, end in zip(dates[0:-1], dates[1:], strict=True):
        timeslice = slice(start, end)
        print(f"Loading data for {start} to {end}...")
        t0 = time()
        additional_data_computed = [
            data.sel(time=timeslice).compute() for data in additional_data
        ]
        era5_computed = ds_era5.sel(time=timeslice).compute()

        additional_data_resampled = [
            data.resample(time="1H").interpolate() for data in additional_data_computed
        ]

        all_data = xr.merge([era5_computed] + additional_data_resampled)
        original_coords = all_data[["longitude", "latitude", "time"]]

        df_input = all_data[X_keys].to_dataframe().dropna()
        del all_data

        if df_input.empty:
            print("Some variables have no data for this time slice. Skipping...")
        else:
            # split data to not reach memory limit of onnx prediction.
            splits = np.array_split(df_input, 8)
            output_data = []
            for split in splits:
                output_data.append(predict(onnx_file, split.to_numpy()))  # type: ignore
            del splits
            output_data = np.concatenate(output_data)

            df_input[y_key] = output_data
            data_out = df_input[[y_key]].to_xarray()
            del df_input

            # ensure data is sorted:
            data_out = data_out.sortby(["longitude", "latitude", "time"])
            # ensure new coords match old ones (i.e. that none are dropped)
            data_out = xr.align(original_coords, data_out, join="outer")[1]

            print(f"Processing took {time()-t0:.0f} seconds.")
            print("Writing prediction to file...")
            data_out.to_netcdf(
                output_dir / f"{y_key}_{start.year}-{start.month}-{start.day}.nc"  # type: ignore
            )
            del data_out
