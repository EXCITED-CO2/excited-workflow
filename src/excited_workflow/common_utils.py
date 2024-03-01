"""Utility functions shared by the Fluxnet and CarbonTracker workflows."""

import json
from pathlib import Path

import xarray as xr


def get_attrs(ds: xr.Dataset, var: str) -> dict[str, str]:
    """Extract the attributes of a variable in an xarray dataset.

    Args:
        ds: Xarray Dataset
        var: Variable name

    Returns:
        Dictionary with the attributes long_name and units.
    """
    return {
        "long_name": (
            ds[var].attrs["long_name"] if "long_name" in ds[var].attrs else "undefined"
        ),
        "units": ds[var].attrs["units"] if "units" in ds[var].attrs else "?",
    }


def write_model_variables(
    output_directory: Path | str,
    dataset: xr.Dataset,
    X_keys: list[str],  # noqa: N803
    y_key: str,
    included_datasets: list[str],
) -> None:
    """Write the ML model's variables to a human- and python-readable file.

    Args:
        output_directory: Output directory in which to write the model vars.
        dataset: The dataset containing all of the model's input vars.
        X_keys: The predictor variables of the model.
        y_key: The target/ouput variable of the model.
        included_datasets: Which additional datasets were included in the training.
    """
    predictors = {var: get_attrs(dataset, var) for var in X_keys}
    target = {y_key: get_attrs(dataset, y_key)}

    with (Path(output_directory) / "model_variables.json").open("w") as f:
        json.dump(
            {
                "predictors": predictors,
                "target": target,
                "included_datasets": included_datasets,
            },
            f,
        )


def read_model_variables(
    output_directory: Path | str,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], list[str]]:
    """Read the ML model's from file.

    Args:
        output_directory: Output directory from which to read the model vars.

    Returns:
        predictors: The predictor variable names of the model and their attributes.
        target: The target/ouput variable name of the model and its attributes.
        included_datasets: Which additional datasets were included in the training.
    """
    with (Path(output_directory) / "model_variables.json").open("r") as f:
        model_vars = json.load(f)
    return (
        model_vars["predictors"],
        model_vars["target"],
        model_vars["included_datasets"],
    )
