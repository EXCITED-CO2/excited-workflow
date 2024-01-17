"""Utility functions shared by the Fluxnet and CarbonTracker workflows."""
import json
from pathlib import Path


def write_model_variables(
    output_directory: Path | str,
    X_keys: list[str],  # noqa: N803
    y_key: str,
    included_datasets: list[str],
) -> None:
    """Write the ML model's variables to a human- and python-readable file.

    Args:
        output_directory: Output directory in which to write the model vars.
        X_keys: The predictor variables of the model.
        y_key: The target/ouput variable of the model.
        included_datasets: Which additional datasets were included in the training.
    """
    with (Path(output_directory) / "model_variables.json").open("w") as f:
        json.dump(
            {
                "X_keys": X_keys,
                "y_key": y_key,
                "included_datasets": included_datasets,
            },
            f,
        )


def read_model_variables(
    output_directory: Path | str,
) -> tuple[list[str], str, list[str]]:
    """Read the ML model's from file.

    Args:
        output_directory: Output directory from which to read the model vars.

    Returns:
        X_keys: The predictor variables of the model.
        y_key: The target/ouput variable of the model.
        included_datasets: Which additional datasets were included in the training.
    """
    with (Path(output_directory) / "model_variables.json").open("r") as f:
        model_vars = json.load(f)
    return model_vars["X_keys"], model_vars["y_key"], model_vars["included_datasets"]
