"""Config handler.

Config files will be searched for in directories in the following order:
    1. $XDG_CONFIG_HOME/excited/
    2. ~/.config/excited/
    3. /etc/excited/

By placing the configuration files in /etc/excited/, the workflow can be configured
    for a shared system such as Surf Research Cloud.
    
If a config file is not found in any of these directories, an error will be raised.

The following config files are required for running the workflow:
    1. data_paths.yaml
    2. dask.yaml (TODO)
"""
import os
from pathlib import Path

import yaml


class ConfigError(Exception):
    """Raised when an error in the config is found."""


SYSTEM_CONFIG_DIR = Path("/etc") / "excited"
USER_HOME_CONFIG_DIR: Path = (
    Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "excited"
)
DATA_PATHS_FNAME = "data_paths.yaml"
DASK_FNAME = "dask.yaml"


EXPECTED_DATASETS = [
    "biomass",
    "copernicus_landcover",
    "era5_hourly",
    "era5_monthly",
    "era5_land_monthly",
    "modis",
    "spei",
]


def _find_config(fname: str) -> Path:
    """Return the path to the config file if found. Else raises a FileNotFoundError."""
    if (USER_HOME_CONFIG_DIR / fname).exists():
        return USER_HOME_CONFIG_DIR / fname
    elif (SYSTEM_CONFIG_DIR / fname).exists():
        return SYSTEM_CONFIG_DIR / fname
    else:
        msg = (
            f"The config file '{fname}' was not found."
            "Please look at the documentation on how to create the config files."
        )
        raise FileNotFoundError(msg)


def load_paths_config() -> dict[str, Path]:
    """Load the Zampy data source paths."""
    with _find_config(DATA_PATHS_FNAME).open(mode="r") as f:
        paths_config: dict = yaml.full_load(f.read())

    if not set(EXPECTED_DATASETS).issubset(paths_config.keys()):
        missing_keys = set(EXPECTED_DATASETS) - set(paths_config.keys())
        msg = f"Some dataset paths are missing from the config: {missing_keys}"
        raise ConfigError(msg)

    return {key: Path(val) for key, val in paths_config.items()}


PATHS_CFG = load_paths_config()
