"""EXCITED dataset protocol definition."""
from pathlib import Path
from typing import Literal
from typing import Protocol

import xarray as xr

from excited_workflow.config import PATHS_CFG


class DataSource(Protocol):
    """Source dataset protocol."""

    name: str
    variable_names: list[str]

    @classmethod
    def load(
        cls,
        freq: Literal["monthly", "hourly"],
        variables: list[str] | None = None,
        target_grid: xr.Dataset | None = None,
    ) -> xr.Dataset:
        """Load variables from this data source and regrid them to the target grid.

        Args:
            freq: Desired frequency of the dataset. Either "monthly" or "hourly".
            variables: List of variable names which should be downloaded.
            target_grid: Grid to which the data should be regridded to.

        Returns:
            Prepared dataset.
        """
        pass

    def validate_variables(self, variables: list[str] | None) -> None:
        """Check if the requested variables are all valid for this dataset."""
        if variables is None:
            return None
        if not set(variables).issubset(self.variable_names):
            msg = "One or more requested variables are not valid for this dataset."
            raise KeyError(msg)

    def get_path(self) -> Path:
        """Returns the path to the folder containing this dataset's data."""
        return PATHS_CFG[self.name]


def get_freq_kw(freq: Literal["hourly", "monthly"]) -> Literal["1H", "1MS"]:
    """Get the frequency keyword corresponding to the hourly/monthly resampling freq.

    Returns:
        Pandas DateOffset string.
    """
    if freq == "hourly":
        return "1H"
    elif freq == "monthly":
        return "1MS"
    else:
        msg = (
            "Invalid value for kwarg 'freq': '{freq}'.\n"
            "Only 'hourly' and 'monthly' are allowed."
        )
        raise ValueError(msg)
