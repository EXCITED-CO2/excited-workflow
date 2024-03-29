"""EXCITED dataset protocol definition."""

import typing
from abc import abstractmethod
from pathlib import Path
from typing import Literal
from typing import Protocol

import xarray as xr

from excited_workflow.config import PATHS_CFG


class DataSource(Protocol):
    """Source dataset protocol."""

    name: str
    variable_names: list[str]

    def __init__(self) -> None:
        """Initialize the datasource."""
        return None

    @abstractmethod
    def load(
        self,
        freq: Literal["monthly", "hourly"] | None = None,
        variables: list[str] | None = None,
        target_grid: xr.Dataset | None = None,
    ) -> xr.Dataset:
        """Load variables from this data source and regrid them to the target grid.

        Args:
            freq: Desired frequency of the dataset. Either "monthly", "hourly", or None.
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
        """Return the path to the folder containing this dataset's data."""
        return PATHS_CFG[self.name]


@typing.overload
def get_freq_kw(
    freq: Literal["hourly", "monthly"],
) -> Literal["1H", "1MS"]: ...


@typing.overload
def get_freq_kw(
    freq: None,
) -> None: ...


def get_freq_kw(
    freq: Literal["hourly", "monthly"] | None,
) -> Literal["1H", "1MS"] | None:
    """Get the frequency keyword corresponding to the hourly/monthly resampling freq.

    Returns:
        Pandas DateOffset string.
    """
    if freq == "hourly":
        return "1H"
    elif freq == "monthly":
        return "1MS"
    elif freq is None:
        return None
    else:
        msg = (
            "Invalid value for kwarg 'freq': '{freq}'.\n"
            "Only 'hourly' and 'monthly' are allowed."
        )
        raise ValueError(msg)
