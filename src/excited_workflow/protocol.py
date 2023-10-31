"""EXCITED dataset protocol definition."""
from pathlib import Path
from typing import Protocol

import xarray as xr

from excited_workflow.config import PATHS_CFG


class DataSource(Protocol):
    """Source dataset protocol."""

    name: str
    variable_names: list[str]

    @classmethod
    def load(
        cls, variables: list[str] | None = None, target_grid: xr.Dataset | None = None
    ) -> xr.Dataset:
        """Load variables from this data source and regrid them to the target grid.

        Args:
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
