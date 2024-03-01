"""Ingest Copernicus land cover data."""

from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr
import xarray_regrid  # noqa: F401

from excited_workflow.source_datasets.protocol import DataSource
from excited_workflow.source_datasets.protocol import get_freq_kw


def _cftime_to_datetime(data: xr.DataArray) -> np.ndarray:
    """Convert cftime dataarray values to a numpy datetime format.

    Args:
        data: DataArray containing the time values, e.g. ds["time"].

    Returns:
        Numpy array with a datetime64 dtype.
    """
    return np.array([np.datetime64(el) for el in data.to_numpy()])


def regrid(
    file: Path,
    target_grid: xr.Dataset,
    variables: list[str] | None = None,
) -> xr.Dataset:
    """Regrid a land cover netCDF file to a target dataset and return regridded dataset.

    Args:
        file: file to regrid
        target_grid: Dataset to which the data should be regridded to.
        variables: List of variable names which should be regridded.

    Returns:
        Regridded Dataset.
    """
    ds = xr.open_dataset(file, chunks={"lat": 2000, "lon": 2000})

    # Set time to middle of bounds.
    time_coords = _cftime_to_datetime(ds["time_bounds"].mean(dim="bounds"))
    ds = ds.drop("time_bounds")
    ds["time"] = time_coords

    ds = ds[["lccs_class"]]
    ds = ds.sortby(["lat", "lon"])
    ds = ds.rename({"lat": "latitude", "lon": "longitude"})

    if variables is not None:
        ds = ds[variables]

    ds = ds.regrid.most_common(target_grid, time_dim="time")

    return ds


def coord_matches(dat: xr.Dataset, target: xr.Dataset, coord: str) -> bool:
    """Check coordinates of two grids match.

    Args:
        dat: Preprocessed dataset
        target: Target grid dataset
        coord: variable name to compare

    Returns:
        boolean of where arrays are the same.
    """
    if dat[coord].size == target[coord].size:
        return np.allclose(dat[coord].to_numpy(), target[coord].to_numpy())

    return False


def valid_regrid_file(name: Path, target: xr.Dataset) -> bool:
    """Check if file has been regridded properly.

    Args:
        name: name of file
        target: Target grid dataset

    Returns:
        boolean of whether the dimensions are the same as target.
    """
    dat = xr.open_dataset(name)
    if coord_matches(dat, target, "latitude") and coord_matches(
        dat, target, "longitude"
    ):
        dat.close()
        return True
    return False


class LandCover(DataSource):
    """Copernicus land cover dataset."""

    name: str = "copernicus_landcover"
    variable_names: list[str] = ["lccs_class"]

    def preprocess(
        self,
        process_path: Path,
        target_grid: xr.Dataset,
        variables: list[str] | None = None,
    ) -> None:
        """Preprocess variable.

        Args:
            process_path: Output path for preprocessed files.
            variables: List of variable names which should be downloaded.
            target_grid: Grid to which the data should be regridded to.
        """
        self.validate_variables(variables)

        files = list(self.get_path().glob("*.nc"))
        if len(files) == 0:
            msg = f"No netCDF files found at path '{self.get_path()}'"
            raise FileNotFoundError(msg)

        for file in files:
            name = process_path / (file.name)

            if not name.is_file() or not valid_regrid_file(name, target_grid):
                print(
                    f"'{file.name}' not found or not valid for target dataset. "
                    f"Regridding."
                )
                regrid(file, target_grid, variables).to_netcdf(name)

    def load(
        self,
        freq: Literal["hourly", "monthly"] | None = None,
        variables: list[str] | None = None,
        target_grid: xr.Dataset | None = None,
    ) -> xr.Dataset:
        """Load variables from this data source and regrid them to the target grid.

        Args:
            freq: Desired frequency of the dataset. Either "hourly" or "monthly".
            variables: List of variable names which should be downloaded.
            target_grid: Grid to which the data should be regridded to.

        Returns:
            Prepared dataset.
        """
        self.validate_variables(variables)

        if target_grid is None:
            msg = "target_grid is not optional for loading landcover data."
            raise ValueError(msg)

        preprocessed_dir = self.get_path() / "preprocessed"
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        self.preprocess(preprocessed_dir, target_grid, variables)

        files = list(preprocessed_dir.glob("*.nc"))
        if len(files) == 0:
            msg = f"No netCDF files found at path '{preprocessed_dir}'"
            raise FileNotFoundError(msg)

        ds = xr.open_mfdataset(files, chunks={"lat": 2000, "lon": 2000})
        if freq is not None:
            ds = ds.resample(time=get_freq_kw(freq)).interpolate("nearest")

        return ds
