"""Ingest Copernicus land cover data."""
from typing import Literal

import numpy as np
import xarray as xr
import xarray_regrid  # noqa: F401

from pathlib import Path

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

class LandCover(DataSource):
    """Copernicus land cover dataset."""

    name: str = "copernicus_landcover"
    variable_names: list[str] = ["lccs_class"]

    def makedir(self) -> str:
        """Create preprocessing directory if it does not already exist.

        Returns:
            String of preprocess directory path
        """

        data_path = str(Path(self.get_path()))  + "/preprocessing"
        Path(data_path).mkdir(parents=True, exist_ok=True)

        return data_path
    
    def regrid(        
        self,
        file,
        variables: list[str] | None = None,
        target_grid: xr.Dataset | None = None,
    ) -> xr.Dataset:
        """Regrids one file to target dataset.

        Args: 
            file
            variables: List of variable names which should be downloaded.
            target_grid: Grid to which the data should be regridded to.

        Returns:
            Regridded Dataset.
        """
        
        ds = xr.open_dataset(file, chunks={"lat": 2000, "lon": 2000})

        # Set time to middle of bounds.
        time_coords = _cftime_to_datetime(ds["time_bounds"].mean(dim="bounds"))
        ds = ds.drop("time_bounds")
        ds["time"] = time_coords

        ds = ds[["lccs_class"]]  # Only take the class variable.
        ds = ds.sortby(["lat", "lon"])
        ds = ds.rename({"lat": "latitude", "lon": "longitude"})

        if variables is not None:
            ds = ds[variables]
        if target_grid is not None:
            ds = ds.regrid.most_common(target_grid, time_dim="time")

        return ds


    def preprocess(
        self,
        process_path: str,
        variables: list[str] | None = None,
        target_grid: xr.Dataset | None = None,
    ) -> xr.Dataset:
        """Load variables from this data source and regrid them to the target grid
        and save to preprocessing directory. 

        Args:
            variables: List of variable names which should be downloaded.
            target_grid: Grid to which the data should be regridded to.

        Returns:
            Preprocessed regridded netcdf files.
        """

        self.validate_variables(variables)

        files = list(self.get_path().glob("*.nc"))
        if len(files) == 0:
            msg = f"No netCDF files found at path '{self.get_path()}'"
            raise FileNotFoundError(msg)

        for f in files: 
            name = process_path + "/" + str(Path(f).stem) + ".nc"

            if Path(name).is_file() and Path(name).stat().st_size > 0:
                dat = xr.open_dataset(name)
                if dat.latitude.size == target_grid.latitude.size and dat.longitude.size == target_grid.longitude.size:
                    pass
                else:
                    print("Not same size")
                    dat.close()
                    ds = self.regrid(f, variables, target_grid)
                    ds.to_netcdf(name)
            else:
                ds = self.regrid(f, variables, target_grid)
                ds.to_netcdf(name)


    def load(
        self,
        freq: Literal["hourly", "monthly"],
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

        path = self.makedir()
        self.preprocess(path, variables, target_grid)

        files = list(Path(path).glob("*.nc"))
        if len(files) == 0:
            msg = f"No netCDF files found at path '{path}'"
            raise FileNotFoundError(msg)

        freq_kw = get_freq_kw(freq)
        ds = xr.open_mfdataset(files, chunks={"lat": 2000, "lon": 2000})
        ds = ds.resample(time=freq_kw).interpolate("nearest")
        
        return ds
