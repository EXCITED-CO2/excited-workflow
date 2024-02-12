"""Module for extracting fluxnet sites from global data."""
from pathlib import Path

import numpy as np
import xarray as xr


def contains_all_sites(ncfile: Path, da_sitenames: xr.DataArray) -> bool:
    """Check if the existing preprocessed file contains all sites in the Fluxnet data.

    Args:
        ncfile: Path to the existing preprocessed file.
        da_sitenames: DataArray of the site names.

    Returns:
        If the preprocessed file contains all sites in da_sitenames.
    """
    ds = xr.open_dataset(ncfile, chunks="auto")
    preprocessed_sites = set(ds["site"].to_numpy())
    input_sites = set(da_sitenames.to_numpy())

    return input_sites.issubset(preprocessed_sites)


def extract_site_data(input_file: Path, output_file: Path, ds_flux: xr.Dataset) -> None:
    """Extract & write away data for each site from the source data.

    Args:
        input_file: File containing the source data.
        output_file: File to which the (merged) extracted data is written to.
        ds_flux: Dataset containing the sites and their locations.
    """
    n_sites = ds_flux["site"].size

    # Loading in the whole file is 3x faster than lazily loading it in (!)
    ds = xr.open_dataset(input_file).load()
    site_data = [xr.Dataset()] * n_sites
    for i_site in range(n_sites):
        ds_site = ds.sel(
            latitude=ds_flux.isel(site=i_site)["latitude"],
            longitude=ds_flux.isel(site=i_site)["longitude"],
            method="nearest",
        )
        site_data[i_site] = ds_site.drop(["latitude", "longitude"])
    data_out = xr.concat(site_data, dim="site")
    data_out.to_netcdf(output_file)
    ds.close()


def preprocess_site_data(
    input_files: list[Path],
    out_dir: str | Path,
    fluxnet_file: str | Path,
    verbose: bool = True,
) -> None:
    """Preprocess the input files by extracting the locations of every fluxnet site.

    Args:
        input_files: netCDF files from which the location data should be extracted to.
            Note that these input files should have longitudes ranging from -180 to 180.
        out_dir: Directory to which the preprocessed files are written to.
        fluxnet_file: Fluxnet netCDF file containing the sites and their locations.
        verbose: If info on the preprocessing should be printed to the terminal.
    """
    Path(out_dir).mkdir(exist_ok=True)

    ds_ameriflux = xr.open_dataset(fluxnet_file)

    for i_ds, file in enumerate(input_files):
        data_name = f"fluxnet-sites_{input_files[i_ds].name.split('_hourly')[0]}.nc"
        out_file = Path(out_dir) / data_name
        if out_file.exists() and contains_all_sites(out_file, ds_ameriflux["site"]):
            if verbose:
                print(
                    f"Valid file {out_file.name} already exists, skipping.",
                    " " * 12,  # Make sure line is cleared.
                    end="\r",
                )
        else:
            if verbose:
                print(
                    f"Processing site {i_ds+1}/{len(input_files)} - {out_file.name}",
                    " " * 12,  # Make sure line is cleared.
                    end="\r",
                )
            extract_site_data(file, out_file, ds_ameriflux)


def extract_sites_from_datasets(
    input_data: xr.Dataset, fluxnet_data: xr.Dataset
) -> xr.Dataset:
    """Extract Fluxnet site locations from monthly datasets.
    
    Takes and returns xarray Datasets (not files), as it's sufficiently fast.
    """
    # Shift to center of month to allow for more accurate interpolation.
    input_data["time"] = input_data["time"].copy(deep=True) + np.timedelta64(14, "D")

    additional_data = []
    for i_site in range(fluxnet_data["site"].size):
        ds_site = fluxnet_data.isel(site=i_site)
        site_data = input_data.sel(
            latitude=ds_site["latitude"],
            longitude=ds_site["longitude"],
            time=ds_site["time"],
            method="nearest",
        ).drop(["latitude", "longitude"])

        # Otherwise the original (monthly) dates are kept:
        site_data["time"] = fluxnet_data["time"]
        additional_data.append(site_data)

    additional_site_data = xr.concat(additional_data, dim="site")
    additional_site_data = additional_site_data.compute()
    return additional_site_data
