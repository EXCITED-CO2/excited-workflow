"""Routines to extract per site data from ERA5."""

from pathlib import Path

import pandas as pd
import xarray as xr


def extract_per_site_era5_data(
    preprocessed_ameriflux_data: Path,
    era5_data_folder: Path,
    output_folder: Path,
) -> None:
    """Extract ERA5 data from all gidcells matching each flux site, and write to file.

    The ERA5 data will be matched with the longitude and latitude of each Ameriflux
    site, and the data corresponding to this cell will be extracted and written to file.

    Data is written as netCDF files to the specified output folder, and is split up
    per site and per variable. E.g., "ERA5_sp_US-ARM.nc" will contain the ERA5 surface
    pressure for the US-ARM site.
    If a file already exists, it is skipped. This allows for efficiently adding
    additional variables without recomputing everything.

    Args:
        preprocessed_ameriflux_data: Preprocessed Ameriflux dataset, containing the
            sites as a dimension.
        era5_data_folder: Folder containing the unprocessed ERA5 netCDF files.
        output_folder: Folder to which the extracted data should be written to.
    """
    ds_sites = xr.open_dataset(
        preprocessed_ameriflux_data,
        chunks={"time": -1, "site": 1},
    )

    ds_era5 = xr.open_mfdataset(
        list(era5_data_folder.glob("*.nc")),
        parallel=True,
        chunks={"time": -1, "latitude": 1, "longitude": 1},  # Rechunk for performance
    )
    era5_variables = list(ds_era5.data_vars)

    for i_site in range(len(ds_sites["site"])):
        ds_site = ds_sites.isel(site=[i_site])
        ds_era5_site = ds_era5.sel(
            latitude=ds_site["latitude"].values,
            longitude=ds_site["longitude"].values,
            method="nearest",
            tolerance=1,
        )

        for era5_var in era5_variables:
            fpath = output_folder / f"ERA5_{era5_var}_{ds_site['site'].values[0]}.nc"

            if not fpath.exists():
                print(
                    f"\rExtracting variable '{era5_var}' for site "
                    f"'{ds_site['site'].values[0]}'"
                    "                 ",
                    flush=True,
                    end="",
                )
                ds_era5_var = ds_era5_site[[era5_var]]
                ds_era5_var = ds_era5_var.drop(["latitude", "longitude"])
                ds_era5_var = ds_era5_var.expand_dims("site")
                ds_era5_var = ds_era5_var.isel(latitude=0, longitude=0)
                ds_era5_var["site"] = (("site",), ds_site["site"].values)
                ds_era5_var.to_netcdf(fpath)
            else:
                print(
                    f"\r{era5_var} for site '{ds_site['site'].values[0]}'"
                    " already exists, skipping...",
                    flush=True,
                    end="",
                )


def combine_era5_fluxnet(
    preprocessed_ameriflux_data: Path,
    preprocessed_era5_folder: Path,
) -> xr.Dataset:
    """Combine the preprocessed ERA5 and Ameriflux datasets into a single dataset.

    Args:
        preprocessed_ameriflux_data: Preprocessed Ameriflux dataset
        preprocessed_era5_folder: Preprocessed ERA5 dataset

    Returns:
        The two datasets merged into a single dataset.
    """
    ds_era5 = xr.open_mfdataset(
        list(preprocessed_era5_folder.glob("*.nc")),
        parallel=True,
    )

    ds_sites = xr.open_dataset(
        preprocessed_ameriflux_data,
        chunks={"time": -1, "site": 1},
    )

    return xr.merge([ds_sites, ds_era5])


def generate_training_df(
    merged_dataset: xr.Dataset,
) -> pd.DataFrame:
    """Generate a dataframe ready for machine learning, from the merged dataset.

    Args:
        merged_dataset: Merged ERA5 and Ameriflux datasets.

    Returns:
        Tidy dataframe with only "time" as index.
    """
    return merged_dataset.to_dataframe().dropna().reset_index().set_index("time")
