from pathlib import Path

import pandas as pd
import xarray as xr


def extract_per_site_era5_data(
    preprocessed_ameriflux_data: Path,
    era5_data_folder: Path,
    output_folder: Path,
) -> None:
    ds_sites = xr.open_dataset(
        preprocessed_ameriflux_data,
        chunks={"time": -1, "site": 1},
    )

    ds_era5 = xr.open_mfdataset(
        era5_data_folder.glob("*.nc"),
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
            tolerance=1
        )

        for era5_var in era5_variables:
            fpath = (
                output_folder / f"ERA5_{era5_var}_{ds_site['site'].values[0]}.nc"
            )

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
    ds_era5 = xr.open_mfdataset(
        preprocessed_era5_folder.glob("*.nc"),
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
    return merged_dataset.to_dataframe().dropna().reset_index().set_index("time")
