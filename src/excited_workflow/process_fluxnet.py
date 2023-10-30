import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytz
import xarray as xr
from timezonefinder import TimezoneFinder


def get_ameriflux_site_names(zip_folder: Path) -> list[str]:
    """Get a list of all Ameriflux sites of which valid .zip archives exist.

    Args:
        zip_folder: Folder containing the ameriflux zip files.

    Raises:
        FileNotFoundError: If no zip files are present in zip_folder.
        ValueError: If the zip files do not conform to the expected naming scheme.

    Returns:
        List of Ameriflux site names.
    """
    zip_files = list(zip_folder.glob("*.zip"))
    if len(zip_files) == 0:
        raise FileNotFoundError(f"Could not find any zip files in {zip_folder}.")

    filenames = "\n".join([f.name for f in zip_files])

    regex_sitename = "AMF_([A-Z]{2}-.{3})_FLUXNET"
    sitenames = re.findall(regex_sitename, filenames)

    if len(sitenames) == 0:
        raise ValueError(
            "Could not find any Ameriflux zip files.\n"
            "The files should have a name such as: 'AMF_XX-Xxx_FLUXNET_*.zip'"
        )
    return sitenames


def read_ameriflux_site_properties(
    metadata_file: Path,
    sitenames: list[str],
    properties: list[str],
) -> dict[str, dict[str, Any]]:
    """Read the metadata of fluxnet sites.

    Multiple sites should be read at once, as read_excel is slow.

    Args:
        metadata_file: Metadata .xlsx file.
        sitename: Name of the site.
        properties: Properties to extract (e.g. ["LOCATION_LAT", ...])

    Returns:
        dict: Dictionary containing the site's properties.
    """
    df_meta = pd.read_excel(metadata_file)
    site_metadata = {}

    for site in sitenames:
        df_site = df_meta.where(df_meta["SITE_ID"] == site).dropna()
        data = {}
        for prop in properties:
            try:
                data[prop] = (
                    df_site.where(df_site["VARIABLE"] == prop)
                    .dropna()["DATAVALUE"]
                    .to_numpy()[0]
                )
            except IndexError:
                print(f"Failed to read property. {site=}, {prop=}")
        if len(data) > 0:
            site_metadata[site] = data
    return site_metadata


def read_ameriflux_csv(
    sitename: str,
    fluxnet_zip_folder: Path,
    site_tz_offset: np.timedelta64,
    variables: list[str],
    quality_flag: str | None = None,
    minimum_qc_value: int | None = None,
) -> xr.Dataset:
    """Read the fluxnet site's FULLSET csv file, and extract the hourly variable.

    The following steps are taken:
        - The hourly or half-hourly file is read out.
        - The variables are extracted.
        - The data is masked for the minimum quality flag (if given).
        - The data is resampled to a 1-hour interval (for ERA5 alignment).
        - The timestamps are corrected to UTC.
        - The data is returned as an xarray Dataset, contining the site as dimension.

    Args:
        sitename: Name of the site
        fluxnet_zip_folder: Folder containing all the FLUXNET zip files.
        site_tz_offset: The offset between the site's timezone and UTC.
        variables: The names of the variable that should be extracted.
            For example, "NEE_VUT_REF".
        quality_flag: Which quality flag should be used. E.g. 'NEE_VUT_REF_QC'.
        minimum_qc_value: Minimum quality flag value, where 0=measured,
            1=high quality gapfill, 2=mediocre gapfill, and 3=low quality.

    Returns:
        An xarray Dataset containing the site's data.
    """
    if quality_flag is None and minimum_qc_value is not None:
        raise ValueError("Please enter a valid quality flag.")

    site_zip_fname = f"AMF_{sitename}_FLUXNET_FULLSET_*.zip"
    zipfiles = list(fluxnet_zip_folder.glob(site_zip_fname))
    if len(zipfiles) == 0:
        raise FileNotFoundError(
            f"Could not find a file with the pattern {site_zip_fname}."
        )
    site_zipfile = zipfiles[0]

    regex_hhourly_file = re.compile(".*FULLSET_H[HR].*")  # grab the (half)hourly files.
    site_zip = zipfile.ZipFile(site_zipfile)

    site_csv_fname = next(filter(regex_hhourly_file.match, site_zip.namelist()))
    with site_zip.open(site_csv_fname) as f:
        df = pd.read_csv(f, na_values="-9999")

    df["TIMESTAMP_END"] = pd.to_datetime(df["TIMESTAMP_END"], format="%Y%m%d%H%M")
    df = df.set_index("TIMESTAMP_END")

    df_req = df[variables]

    if quality_flag:
        df_req = df_req.where(df[quality_flag] <= minimum_qc_value, np.nan)

    ds_site = df_req.to_xarray()

    ds_site = df_req.to_xarray()
    ds_site = ds_site.rename({"TIMESTAMP_END": "time"})
    ds_site = ds_site.resample(time="1H").mean()  # faster if flox is installed!
    ds_site["time"] = ds_site["time"].values - site_tz_offset
    ds_site = ds_site.expand_dims("site")

    return ds_site


def find_site_utc_offset(
    site_props: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Add the site's timezone offset to the site properties dictionary.

    Args:
        site_props: Dictionary containing properties per site.

    Returns:
        Site properties dictionary, with a new key "UTC_offset" for every site.
    """
    tf = TimezoneFinder()  # reuse

    for site in site_props:
        lat = float(site_props[site]["LOCATION_LAT"])
        lon = float(site_props[site]["LOCATION_LONG"])
        tz_name = tf.timezone_at(lng=lon, lat=lat)
        if tz_name is None:
            raise ValueError(f"Could not find timezone of location {lat},{lon}")
        tz = pytz.timezone(tz_name)
        # Use a winter-date: the sites are in standard time only (no daylight savings)
        td = np.timedelta64(tz.utcoffset(datetime(2010, 1, 1)))
        site_props[site]["UTC_offset"] = td
    return site_props


def mask_sites_transcom(
    transcom_regions_file: Path,
    transcom_region: int,
    site_props: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Mask the sites of site_props, so only sites within a transcom region remain.

    Args:
        transcom_regions_file: The netCDF file containing the transcom regions.
        transcom_region: Which transcom region ID should be used to mask.
        site_props: Dictionary containing properties per site.

    Returns:
        Site properties dictionary, containing only the sites in the specified transcom
            region.
    """
    ds_regions = xr.open_dataset(transcom_regions_file)

    transcom2_sites: list[str] = []
    for site in site_props:
        if (
            ds_regions["transcom_regions"].sel(
                latitude=float(site_props[site]["LOCATION_LAT"]),
                longitude=float(site_props[site]["LOCATION_LONG"]),
                method="nearest",
            )
            == transcom_region
        ):
            transcom2_sites.append(site)

    return {site: site_props[site] for site in site_props if site in transcom2_sites}


def preprocess_ameriflux_sites(
    zip_folder: Path,
    metadata_file: Path,
    transcom_regions_file: Path,
) -> xr.Dataset:
    """Preprocess the Ameriflux sites into analysis-ready data.

    The following operations are performed:
    - The latitude and longitude of every site is found
    - The correct offset from UTC is determined
    - The sites corresponding to transcom region 2 are determined
    - The required variables are loaded from the csv file, masked for the quality flag,
        have the time corrected to UTC, and are resampled to 1 hour intervals.

    Args:
        zip_folder: Folder containing the Ameriflux site .zip files.
        metadata_file: Excel file containing the Ameriflux metadata.
        transcom_regions_file: netCDF file describing the transcom regions.

    Returns:
        The preprocessed data of all Ameriflux sites.
    """
    sitenames = get_ameriflux_site_names(zip_folder=zip_folder)
    print(f"Found {len(sitenames)} Ameriflux sites.")
    site_props = read_ameriflux_site_properties(
        metadata_file=metadata_file,
        sitenames=sitenames,
        properties=["LOCATION_LAT", "LOCATION_LONG"],
    )
    site_props = find_site_utc_offset(site_props)

    site_props = mask_sites_transcom(
        transcom_regions_file=transcom_regions_file,
        transcom_region=2,
        site_props=site_props,
    )
    print(f"Of which {len(site_props)} are located within transcom region 2.")

    print("Starting to load the .csv files...")
    ds_sites: list[xr.Dataset] = [xr.Dataset()] * len(site_props)
    for i_site, site in enumerate(site_props):
        ds_site = read_ameriflux_csv(
            sitename=site,
            fluxnet_zip_folder=zip_folder,
            site_tz_offset=site_props[site]["UTC_offset"],
            variables=["GPP_NT_VUT_REF", "GPP_DT_VUT_REF", "NEE_VUT_REF"],
            quality_flag="NEE_VUT_REF_QC",
            minimum_qc_value=1,
        )

        ds_site["site"] = (["site"], np.array([site], dtype="<U6"))

        ds_site["latitude"] = (["site"], [float(site_props[site]["LOCATION_LAT"])])
        ds_site["longitude"] = (["site"], [float(site_props[site]["LOCATION_LONG"])])

        ds_sites[i_site] = ds_site
    print(".csv files loaded. Merging them and creating the dataset.")
    return xr.concat(ds_sites, dim="site")
