This repository serves to reproduce the workflow of Conor's MSc thesis.

# EXCITED workflow

The following flowchart lays out the workflow of EXCITED. Two models are trained, the first one on (hourly)
```mermaid
graph TD;
    monthlymodel(Monthly ML model);
    input[(ERA5, MODIS, etc.)];
    fluxnet[(Fluxnet)];
    carbontracker[(CarbonTracker)];
    hourlymodel(Hourly model);
    dailydataset["Hourly fluxnet NEE\n(biased in long term)"];
    hourlymodel(Hourly ML model);

    input-->monthlymodel;
    fluxnet-->hourlymodel;
    carbontracker-->monthlymodel;
    input-->hourlymodel;
    hourlymodel-->dailydataset;
    input-->mp;
    monthlymodel-->mp[(Monthly NEE\ndataset)];
    dailydataset-->hpf([high pass filter]);
    hpf-->dailyd[(Final daily\nNEE dataset)];
    mp-->dailyd;
    input-->dailydataset;
```

# Getting started

## Setting up your environment

Install Python 3.10 for your operating system.

Within the `scratch` repository's main folder do:

```bash
python -m venv .venv
```

This creates a python environment. Activate the environment with:

```bash
source .venv/bin/activate
```

Now you can install all the required packages with the following command:

```bash
pip install -r requirements.txt
```

Additionally, to be able to work with the Jupyter notebooks:
```bash
pip install ipython jupyter
```

Now you can run the following command to start jupyter:
```bash
jupyter-notebook
```

## Getting the data

### CarbonTracker
The CarbonTracker data is available from NOAA's Global Monitoring Laboratory. You will need the following files:

- [The Transcom regions file (~2 MB)](https://gml.noaa.gov/aftp//products/carbontracker/co2/regions.nc)
- [The monthly average carbon flux file (~500 MB)](https://gml.noaa.gov/aftp//products/carbontracker/co2/CT2022/fluxes/monthly/CT2022.flux1x1-monthly.nc)

### ERA5
#### Hourly data (Fluxnet model)
Do note that this is a lot of data (~230 GB), and the download will take a long time. 

Change to the folder where you want and run the following command:
With [era5cli](https://github.com/eWaterCycle/era5cli) the ERA5 data can be downloaded using the following command. Do note that this is a lot of data (~230 GB), and the download will take a long time. Do `pip install era5cli` to be able to run the command.

```bash
era5cli hourly \
    --variables 2m_temperature 2m_dewpoint_temperature surface_net_solar_radiation \
    surface_net_thermal_radiation mean_surface_sensible_heat_flux mean_surface_latent_heat_flux \
    surface_pressure total_precipitation \
    --startyear 1995 --endyear 2020 --levels surface \
    --area 60 -140 15 -55
```

#### Monthly data (CarbonTracker model)
For CarbonTracker only monthly data is required. For this data the required storage is 4.2 GB.

Change to the desired folder and run:
```bash
era5cli monthly \
    --variables 2m_temperature 2m_dewpoint_temperature surface_net_solar_radiation \
    surface_net_thermal_radiation mean_surface_sensible_heat_flux mean_surface_latent_heat_flux \
    type_of_low_vegetation type_of_high_vegetation \
    surface_pressure total_precipitation \
    type_of_low_vegetation type_of_high_vegetation \
    --startyear 2000 --endyear 2020 --levels surface \
    --area 60 -140 15 -55
```

### Fluxnet

The Ameriflux data was downloaded from the following link https://ameriflux.lbl.gov/data/download-data/

To be able to download the data you need to be registered.

Go through the following steps:

1. Select `AmeriFlux FLUXNET`
2. Select `FULLSET`, `Include BADM`
3. Select all sites.
4. Describe why you want the data, and agree to the policy
5. Download the zip files for all sites, as well as the metadata file `AMF_AA-Flx_FLUXNET-BIF_CCBY4_20221210.xlsx`.
   - Move all the downloaded zip files to a single folder


## Run the notebooks
Now you can run the notebooks.

- Start with `preprocess_ameriflux.ipynb`. This notebook will preprocess the Ameriflux data to be in a more useful format.
- Next you can preprocess the ERA5 data (extract data per site) with `preprocess_ERA5_sites.ipynb`.
- Now you can train the ML model on the ERA5 and Fluxnet data with `training_a_model.ipynb`.
