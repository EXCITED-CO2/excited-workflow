# EXCITED workflow

An open workflow for creating machine learning models for estimating the global biospheric CO2 exchange.

Using this workflow we aim to better constrain the CO2 exchange in terrestrial ecosystems on longer timescales using estimates from inverse models (e.g., CarbonTracker) as additional input data.

For information on how to set up the workflow on your system, see the [setup page](system_setup.md).
The workflow notebooks are all available [here](workflow_notebooks.md).

The following flowchart lays out the workflow of EXCITED:

```mermaid
graph TD;
    monthlymodel(Monthly ML model);
    input[(ERA5, MODIS, etc.)];
    fluxnet[(Fluxnet)];
    carbontracker[(CarbonTracker)];
    hourlydataset[("Hourly fluxnet NEE\n(biased in long term)")];
    hourlymodel("Hourly ML models\n(GPP and respiration)");
    monthlydataset[(Monthly NEE\ndataset)];
    finaldataset[(Final hourly\nNEE dataset)];

    fluxnet-->|target| hourlymodel;
    input-->|predictors| hourlymodel;
    input-->|predictors| monthlymodel;
    carbontracker-->|target| monthlymodel;
    hourlymodel-->hourlydataset;
    input-->monthlydataset;
    monthlymodel-->monthlydataset;
    hourlydataset-->hpf([high pass filter]);
    hpf-->finaldataset;
    monthlydataset-->|interpolate| finaldataset;
    input-->hourlydataset;
```
