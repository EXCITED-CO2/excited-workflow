# EXCITED workflow

An open workflow for creating machine learning models for estimating the global biospheric CO2 exchange.

Using this workflow we aim to better constrain the CO2 exchange in terrestrial ecosystems on longer timescales using estimates from inverse models (e.g., CarbonTracker) as additional input data.

More information is available on the documentation pages.

The following flowchart lays out the workflow of EXCITED:

<details><summary>View flowchart</summary>

```mermaid
graph TD;
    monthlymodel(Monthly ML model);
    input[(ERA5, MODIS, etc.)];
    fluxnet[(Fluxnet)];
    carbontracker[(CarbonTracker)];
    dailydataset["hourly fluxnet NEE\n(biased in long term)"];
    hourlymodel("Hourly ML models\n(GPP and respiration)");
    monthlydataset[(Monthly NEE\ndataset)];
    finaldataset[(Final daily\nNEE dataset)];

    fluxnet-->|target| hourlymodel;
    input-->|predictors| hourlymodel;
    input-->|predictors| monthlymodel;
    carbontracker-->|target| monthlymodel;
    hourlymodel-->dailydataset;
    input-->monthlydataset;
    monthlymodel-->monthlydataset;
    dailydataset-->hpf([high pass filter]);
    hpf-->finaldataset;
    monthlydataset-->finaldataset;
    input-->dailydataset;
```

</details>
