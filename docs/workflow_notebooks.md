The full workflow is split over several notebooks.
These guide you through all steps required, from preprocessing, to model training and finally the dataset production.

As explained on the [main page](index.md), we train models on two separate datasets.
Fluxnet site-based data for hourly fluxes to resolve smaller spatial and temporal scale fluxes,
and CarbonTracker inverse modelling data to achieve higher accuracy on the longer-term.

As the Fluxnet data is (half-)hourly, some preprocessing is required. This is performed in the first two notebooks.
The CarbonTracker data is at a lower frequency, and more ready to immediately use for analysis.

**Fluxnet-based hourly model:**

 1. [Preprocessing Ameriflux data](notebooks/preprocess_ameriflux.ipynb)
 1. [Preprocessing ERA5](notebooks/preprocess_ERA5_sites.ipynb)
 1. [Fluxnet model training](notebooks/train_fluxnet_models.ipynb)
 1. [Hourly dataset production](notebooks/produce_fluxnet_dataset.ipynb)

**CarbonTracker-based monthly model:**

 1. [CarbonTracker model training](notebooks/train_carbontracker_model.ipynb)
 1. [Monthly dataset production](notebooks/produce_carbontracker_dataset.ipynb)
