# System setup
When setting up the excited workflow on a system, you will need to first download the data.
Instructions for each data set are available [here](input_data.md).

## Configuration file
The paths to the datasets are added to a config file. 
This makes it easy for everyone on the system to load the data.

You will have to create a config file in either `~/.config/excited/data_paths.yaml` or `/etc/excited/data_paths.yaml`.
In this config file you have to point the datasets to the right path, for example:

```yaml
biomass: /data/volume_2/xusaatchi
copernicus_landcover: /data/volume_2/land_cover
era5_hourly: /data/volume_2/hourly_era5
era5_monthly: /data/volume_2/monthly_era5
era5_land_monthly: /data/volume_2/monthly_era5-land
modis: /data/volume_2/modis_nirv_v2
spei: /data/volume_2/spei
```

## Setting up your Python environment

=== "Python virtual environment"
    Install Python 3.10 or Python 3.11 for your operating system.

    Clone the workflow to your machine using git:

    ```bash
    git clone https://github.com/EXCITED-CO2/excited-workflow.git
    cd excited-workflow
    ```

    Within the `excited-workflow` repository's main folder do:

    ```bash
    python -m venv .venv
    ```

    This creates a python environment. Activate the environment with:

    ```bash
    source .venv/bin/activate
    ```

    Now you can install the workflow with the following command:

    ```bash
    pip install -e .
    ```

    Additionally, to be able to work with the Jupyter notebooks:
    ```bash
    pip install ipython jupyter
    ```

=== "Conda"

    First create a new Python 3.10 or 3.11 environment:

    ```bash
    conda env create 
    ```

    Now you can install the workflow with the following command:

    ```bash
    pip install -e .
    ```

    Additionally, to be able to work with the Jupyter notebooks:
    ```bash
    pip install ipython jupyter
    ```


Now you can run the following command to start jupyter:
```bash
jupyter-notebook
```

## Final notes

Now you should be ready to run the EXCITED workflow.
The [notebooks](workflow_notebooks.md) show which steps you have to go through, and offer more explanation.
