"""Source datasets for EXCITED."""

from excited_workflow.source_datasets.biomass import Biomass
from excited_workflow.source_datasets.era5 import ERA5Hourly
from excited_workflow.source_datasets.era5 import ERA5LandMonthly
from excited_workflow.source_datasets.era5 import ERA5Monthly
from excited_workflow.source_datasets.land_cover import LandCover
from excited_workflow.source_datasets.modis import Modis
from excited_workflow.source_datasets.protocol import DataSource
from excited_workflow.source_datasets.spei import Spei


datasets: dict[str, DataSource] = {
    "biomass": Biomass(),
    "era5_hourly": ERA5Hourly(),
    "era5_monthly": ERA5Monthly(),
    "era5_land_monthly": ERA5LandMonthly(),
    "copernicus_landcover": LandCover(),
    "modis": Modis(),
    "spei": Spei(),
}
