"""Source datasets for EXCITED."""
from excited_workflow.source_datasets.biomass import Biomass
from excited_workflow.source_datasets.spei import Spei
from excited_workflow.protocol import DataSource

datasets: dict[str, DataSource] = {
    "biomass": Biomass,
    "spei": Spei,
}
