"""Mapping module for Tangram - cell-to-space mapping using PyTorch Lightning."""

from .trainer import (
    map_cells_to_space,
    validate_mapping_inputs,
    run_multiple_mappings,
)
from .utils import (
    validate_mapping_experiment,
    project_sc_genes_onto_space,
    compare_spatial_geneexp,
)
from .lit_mapper import (
    MapperLightning,
    EpochProgressBar,
    poly2_auc,
)
from .loss import TangramLoss

__all__ = [
    "map_cells_to_space",
    "validate_mapping_inputs",
    "project_sc_genes_onto_space",
    "validate_mapping_experiment",
    "compare_spatial_geneexp",
    "MapperLightning",
    "EpochProgressBar",
    "poly2_auc",
    "TangramLoss",
    "run_multiple_mappings",
]
