"""Tangram-Lightning: Cell-to-space mapping using PyTorch Lightning."""

# Data module exports
from .data import (
    PairedDataModule,
    AdataPairDataset,
    split_train_val_test_genes,
    kfold_gene_splits,
    fps_seeds,
    bfs_bin_growth,
    compute_bin_celltype_fractions,
)

# Mapping module exports
from .mapping import (
    map_cells_to_space,
    validate_mapping_inputs,
    validate_mapping_experiment,
    project_sc_genes_onto_space,
    compare_spatial_geneexp,
    MapperLightning,
    EpochProgressBar,
    poly2_auc,
    TangramLoss,
    run_multiple_mappings,
)

# Analysis module exports
from .analysis import (
    deterministic_annotation,
    transfer_annotation,
    annotation_report,
    one_hot_encoding,
    compute_filter_corr,
    filter_cell_choice_consistency,
    get_cell_spot_pair,
    get_spot_cell_pair,
    count_deterministic_mapping_matches,
    deterministic_mapping_similarity,
    plot_filter_weights,
    plot_filter_count,
    plot_training_history,
    plot_loss_terms,
    plot_validation_metrics_history,
    plot_training_scores,
    plot_auc_curve,
    plot_score_SA_corr,
    plot_score_histograms,
    traffic_light_plot,
)

# Benchmarking module exports
from .benchmarking import (
    SSIM,
    PCC,
    JS,
    RMSE,
    benchmark_mapping,
)

# Optuna tuning module exports
from .hpo import tune_loss_coefficients

__all__ = [
    # Data
    "PairedDataModule",
    "AdataPairDataset",
    "split_train_val_test_genes",
    "kfold_gene_splits",
    "fps_seeds",
    "bfs_bin_growth",
    "compute_bin_celltype_fractions",
    # Mapping
    "map_cells_to_space",
    "validate_mapping_inputs",
    "validate_mapping_experiment",
    "project_sc_genes_onto_space",
    "compare_spatial_geneexp",
    "MapperLightning",
    "EpochProgressBar",
    "poly2_auc",
    "TangramLoss",
    "run_multiple_mappings",
    # Analysis
    "deterministic_annotation",
    "transfer_annotation",
    "annotation_report",
    "one_hot_encoding",
    "compute_filter_corr",
    "filter_cell_choice_consistency",
    "get_cell_spot_pair",
    "get_spot_cell_pair",
    "count_deterministic_mapping_matches",
    "deterministic_mapping_similarity",
    # Benchmarking
    "SSIM",
    "PCC",
    "JS",
    "RMSE",
    "cal_ssim",
    "scale_max",
    "scale_z_score",
    "scale_plus",
    "benchmark_mapping",
    # Plotting
    "plot_filter_weights",
    "plot_filter_count",
    "plot_training_history",
    "plot_loss_terms",
    "plot_validation_metrics_history",
    "plot_training_scores",
    "plot_auc_curve",
    "plot_score_SA_corr",
    "plot_score_histograms",
    "traffic_light_plot",
    # Optuna
    "tune_loss_coefficients",
]


