"""Analysis module for Tangram - post-mapping analysis and plotting tools."""

from .annotation import (
    deterministic_annotation,
    transfer_annotation,
    annotation_report,
    one_hot_encoding,
)
from .cells_filter import (
    compute_filter_corr,
    filter_cell_choice_consistency,
    plot_filter_weights,
    plot_filter_count,
)
from .deterministic_mapping import (
    get_cell_spot_pair,
    get_spot_cell_pair,
    count_deterministic_mapping_matches,
    deterministic_mapping_similarity,
)
from .plot_losses import (
    plot_training_history,
    plot_loss_terms,
    plot_validation_metrics_history,
)
from .plot_scores import (
    plot_training_scores,
    plot_auc_curve,
    plot_score_SA_corr,
    plot_score_histograms,
)
from .training_genes import (
    traffic_light_plot,
)

__all__ = [
    # Annotation
    "deterministic_annotation",
    "transfer_annotation",
    "annotation_report",
    "one_hot_encoding",
    # Cells filter
    "compute_filter_corr",
    "filter_cell_choice_consistency",
    "plot_filter_weights",
    "plot_filter_count",
    # Deterministic mapping
    "get_cell_spot_pair",
    "get_spot_cell_pair",
    "count_deterministic_mapping_matches",
    "deterministic_mapping_similarity",
    # Plotting
    "plot_training_history",    
    "plot_loss_terms",
    "plot_validation_metrics_history",
    "plot_training_scores",
    "plot_auc_curve",
    "plot_score_SA_corr",
    "plot_score_histograms",
    # Training genes
    "traffic_light_plot",
]
