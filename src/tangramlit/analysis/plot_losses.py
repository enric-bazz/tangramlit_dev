from matplotlib import pyplot as plt
import numpy as np
import torch
import math



def plot_training_history(adata_map, hyperpams=None, lambda_scale=True, log_scale=False, show_total_loss=False):
    """
        Plots a panel with all loss term curves in training.

        Args:
            adata_map (anndata object): input containing .uns["training_history"] returned by map_cells_to_space()
            hyperpams (dict): dictionary containing the hyperparameters used for the mapping
            lambda_scale (bool): Whether to scale the loss terms by lambda (default: True)
            log_scale (bool): Whether the y axis plots should be in log-scale (default: False)
            show_total_loss (bool): Whether to show the total loss term named 'loss' (default: False)

        Returns:
            Note that the trainig step stores in adata_map.uns["training_history"] the loss terms for each epoch already scaled
            by their respective hyperparameters, thus to get non scaled values we divide by the respective lambda.
    """

    # Check if training history is present
    if not "training_history" in adata_map.uns.keys():
        raise ValueError("Missing training history in mapped input data.")

    if not lambda_scale and hyperpams is None:
        raise ValueError("Missing hyperparamters for re-scaling.")

    # Retrieve loss terms labels that are not empty
    loss_terms_labels = [k for k, v in adata_map.uns['training_history'].items() if v]

    # Initiate empty dict containing numpy arrays
    loss_dict = {key: None for key in loss_terms_labels}

    # Some terms are returned as a list of torch tensors (scalars) others as lists of float: turn all into ndarray
    for k in loss_terms_labels:
        if type(adata_map.uns["training_history"][k][0]) == torch.Tensor and not torch.isnan(adata_map.uns["training_history"][k][0]):
            loss_term_values = []
            for entry in adata_map.uns["training_history"][k]:
                loss_term_values.append(entry.detach())
            loss_term_values = np.asarray(loss_term_values)
        elif type(adata_map.uns["training_history"][k][0]) == float and not np.isnan(adata_map.uns["training_history"][k][0]):
            loss_term_values = np.asarray(adata_map.uns["training_history"][k])
            # does not implement .copy()
        loss_dict[k] = loss_term_values

    # Scale by lambda (bool)
    loss_lambda_map = {
        "main_loss": "lambda_g1",
        "vg_reg": "lambda_g2",
        "kl_reg": "lambda_d",
        "entropy_reg": "lambda_r",
        "l1_term": "lambda_l1",
        "l2_term": "lambda_l2",
        "count_reg": "lambda_count",
        "filt_reg": "lambda_f_reg",
        "sparsity_term": "lambda_sparsity_g1",
        "neighborhood_term": "lambda_neighborhood_g1",
        "ct_island_term": "lambda_ct_islands",
        "getis_ord_term": "lambda_getis_ord",
        "moran_term": "lambda_moran",
        "geary_term": "lambda_geary",
    }
    if not lambda_scale:
        for loss_key in (loss_dict.keys() & loss_lambda_map.keys()):
            #if loss_dict[loss_key].any():  # truthy keys only
            loss_dict[loss_key] = loss_dict[loss_key] / hyperpams[loss_lambda_map[loss_key]]

    # Optionally hide the total loss term named 'total_loss'
    if not show_total_loss and 'total_loss' in loss_dict:
        # remove from dict so it's not plotted below
        loss_dict.pop('total_loss', None)

    # Create plot
    plt.figure(figsize=(10,20))

    title = 'Loss terms over epochs'
    if lambda_scale:
        title += ' (scaled by lambda)'
    if log_scale:
        title = title + ' (logscale)'

    for curve in loss_dict:
        if log_scale:
            plt.semilogy(abs(loss_dict[curve]), label=curve)
        else:
            plt.plot(loss_dict[curve], label=curve)
        plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.show()


def plot_loss_terms(
    adata_map,
    loss_key,
    lambda_coeff=None,
    lambda_scale=False,
    log_scale=False,
    make_subplot=False,
    subplot_shape=None,
):   
    """
    Plot one or more loss terms from adata_map.uns['training_history'].

    Args:
        adata_map (AnnData): contains .uns['training_history']
        loss_key (str | list[str]): one or more keys of loss terms
        lambda_coeff (float | list[float] | None): λ coefficient(s)
        lambda_scale (bool): if True, plot λ-weighted loss (as stored);
                             if False, divide by λ to show unweighted loss
        log_scale (bool): if True, use semilog-y
        make_subplot (bool): if True, draw all curves in subplots
        subplot_shape (tuple[int, int] | None): (nrows, ncols) for subplots
    """
    history = adata_map.uns.get("training_history")
    if history is None:
        raise ValueError("Missing 'training_history' in adata_map.uns.")

    # Normalize inputs
    if isinstance(loss_key, str):
        loss_key = [loss_key]
    if lambda_coeff is None:
        lambda_coeff = [1.0] * len(loss_key)
    elif isinstance(lambda_coeff, (float, int)):
        lambda_coeff = [lambda_coeff]
    if len(loss_key) != len(lambda_coeff):
        raise ValueError("loss_key and lambda_coeff must have equal length.")

    n_terms = len(loss_key)

    if make_subplot:
        if subplot_shape is None:  # square subplot shape
            ncols = math.ceil(math.sqrt(n_terms))
            nrows = math.ceil(n_terms / ncols)
        else:
            nrows, ncols = subplot_shape
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).reshape(-1)
    else:
        fig, axes = None, None

    for i, (key, coeff) in enumerate(zip(loss_key, lambda_coeff)):
        if key not in history:
            raise ValueError(f"Loss term '{key}' not found in training_history.")
        values = history[key]
        if not values:
            raise ValueError(f"Loss term '{key}' has empty history.")

        # to numpy
        if isinstance(values[0], torch.Tensor):
            values = [v.item() if v.numel() == 1 else v.detach().cpu().numpy() for v in values]
        values = np.asarray(values, dtype=float)

        # Unscale if required
        if not lambda_scale and coeff != 0:
            values = values / coeff

        # pick axis
        if make_subplot:
            ax = axes[i]
        else:
            plt.figure(figsize=(7, 4))
            ax = plt.gca()

        if log_scale:
            ax.semilogy(np.abs(values))
        else:
            ax.plot(values)

        ax.set_title(f"{key} ({'λ-scaled' if lambda_scale else 'unscaled'})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

    plt.tight_layout()
    plt.show()
        


def plot_validation_metrics_history(adata_map, figsize=(10, 5), add_training_scores=False):
    """
    Plots the history of validation metrics stored in adata_map.uns['validation_history'] in separated plots.
    Optionally compare training score and sparsity-weighted score with the validation ones if add_training_scores is True.

    Args:
        adata_map: anndata object returned by the mapping containing the validation history
        figsize: tuple specifying the figure size
    """
    if 'validation_history' not in adata_map.uns:
        raise ValueError("Missing 'validation_history' in adata_map.uns.")

    val_history = adata_map.uns['validation_history']
    metrics = val_history.keys()
    epochs = range(0, len(next(iter(val_history.values()))))

    for metric in metrics:
        plt.figure(figsize=figsize)
        if add_training_scores and metric == 'val_score':
            # Plot both training and validation scores
            plt.plot(epochs, adata_map.uns['training_history']['main_loss'], '-o', label='Training Score')
            plt.plot(epochs, val_history[metric], '-o', label='Validation Score')
            plt.ylabel('Score')
            plt.title('Training vs Validation Score over epochs')
        elif add_training_scores and metric == 'val_sparsity-weighted_score':
            # Plot both training and validation sparsity-weighted scores
            plt.plot(epochs, adata_map.uns['training_history']['sparsity_term'], '-o', label='Training Sparsity-Weighted Score')
            plt.plot(epochs, val_history[metric], '-o', label='Validation Sparsity-Weighted Score')
            plt.ylabel('Sparsity-Weighted Score')
            plt.title('Training vs Validation Sparsity-Weighted Score over epochs')
        else:
            plt.plot(epochs, val_history[metric], '-o')
            plt.ylabel(metric)
            plt.title(f'Validation {metric} over epochs')

        plt.xlabel('Epoch')
        plt.grid(True)
        plt.legend()
        plt.show()



