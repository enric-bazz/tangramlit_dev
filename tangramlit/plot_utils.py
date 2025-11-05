import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import math
from matplotlib.patches import Patch

from . import validation_metrics as vm


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

    # Optionally hide the total loss term named 'loss'
    if not show_total_loss and 'loss' in loss_dict:
        # remove from dict so it's not plotted below
        loss_dict.pop('loss', None)

    # Create plot
    plt.figure(figsize=(10,20))

    title = 'Loss terms over epochs'
    if lambda_scale:
        title += ' (scaled by lambda)'
    if log_scale:
        title = title + ' (logscale)'

    for curve in loss_dict:
        #if loss_dict[curve].any():  # truthy keys only
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
        

def plot_filter_weights(
    adata_map,
    plot_heatmap=False,
    plot_spaghetti=False,
    plot_envelope=False,
    threshold=0.5,
):
    """
    Plots optional visualizations of filter weight dynamics.

    Args:
        adata_map (anndata): must contain .uns['filter_history']['filter_values']
        plot_heatmap (bool): plot cell x epoch heatmap
        plot_spaghetti (bool): plot individual cell trajectories
        plot_envelope (bool): plot mean ± std envelopes for two trajectory groups
        threshold (float): threshold to split cells for envelope plot
    """
    matrix = np.column_stack(adata_map.uns["filter_history"]["filter_values"])
    n_cells, n_epochs = matrix.shape
    epochs = np.arange(n_epochs)
    base_width = 12

    if plot_heatmap:
        aspect_ratio = n_epochs / n_cells
        fig_height = min(base_width / aspect_ratio, 16)
        plt.figure(figsize=(base_width, fig_height))
        im = plt.imshow(matrix, aspect="auto")
        plt.colorbar(im, fraction=0.03, pad=0.05)
        plt.xlabel("Epoch")
        plt.ylabel("Cell")
        plt.title("Sigmoid filter weights over epochs")
        plt.tight_layout()
        plt.show()

    if plot_spaghetti:
        plt.figure(figsize=(base_width, 5))
        for cell_idx in range(n_cells):
            plt.plot(epochs, matrix[cell_idx, :], alpha=0.1, color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Filter Weight")
        plt.title("Individual cell filter weight trajectories")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()

    if plot_envelope:
        # Split cells based on final epoch mean threshold
        final_vals = matrix[:, -1]
        high_group = matrix[final_vals >= threshold, :]
        low_group = matrix[final_vals < threshold, :]

        plt.figure(figsize=(base_width, 5))
        for group, color, label in [
            (low_group, "orange", "Low group"),
            (high_group, "blue", "High group"),
        ]:
            if group.size == 0:
                continue
            mean_signal = np.mean(group, axis=0)
            std_signal = np.std(group, axis=0)
            plt.plot(epochs, mean_signal, color=color, label=f"{label} mean")
            plt.fill_between(
                epochs,
                mean_signal - std_signal,
                mean_signal + std_signal,
                alpha=0.3,
                color=color,
                label=f"{label} ±1 std",
            )

        plt.xlabel("Epoch")
        plt.ylabel("Filter Weight")
        plt.title(f"Mean ± std envelopes (threshold={threshold})")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.show()



def plot_filter_count(adata_map, target_count=None, threshold=0.5, figsize=(10, 5)):
    """
    Plot the number of cells passing the filter threshold over epochs.

    Args:
        adata_map: anndata object returned my the mapping containing the filter history
        and target count equal to the one used for the mapping (if missing it is internally computed
        as in the optimizer class) and a threshold.

        This is a useful diagnostic plot as it shows how far the final number of cells is from the target.
        It should be related to the corresponding term in the loss function.
    """
    n_spots = adata_map.X.shape[1]

    # Set target count if not provided
    if target_count is None:
        target_count = n_spots

    fig, ax = plt.subplots(figsize=figsize)
    n_cells = (np.array(adata_map.uns['filter_history']['filter_values']) > threshold).sum(axis=1)
    #n_cells = n_cells.append((adata_map.obs['F_out'] > 0.5).sum().item())  # last value
    epochs = range(1, len(n_cells) + 1)

    # Plot number of cells
    ax.plot(epochs, n_cells, '-o', label='Filtered cells')

    # Add horizontal line for target count
    ax.axhline(y=target_count, color='r', linestyle='--', label='Target count')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Number of cells')
    ax.set_title('Number of cells passing filter threshold per epoch')
    ax.grid(True)
    ax.legend()
    plt.show()


def traffic_light_plot(genes_list, values_sc=None, values_sp=None, figsize=(10, 10)):
    """
    Creates a traffic light visualization where genes are represented as RGB elements
    arranged in a square/rectangular matrix. The first set of values controls the red channel,
    the second set controls the green channel. Values are automatically normalized to [0,1] range.
    
    Args:
        genes_list (list): List of gene names
        values_sc (numpy.ndarray, optional): Values for single cell data (controls red channel).
            Can be continuous or boolean. If None, defaults to ones.
        values_sp (numpy.ndarray, optional): Values for spatial data (controls green channel).
            Can be continuous or boolean. If None, defaults to ones.
        figsize (tuple): Figure size in inches (width, height)
    
    Returns:
        None (displays the plot)
    """
    n_genes = len(genes_list)
    
    # If values are not provided, raise error
    if values_sc is None:
        raise ValueError("single-cell values must be provided.")
    if values_sp is None:
        raise ValueError("spatial values must be provided.")
    if not n_genes == len(values_sc) == len(values_sp):
        raise ValueError("values must be of the same length as genes_list.")

    # Convert to numpy arrays if they aren't already
    values_sc = np.asarray(values_sc)
    values_sp = np.asarray(values_sp)
    
    # Normalize values to [0,1] if they aren't boolean
    if not values_sc.dtype == bool:
        if values_sc.max() != values_sc.min():
            values_sc = (values_sc - values_sc.min()) / (values_sc.max() - values_sc.min())
    if not values_sp.dtype == bool:
        if values_sp.max() != values_sp.min():
            values_sp = (values_sp - values_sp.min()) / (values_sp.max() - values_sp.min())
    
    # Convert boolean arrays to float
    values_sc = values_sc.astype(float)
    values_sp = values_sp.astype(float)
    
    # Create the RGB array (n_genes x 3)
    rgb_array = np.zeros((n_genes, 3))
    rgb_array[:, 0] = values_sc  # Red channel
    rgb_array[:, 1] = values_sp  # Green channel
    # Blue channel remains 0
    
    # Calculate dimensions for the square/rectangular matrix
    width = int(np.ceil(np.sqrt(n_genes)))
    height = int(np.ceil(n_genes / width))
    
    # Create the padded matrix
    total_cells = width * height
    padding_needed = total_cells - n_genes
    
    # Add padding (white cells) if needed
    if padding_needed > 0:
        padding = np.ones((padding_needed, 3))
        rgb_array = np.vstack([rgb_array, padding])
    
    # Reshape into 2D matrix
    matrix = rgb_array.reshape(height, width, 3)
    
    # Create the plot
    plt.figure(figsize=figsize)
    plt.imshow(matrix)
    
    # Remove all axes, labels, and ticks
    plt.axis('off')
    
    # Add legend
    legend_elements = [
        Patch(facecolor='red', label='Single cell signal (Red)'),
        Patch(facecolor='green', label='Spatial signal (Green)'),
        Patch(facecolor='yellow', label='High in both (Red + Green)'),
        Patch(facecolor='black', label='Padding')
    ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 0.5), loc='center left')

    plt.title(f'Gene Traffic Light Matrix ({height}×{width})')
    
    plt.tight_layout()
    plt.show()

    # TODO: pass masks as pandas Series and use indexing for intersection --> use utils.get_matched_genes()

def plot_training_scores(df_g, bins=10, alpha=0.7):
    """
        Plots the 4-panel training diagnosis plot. Restricted on genes flagged as 'is_training'.

        Args:
            df_g (pandas.DataFrame): Contains overlap genes sparsity/score values produced by mapping_utils.compare_spatial_gene_expr()
            bins (int or string): Optional. Default is 10.
            alpha (float): Optional. Ranges from 0-1, and controls the opacity. Default is 0.7.

        Returns:
            None
    """
    df_g = df_g.loc[df_g['is_training']]

    fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    axs_f = axs.flatten()

    # set limits for axis
    axs_f[0].set_ylim([0.0, 1.0])
    for i in range(1, len(axs_f)):
        axs_f[i].set_xlim([0.0, 1.0])
        axs_f[i].set_ylim([0.0, 1.0])

    axs_f[0].set_title('Training scores for single genes')
    sns.histplot(data=df_g, y="score", bins=bins, ax=axs_f[0], color="coral")

    axs_f[1].set_title("score vs sparsity (single cells)")
    sns.scatterplot(
        data=df_g,
        y="score",
        x="sparsity_sc",
        ax=axs_f[1],
        alpha=alpha,
        color="coral",
    )

    axs_f[2].set_title("score vs sparsity (spatial)")
    sns.scatterplot(
        data=df_g,
        y="score",
        x="sparsity_st",
        ax=axs_f[2],
        alpha=alpha,
        color="coral",
    )

    axs_f[3].set_title("score vs sparsity (sp - sc)")
    sns.scatterplot(
        data=df_g,
        y="score",
        x="sparsity_diff",
        ax=axs_f[3],
        alpha=alpha,
        color="coral",
    )

    plt.tight_layout()
    plt.show()

def plot_auc_curve(df_g, test_genes=None, plot_train=False, plot_test=True):
    """
        Plots auc curve of non-training genes score. Test genes are either input or deduced from df_g['is_training'].

        Args:
            df_g (pandas.DataFrame): returned by compare_spatial_gene_expr(adata_ge, adata_sp).
            test_genes (list): list of genes to restrict plot to (keeps behavior compatible with previous API).
            plot_train (bool): if True, include genes flagged as training (df_g['is_training'] == True).
            plot_test (bool): if True, include genes flagged as NOT training (df_g['is_training'] == False). Default True.

        Returns:
            None
        """
    # Restrict to provided gene list if given
    if test_genes is not None:
        # prefer .loc with gene labels; if that fails, fall back to membership
        try:
            df_sel = df_g.loc[test_genes].copy()
        except Exception:
            df_sel = df_g[df_g.index.isin(test_genes)].copy()
    else:
        df_sel = df_g.copy()

    # Decide which groups to plot
    if not plot_train and not plot_test:
        raise ValueError('At least one of plot_train or plot_test must be True.')

    def prepare_group(df, training_flag):
        grp = df.loc[df['is_training'] == training_flag].copy()
        # remove zero-score entries (poly fit / auc routine requires non-zero scores)
        grp = grp.loc[grp['score'] != 0]
        return grp

    plt.figure(figsize=(6, 5))

    # Single-group: only test (default) or only train
    if plot_test and not plot_train:
        grp = prepare_group(df_sel, False)
        if grp.empty:
            raise ValueError('No test genes with non-null score, CosSim not computable.')
        auc_score, ((pol_xs, pol_ys), (xs, ys)) = vm.poly2_auc(grp['score'], grp['sparsity_st'], plot_auc=False)

        plt.plot(pol_xs, pol_ys, c='r', label=f'Test poly (auc={auc_score:.3f})')
        sns.scatterplot(x=xs, y=ys, alpha=0.5, edgecolors='face', color='red', label='Test')

        plt.title('Prediction on test transcriptome')
        textstr = 'test_auc={}'.format(np.round(auc_score, 3))

    elif plot_train and not plot_test:
        grp = prepare_group(df_sel, True)
        if grp.empty:
            raise ValueError('No training genes with non-null score, CosSim not computable.')
        auc_score, ((pol_xs, pol_ys), (xs, ys)) = vm.poly2_auc(grp['score'], grp['sparsity_st'], plot_auc=False)

        plt.plot(pol_xs, pol_ys, c='b', label=f'Train poly (auc={auc_score:.3f})')
        sns.scatterplot(x=xs, y=ys, alpha=0.5, edgecolors='face', color='blue', label='Train')

        plt.title('Prediction on training transcriptome')
        textstr = 'train_auc={}'.format(np.round(auc_score, 3))

    else:
        # Both groups: color points by is_training and fit two polynomials
        grp_train = prepare_group(df_sel, True)
        grp_test = prepare_group(df_sel, False)
        if grp_train.empty or grp_test.empty:
            raise ValueError('Both training and test groups must contain genes with non-null scores for combined plotting.')

        auc_train, ((pol_x_tr, pol_y_tr), (xs_tr, ys_tr)) = vm.poly2_auc(grp_train['score'], grp_train['sparsity_st'], plot_auc=False)
        auc_test, ((pol_x_te, pol_y_te), (xs_te, ys_te)) = vm.poly2_auc(grp_test['score'], grp_test['sparsity_st'], plot_auc=False)

        # plot polynomial fits
        plt.plot(pol_x_tr, pol_y_tr, c='b', label=f'Train poly (auc={auc_train:.3f})')
        plt.plot(pol_x_te, pol_y_te, c='r', label=f'Test poly (auc={auc_test:.3f})')

        # plot points
        sns.scatterplot(x=xs_tr, y=ys_tr, alpha=0.6, edgecolors='face', color='blue', label='Training')
        sns.scatterplot(x=xs_te, y=ys_te, alpha=0.6, edgecolors='face', color='red', label='Test')

        textstr = 'train_auc={0}\ntest_auc={1}'.format(np.round(auc_train, 3), np.round(auc_test, 3))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.gca().set_aspect(.5)
    plt.xlabel('score')
    plt.ylabel('spatial sparsity')
    plt.tick_params(axis='both', labelsize=8)
    plt.legend()

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    # place a text box in upper left in axes coords
    plt.text(0.03, 0.1, textstr, fontsize=11, verticalalignment='top', bbox=props)
    plt.show()