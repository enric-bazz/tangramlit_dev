import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.patches import Patch

from . import validation_metrics as vm


def plot_training_history(adata_map, hyperpams=None, lambda_scale=True, log_scale=False):
    """
        Plots a panel with all loss term curves in training.

        Args:
            adata_map (anndata object): input containing .uns["training_history"] returned by map_cells_to_space()
            hyperpams (dict): dictionary containing the hyperparameters used for the mapping
            lambda_scale (bool): Whether to scale the loss terms by lambda (default: True)
            log_scale (bool): Whether the y axis plots should be in log-scale (default: False)

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
        "lambda_f_reg": "lambda_f_reg",
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


def plot_loss_term(adata_map, loss_key: str, lambda_coeff=None, lambda_scale=False, log_scale=False):
    """
    Plots the single loss term specified in input over the training epochs, optionally scaling by its coefficient.

    Args:
        adata_map (anndata object): input containing .uns["training_history"] returned by map_cells_to_space()
        loss_key (str): loss term key in adata_map.uns['training_history']
        lambda_coeff (float): regularization coefficient of the term
        lambda_scale (bool): Whether to scale the loss terms by lambda (default: True)
        log_scale (bool): Whether the y axis plots should be in log-scale (default: False)

    Returns:
        plt.plot()
    """
    # Check if key is present
    if not loss_key in adata_map.uns["training_history"].keys():
        raise ValueError("Loss term not in training history.")
    # Check if coeff is present
    if lambda_scale and lambda_coeff is None:
        raise ValueError("Missing coefficient for scaling.")
    # Check that history is not empty
    if not adata_map.uns["training_history"][loss_key]:
        raise ValueError("Loss term has not been tracked in training.")

    # Retrieve curve
    if type(adata_map.uns["training_history"][loss_key][0]) == torch.Tensor and not torch.isnan(
            adata_map.uns["training_history"][loss_key][0]):
        loss_term_values = []
        for entry in adata_map.uns["training_history"][loss_key]:
            loss_term_values.append(entry.detach())
        loss_term_values = np.asarray(loss_term_values)
    elif type(adata_map.uns["training_history"][loss_key][0]) == float and not np.isnan(
            adata_map.uns["training_history"][loss_key][0]):
        loss_term_values = np.asarray(adata_map.uns["training_history"][loss_key])

        if not loss_term_values.any():  # no truthy values
            raise ValueError("Loss term has not been tracked in training.")

        # Create plot
        plt.figure(figsize=(10, 6))
        title = 'Loss terms over epochs'
        if lambda_scale:
            title += ' (scaled by lambda)'
        if log_scale:
            title = title + ' (logscale)'
        if log_scale:
            plt.semilogy(abs(loss_term_values), label=loss_key)
        else:
            plt.plot(loss_term_values, label=loss_key)
            plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title)
        plt.show()

def plot_filter_weights(adata_map, plot_spaghetti=False, plot_envelope=False):
    """
    Plots the filter weights evolution over epochs with optional additional visualizations.

    Args:
        adata_map (anndata object): input containing .uns["filter_history"] returned by map_cells_to_space()
        plot_spaghetti (bool): If True, plots individual cell trajectories over epochs
        plot_envelope (bool): If True, plots the mean signal with ±1 std deviation envelope
    """
    matrix = np.column_stack(adata_map.uns['filter_history']['filter_values'])
    #matrix = np.concatenate((matrix, np.expand_dims(adata_map.obs['F_out'].to_numpy(), axis=1)), axis=1)   # append final filter values
    
    # Calculate appropriate figure size and aspect ratio
    n_cells, n_epochs = matrix.shape
    aspect_ratio = n_epochs / n_cells  # This gives us the data aspect ratio
    
    # Set base width and adjust height accordingly
    base_width = 12
    fig_height = base_width / aspect_ratio
    
    # Limit maximum height to keep plot reasonable
    fig_height = min(fig_height, 16)
    
    # Main heatmap plot
    plt.figure(figsize=(base_width, fig_height))
    im = plt.imshow(matrix, aspect='auto')  # 'auto' ensures the plot fills the figure
    plt.colorbar(im, fraction=0.03, pad=0.05)
    plt.xlabel('Epoch')
    plt.ylabel('Cell')
    plt.title('Sigmoid weights over epochs')
    plt.show()

    # Additional plots if requested
    if plot_spaghetti or plot_envelope:
        epochs = np.arange(n_epochs)
        
        if plot_spaghetti and plot_envelope:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(base_width, 10))
            
            # Spaghetti plot
            for cell_idx in range(matrix.shape[0]):
                ax1.plot(epochs, matrix[cell_idx, :], alpha=0.1, color='blue')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Filter Weight')
            ax1.set_title('Individual Cell Filter Weight Trajectories')
            ax1.set_ylim(0, 1)
            
            # Envelope plot
            mean_signal = np.mean(matrix, axis=0)
            std_signal = np.std(matrix, axis=0)
            
            ax2.plot(epochs, mean_signal, 'b-', label='Mean')
            ax2.fill_between(epochs, 
                           mean_signal - std_signal,
                           mean_signal + std_signal,
                           alpha=0.3, color='blue', label='±1 std dev')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Filter Weight')
            ax2.set_title('Mean Filter Weight with Standard Deviation Envelope')
            ax2.set_ylim(0, 1)
            ax2.legend()
            
        elif plot_spaghetti:
            plt.figure(figsize=(base_width, 5))
            for cell_idx in range(matrix.shape[0]):
                plt.plot(epochs, matrix[cell_idx, :], alpha=0.1, color='blue')
            plt.xlabel('Epoch')
            plt.ylabel('Filter Weight')
            plt.title('Individual Cell Filter Weight Trajectories')
            plt.ylim(0, 1)
            
        elif plot_envelope:
            plt.figure(figsize=(base_width, 5))
            mean_signal = np.mean(matrix, axis=0)
            std_signal = np.std(matrix, axis=0)
            
            plt.plot(epochs, mean_signal, 'b-', label='Mean')
            plt.fill_between(epochs,
                           mean_signal - std_signal,
                           mean_signal + std_signal,
                           alpha=0.3, color='blue', label='±1 std dev')
            plt.xlabel('Epoch')
            plt.ylabel('Filter Weight')
            plt.title('Mean Filter Weight with Standard Deviation Envelope')
            plt.ylim(0, 1)
            plt.legend()
        
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

def plot_training_scores(df, bins=10, alpha=0.7):
    """
        Plots the 4-panel training diagnosis plot

        Args:
            adata_map (AnnData):
            bins (int or string): Optional. Default is 10.
            alpha (float): Optional. Ranges from 0-1, and controls the opacity. Default is 0.7.

        Returns:
            None
    """
    fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    #df = adata_map.uns["train_genes_df"]
    axs_f = axs.flatten()

    # set limits for axis
    axs_f[0].set_ylim([0.0, 1.0])
    for i in range(1, len(axs_f)):
        axs_f[i].set_xlim([0.0, 1.0])
        axs_f[i].set_ylim([0.0, 1.0])

    #     axs_f[0].set_title('Training scores for single genes')
    sns.histplot(data=df, y="score", bins=bins, ax=axs_f[0], color="coral")

    axs_f[1].set_title("score vs sparsity (single cells)")
    sns.scatterplot(
        data=df,
        y="score",
        x="sparsity_sc",
        ax=axs_f[1],
        alpha=alpha,
        color="coral",
    )

    axs_f[2].set_title("score vs sparsity (spatial)")
    sns.scatterplot(
        data=df,
        y="score",
        x="sparsity_st",
        ax=axs_f[2],
        alpha=alpha,
        color="coral",
    )

    axs_f[3].set_title("score vs sparsity (sp - sc)")
    sns.scatterplot(
        data=df,
        y="score",
        x="sparsity_diff",
        ax=axs_f[3],
        alpha=alpha,
        color="coral",
    )

    plt.tight_layout()
    plt.show()

def plot_auc_curve(df_all_genes, test_genes=None):
    """
            Plots auc curve which is used to evaluate model performance.

        Args:
            df_all_genes (Pandas dataframe): returned by compare_spatial_gene_exp(adata_ge, adata_sp);
            test_genes (list): list of test genes, if not given, test_genes will be set to genes where 'is_training' field is False

        Returns:
            None
        """
    df_all_genes = df_all_genes[test_genes]
    auc_score, ((pol_xs, pol_ys), (xs, ys)) = vm.poly2_auc(df_all_genes['score'], df_all_genes['sparsity_st'])

    plt.figure(figsize=(6, 5))

    plt.plot(pol_xs, pol_ys, c='r')
    sns.scatterplot(x=xs, y=ys, alpha=0.5, edgecolors='face')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.gca().set_aspect(.5)
    plt.xlabel('score')
    plt.ylabel('spatial sparsity')
    plt.tick_params(axis='both', labelsize=8)
    plt.title('Prediction on test transcriptome')

    textstr = 'auc_score={}'.format(np.round(auc_score, 3))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    # place a text box in upper left in axes coords
    plt.text(0.03, 0.1, textstr, fontsize=11, verticalalignment='top', bbox=props)
    plt.show()