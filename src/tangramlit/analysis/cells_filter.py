from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import linregress

def compute_filter_corr(adata_map, plot=True, plot_regression=False):
    """
    Compute correlation of initial and final filter values
    and optionally plot scatter with regression line.

    Args:
        adata_map: adata object output of map_cells_to_space(). Must contain adata_map.uns['filter_history']
        plot (bool): If True, show scatterplot with regression line.
    """

    if "filter_history" not in adata_map.uns.keys():
        raise ValueError("Missing filter history in mapped input data.")
    # Retrieve filter history
    filter_history = adata_map.uns['filter_history']['filter_values']  # shape = (n_epochs, n_cells)

    # Retrieve initial and final filter values
    filter_init = filter_history[0]
    filter_final = filter_history[-1]

    # Compute correlation
    filter_corr = np.corrcoef(filter_init, filter_final)[0, 1]
    print(f"Pearson correlation coefficient of filter values: {filter_corr}")

    if plot:
        plt.figure(figsize=(6, 6))
        plt.scatter(filter_init, filter_final, alpha=0.5, color="blue", label="Cells")

        if plot_regression:
            # Fit regression line
            slope, intercept, _, _, _ = linregress(filter_init, filter_final)
            x_vals = np.array([filter_init.min(), filter_init.max()])
            y_vals = intercept + slope * x_vals
            plt.plot(x_vals, y_vals, color="red", lw=2, label="Fit")

        plt.xlabel("Initial filter values")
        plt.ylabel("Final filter values")
        plt.title(f"Correlation = {filter_corr:.3f}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return filter_corr

def filter_cell_choice_consistency(filter_square, threshold=0.5):
    """
    Evaluate cell choice consistency across multiple runs.

    Arg:
        filter_square (array shape = (n_cells, n_runs)): Square containing in each column the final filter values of each run.
        threshold (float): Threshold value for cell filtering.
    Returns:
        Prints a histogram of the number of runs each cell is selected in (is above threshold).
        Reports the filter consistency as the percentage, over all cells, of cells that are selected in all runs.
    """

    # Get number of runs
    n_runs = filter_square.shape[1]

    # Count number of runs each cell is selected in
    selected_cells = np.sum(filter_square > threshold, axis=1)

    # Plot histogram of selected cells
    plt.hist(selected_cells, bins=np.arange(0, n_runs+1), edgecolor="black")
    plt.title("Histogram of selected cells")
    plt.xlabel("Number of selected runs")
    plt.show()

    # Compute consistency
    consistency = np.sum(selected_cells == n_runs) / len(selected_cells)

    print(f"Filter consistency: {consistency * 100:.3f} %")

    return consistency

# TODO: as of now filter_value > threshold, alternatively move everything to filter_value >= threshold

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