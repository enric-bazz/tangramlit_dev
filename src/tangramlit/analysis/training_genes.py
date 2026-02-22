import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


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