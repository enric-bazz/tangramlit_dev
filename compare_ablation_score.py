#!/usr/bin/env python3
"""
Script to compare ablation study validation scores across different datasets.
Usage: python compare_ablation_score.py 1 2 3 4 5
"""

import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
import pandas as pd
import seaborn as sns
from pathlib import Path


def load_validation_scores(h5ad_file):
    """
    Load validation scores from an h5ad file.
    
    Args:
        h5ad_file (str): Path to the h5ad file
        
    Returns:
        tuple: (label, val_scores) where label is extracted from filename
               and val_scores is the validation history array
    """
    # Load data lazily (backed mode)
    adata = ad.read_h5ad(h5ad_file, backed='r')
    
    # Extract label from filename (the part after Dataset{X}_)
    filename = Path(h5ad_file).stem
    label = filename.split('_', 1)[1]
    
    # Get validation scores from uns
    val_scores = np.array(adata.uns['validation_history']['val_score'])
    return label, val_scores


def plot_dataset_scores(dataset_num, root_dir="./ablation_study"):
    """
    Plot validation score trajectories for all h5ad files in a dataset directory.
    
    Args:
        dataset_num (int): Dataset number (1, 2, 3, 4, 5)
        root_dir (str): Root directory containing ablation_study folder
    """
    dataset_dir = Path(root_dir) / f"Dataset{dataset_num}"
    
    if not dataset_dir.exists():
        print(f"Error: Directory {dataset_dir} does not exist")
        return
    
    # Find all h5ad files in the dataset directory
    h5ad_pattern = str(dataset_dir / f"Dataset{dataset_num}_*.h5ad")
    h5ad_files = glob.glob(h5ad_pattern)
    
    if not h5ad_files:
        print(f"No h5ad files found in {dataset_dir}")
        return
    
    print(f"Found {len(h5ad_files)} h5ad files for Dataset{dataset_num}")
    
    # Create figure for this dataset
    plt.figure(figsize=(12, 8))
    
    # Load and plot each file
    for h5ad_file in sorted(h5ad_files):
        label, val_scores = load_validation_scores(h5ad_file)
        epochs = np.arange(1, len(val_scores) + 1)
        plt.plot(epochs, val_scores, label=label, marker='o', markersize=4, linewidth=2)
        print(f"  - {label}: {len(val_scores)} epochs")
    
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Validation Score', fontsize=12)
    plt.title(f'Validation Score Trajectories - Dataset{dataset_num}', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot in the dataset folder
    output_path = dataset_dir / 'score_trajectories.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Score trajectories plot saved as {output_path}")
    plt.show()


def plot_violin_scores(dataset_num, root_dir="./ablation_study"):
    """
    Plot violin plots of train and validation scores from benchmark.xlsx files in ablated_ subdirectories.
    
    Args:
        dataset_num (int): Dataset number (1, 2, 3, 4, 5)
        root_dir (str): Root directory containing ablation_study folder
    """
    dataset_dir = Path(root_dir) / f"Dataset{dataset_num}"
    
    if not dataset_dir.exists():
        print(f"Error: Directory {dataset_dir} does not exist")
        return
    
    # Find all ablated_ subdirectories
    ablated_dirs = [d for d in dataset_dir.glob("ablated_*") if d.is_dir()]
    
    if not ablated_dirs:
        print(f"No ablated_ subdirectories found in {dataset_dir}")
        return
    
    print(f"Found {len(ablated_dirs)} ablated subdirectories for Dataset{dataset_num}")
    
    # Create subplots - 2 rows (train and val scores) x n columns (subdirectories)
    n_dirs = len(ablated_dirs)
    fig, axes = plt.subplots(2, n_dirs, figsize=(4*n_dirs, 8))
    
    # Handle case with single directory
    if n_dirs == 1:
        axes = axes.reshape(2, 1)
    
    # Set light colors for train and validation
    train_color = '#FFB6C1'  # Light pink
    val_color = '#ADD8E6'    # Light blue
    
    for i, ablated_dir in enumerate(sorted(ablated_dirs)):
        # Extract subdirectory name (remove 'ablated_' prefix)
        subdir_name = ablated_dir.name.replace('ablated_', '')
        
        # Read the benchmark.xlsx file
        benchmark_file = ablated_dir / 'benchmark.xlsx'
        df = pd.read_excel(benchmark_file)
        
        # Separate scores based on boolean columns
        train_data = df[df['is_training'] == True]['score']
        val_data = df[df['is_validation'] == True]['score']
        
        # Plot train scores (top row)
        sns.violinplot(y=train_data, ax=axes[0, i], color=train_color)
        axes[0, i].set_title(f'{subdir_name}')
        axes[0, i].set_ylabel('Training Score')
        axes[0, i].set_ylim(0, 1)
        
        # Plot validation scores (bottom row)
        sns.violinplot(y=val_data, ax=axes[1, i], color=val_color)
        axes[1, i].set_ylabel('Validation Score')
        axes[1, i].set_ylim(0, 1)
        
        print(f"  - {subdir_name}: {len(train_data)} train scores, {len(val_data)} validation scores")
    
    plt.tight_layout()
    
    # Save the plot
    save_path = dataset_dir / 'scores_violin_plot.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Violin plot saved as {save_path}")
    
    plt.show()


def main():
    """Main function to process command line arguments and generate plots."""
    if len(sys.argv) < 2:
        print("Usage: python compare_ablation_score.py <dataset_numbers>")
        print("Example: python compare_ablation_score.py 1 2 3 4 5")
        sys.exit(1)
    
    # Get dataset numbers from command line arguments
    try:
        dataset_numbers = [int(arg) for arg in sys.argv[1:]]
    except ValueError:
        print("Error: All arguments must be integers (dataset numbers)")
        sys.exit(1)
    
    # Check if we're in the right directory
    root_dir = Path.cwd()
    ablation_dir = root_dir / "ablation_study"
    
    if not ablation_dir.exists():
        print(f"Error: ablation_study directory not found in {root_dir}")
        sys.exit(1)
    
    print(f"Processing datasets: {dataset_numbers}")
    print(f"Working directory: {root_dir}")
    print(f"Ablation study directory: {ablation_dir}")
    print("-" * 50)
    
    # Generate plots for each requested dataset
    for dataset_num in dataset_numbers:
        print(f"\nProcessing Dataset{dataset_num}...")
        plot_dataset_scores(dataset_num, ablation_dir)
        plot_violin_scores(dataset_num, ablation_dir)
        print("-" * 30)
    
    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
