#!/usr/bin/env python3
"""
Boxplot generation script for ablation study validation scores.

This script generates boxplots showing validation score distributions
for each ablated term across all datasets.

Output: ablation_validation_scores_boxplots.png
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def get_datasets(selected_datasets=None):
    """Get list of available datasets."""
    # Find ablation_study directory relative to current working directory
    if os.path.exists("ablation_study"):
        ablation_dir = Path("ablation_study")
    elif os.path.exists("../ablation_study"):
        ablation_dir = Path("../ablation_study")
    else:
        raise FileNotFoundError("Could not find ablation_study directory")
    
    all_datasets = [d.name for d in ablation_dir.iterdir() if d.is_dir() and d.name.startswith("Dataset")]
    
    if selected_datasets:
        # Filter to only requested datasets
        selected = [f"Dataset{i}" for i in selected_datasets if f"Dataset{i}" in all_datasets]
        return sorted(selected)
    
    return sorted(all_datasets)

def get_ablated_terms(selected_datasets=None):
    """Get list of all unique ablated terms across datasets."""
    # Find ablation_study directory
    if os.path.exists("ablation_study"):
        ablation_dir = Path("ablation_study")
    elif os.path.exists("../ablation_study"):
        ablation_dir = Path("../ablation_study")
    else:
        raise FileNotFoundError("Could not find ablation_study directory")
    
    ablated_terms = set()
    
    for dataset in get_datasets(selected_datasets):
        dataset_dir = ablation_dir / dataset
        
        # Check for ablated_* directories at dataset level only (simplified structure)
        for item in dataset_dir.iterdir():
            if item.is_dir() and item.name.startswith("ablated_"):
                term = item.name.replace("ablated_", "")
                ablated_terms.add(term)
        
        # Also check validation results CSV for additional terms
        val_results_file = dataset_dir / f"{dataset}_validation_results.csv"
        if val_results_file.exists():
            try:
                df = pd.read_csv(val_results_file)
                if 'ablated_term' in df.columns:
                    for term in df['ablated_term'].unique():
                        ablated_terms.add(term)
            except Exception as e:
                print(f"Warning: Could not read {val_results_file}: {e}")
    
    return sorted(list(ablated_terms))

def load_benchmark_scores(dataset, ablated_term):
    """Load validation scores from benchmark.xlsx for a specific dataset and ablated term."""
    # Find ablation_study directory
    if os.path.exists("ablation_study"):
        ablation_base = "ablation_study"
    elif os.path.exists("../ablation_study"):
        ablation_base = "../ablation_study"
    else:
        return None
    
    # Look for benchmark file in simplified structure
    benchmark_file = Path(f"{ablation_base}/{dataset}/ablated_{ablated_term}/benchmark.xlsx")
    
    if not benchmark_file.exists():
        return None
    
    try:
        df = pd.read_excel(benchmark_file)
        # Filter for validation scores
        val_scores = df[df['is_validation'] == True]['score'].values
        return val_scores
    except Exception as e:
        print(f"Warning: Could not read {benchmark_file}: {e}")
        return None

def create_boxplot_figure(selected_datasets=None):
    """Create figure with boxplots of validation scores."""
    datasets = get_datasets(selected_datasets)
    ablated_terms = get_ablated_terms(selected_datasets)
    n_datasets = len(datasets)
    
    fig, axes = plt.subplots(1, n_datasets, figsize=(5*n_datasets, 7))
    if n_datasets == 1:
        axes = [axes]
    
    # Create consistent color mapping for all terms
    colors_dict = {}
    colors = plt.cm.Set3(np.linspace(0, 1, len(ablated_terms)))
    for j, term in enumerate(ablated_terms):
        colors_dict[term] = colors[j]
    
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        
        # Collect data for boxplot
        boxplot_data = []
        labels = []
        box_colors = []
        
        for term in ablated_terms:
            scores = load_benchmark_scores(dataset, term)
            if scores is not None and len(scores) > 0:
                boxplot_data.append(scores)
                labels.append(term)
                box_colors.append(colors_dict[term])
        
        if boxplot_data:
            # Create boxplot
            bp = ax.boxplot(boxplot_data, labels=labels, patch_artist=True)
            
            # Color the boxes with consistent colors
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.set_title(f'{dataset}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Validation Score', fontsize=10)
        ax.set_xlabel('Ablated Terms', fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(True, alpha=0.3)
        
        # Set consistent y-axis range for all subplots
        ax.set_ylim(0, 1)
    
    # Add a legend for the whole figure
    if datasets and any(load_benchmark_scores(dataset, term) is not None for dataset in datasets for term in ablated_terms):
        # Create legend handles
        legend_handles = []
        legend_labels = []
        
        for i, term in enumerate(ablated_terms):
            # Check if this term has data in any dataset
            has_data = any(load_benchmark_scores(dataset, term) is not None for dataset in datasets)
            if has_data:
                legend_handles.append(plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.7))
                legend_labels.append(term)
        
        # Add legend to the figure
        if legend_handles:
            fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                      ncol=min(len(legend_labels), 5), fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend
    
    # Ensure output goes to assessment directory
    script_dir = Path(__file__).parent
    output_file = script_dir / 'ablation_validation_scores_boxplots.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Boxplot figure saved: {output_file}")
    return fig

def main():
    """Main function to generate boxplot figure."""
    # Parse command line arguments for dataset selection
    selected_datasets = None
    if len(sys.argv) > 1:
        try:
            selected_datasets = [int(x) for x in sys.argv[1:]]
            print(f"Processing selected datasets: {selected_datasets}")
        except ValueError:
            print("Error: Dataset arguments must be integers (e.g., 1 3 5)")
            return
    
    print("Starting boxplot generation for ablation study...")
    print(f"Working directory: {os.getcwd()}")
    
    # Check if we can find ablation_study directory
    try:
        datasets = get_datasets(selected_datasets)
        ablated_terms = get_ablated_terms(selected_datasets)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run from a directory that contains 'ablation_study' or has '../ablation_study'")
        return
    
    print(f"\nFound {len(datasets)} datasets: {datasets}")
    print(f"Found {len(ablated_terms)} ablated terms: {ablated_terms}")
    
    if not datasets:
        print("Error: No datasets found or selected datasets don't exist")
        return
    
    if not ablated_terms:
        print("Error: No ablated terms found")
        return
    
    # Generate boxplot
    print("\nGenerating validation scores boxplots...")
    fig = create_boxplot_figure(selected_datasets)
    
    script_dir = Path(__file__).parent
    print("\n" + "="*60)
    print("BOXPLOT GENERATION COMPLETE!")
    print(f"Generated file: {script_dir / 'ablation_validation_scores_boxplots.png'}")
    print("="*60)

if __name__ == "__main__":
    main()