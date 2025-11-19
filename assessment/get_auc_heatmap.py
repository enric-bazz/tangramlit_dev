#!/usr/bin/env python3
"""
Heatmap generation script for ablation study validation AUC results.

This script generates:
1. A heatmap matrix (n datasets x n ablated models) showing validation AUC values
2. A CSV file with all the aggregated validation AUC values
3. Prints a summary table

Output directory: root_dir/assessment (working directory of this script)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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



def load_validation_auc_matrix(selected_datasets=None):
    """Load validation AUC values for all dataset-ablated_term combinations."""
    datasets = get_datasets(selected_datasets)
    ablated_terms = get_ablated_terms(selected_datasets)
    
    # Find ablation_study directory
    if os.path.exists("ablation_study"):
        ablation_base = "ablation_study"
    elif os.path.exists("../ablation_study"):
        ablation_base = "../ablation_study"
    else:
        raise FileNotFoundError("Could not find ablation_study directory")
    
    # Initialize matrix with NaN
    auc_matrix = np.full((len(datasets), len(ablated_terms)), np.nan)
    auc_df = pd.DataFrame(auc_matrix, index=datasets, columns=ablated_terms)
    
    for i, dataset in enumerate(datasets):
        # Look for validation results file
        val_results_file = Path(f"{ablation_base}/{dataset}/{dataset}_validation_results.csv")
        
        if val_results_file.exists():
            try:
                df = pd.read_csv(val_results_file)
                if 'ablated_term' in df.columns and 'auc_val' in df.columns:
                    for _, row in df.iterrows():
                        term = row['ablated_term']
                        auc_val = row['auc_val']
                        if term in ablated_terms:
                            j = ablated_terms.index(term)
                            auc_df.iloc[i, j] = auc_val
                else:
                    print(f"Warning: Required columns not found in {val_results_file}")
            except Exception as e:
                print(f"Warning: Could not read {val_results_file}: {e}")
        else:
            print(f"Warning: No validation results file found for {dataset}")
    
    return auc_df



def create_heatmap_figure(auc_df):
    """Create heatmap of validation AUC values."""
    # Remove completely empty rows and columns
    auc_df_clean = auc_df.dropna(how='all', axis=0).dropna(how='all', axis=1)
    
    if auc_df_clean.empty:
        print("Warning: No valid AUC data found for heatmap")
        return None
    
    plt.figure(figsize=(max(8, len(auc_df_clean.columns)), max(6, len(auc_df_clean.index))))
    
    # Create heatmap
    mask = auc_df_clean.isna()
    sns.heatmap(auc_df_clean, 
                annot=True, 
                fmt='.3f', 
                cmap='RdYlBu_r', 
                center=0.5,
                mask=mask,
                cbar_kws={'label': 'Validation AUC'},
                square=False)
    
    plt.title('Validation AUC Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Ablated Terms', fontsize=12)
    plt.ylabel('Datasets', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Ensure output goes to assessment directory
    script_dir = Path(__file__).parent
    output_file = script_dir / 'ablation_auc_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Heatmap figure saved: {output_file}")
    return plt.gcf()

def save_auc_csv(auc_df):
    """Save the AUC matrix to CSV and print summary table."""
    # Save to CSV in assessment directory
    script_dir = Path(__file__).parent
    csv_file = script_dir / 'ablation_auc_matrix.csv'
    auc_df.to_csv(csv_file)
    print(f"AUC matrix saved to: {csv_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("VALIDATION AUC SUMMARY TABLE")
    print("="*80)
    
    # Remove completely empty rows and columns for display
    auc_display = auc_df.dropna(how='all', axis=0).dropna(how='all', axis=1)
    
    if not auc_display.empty:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', '{:.3f}'.format)
        print(auc_display)
        
        # Summary statistics
        print("\n" + "-"*40)
        print("SUMMARY STATISTICS")
        print("-"*40)
        
        print(f"Available datasets: {len(auc_display.index)}")
        print(f"Available ablated terms: {len(auc_display.columns)}")
        print(f"Total combinations: {auc_display.size}")
        print(f"Valid combinations: {auc_display.notna().sum().sum()}")
        print(f"Missing combinations: {auc_display.isna().sum().sum()}")
        
        if auc_display.notna().any().any():
            valid_values = auc_display.values[~np.isnan(auc_display.values)]
            print(f"Mean AUC: {np.mean(valid_values):.3f}")
            print(f"Std AUC: {np.std(valid_values):.3f}")
            print(f"Min AUC: {np.min(valid_values):.3f}")
            print(f"Max AUC: {np.max(valid_values):.3f}")
        
        # Best performing combinations
        print("\n" + "-"*40)
        print("TOP 10 PERFORMING COMBINATIONS")
        print("-"*40)
        
        # Flatten the dataframe to get top combinations
        flattened = []
        for dataset in auc_display.index:
            for term in auc_display.columns:
                value = auc_display.loc[dataset, term]
                if not np.isnan(value):
                    flattened.append({
                        'Dataset': dataset,
                        'Ablated_Term': term,
                        'AUC_Val': value
                    })
        
        if flattened:
            top_df = pd.DataFrame(flattened).sort_values('AUC_Val', ascending=False).head(10)
            print(top_df.to_string(index=False, float_format='{:.3f}'.format))
    else:
        print("No valid AUC data found!")

def main():
    """Main function to generate heatmap and AUC analysis."""
    # Parse command line arguments for dataset selection
    selected_datasets = None
    if len(sys.argv) > 1:
        try:
            selected_datasets = [int(x) for x in sys.argv[1:]]
            print(f"Processing selected datasets: {selected_datasets}")
        except ValueError:
            print("Error: Dataset arguments must be integers (e.g., 1 3 5)")
            return
    
    print("Starting heatmap generation for ablation study...")
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
    
    # Load validation AUC matrix
    print("\nLoading validation AUC data...")
    auc_df = load_validation_auc_matrix(selected_datasets)
    
    # Generate heatmap
    print("\nGenerating AUC heatmap...")
    fig = create_heatmap_figure(auc_df)
    
    # Save AUC data and generate summary
    print("\nSaving AUC data and generating summary...")
    save_auc_csv(auc_df)
    
    script_dir = Path(__file__).parent
    print("\n" + "="*80)
    print("HEATMAP GENERATION COMPLETE!")
    print("="*80)
    print("Generated files in assessment directory:")
    print(f"- {script_dir / 'ablation_auc_heatmap.png'}")
    print(f"- {script_dir / 'ablation_auc_matrix.csv'}")
    print("="*80)

if __name__ == "__main__":
    main()
