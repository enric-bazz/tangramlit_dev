import pandas as pd
import os


def main():
    # Load the full benchmark data (normalize index to lowercase for robust matching)
    bench_data_full = pd.read_excel('benchmarking/bench_data_full.xlsx', index_col=0)
    bench_data_full.index = bench_data_full.index.astype(str).str.lower()

    # Mapping from Dataset number to Data number
    dataset_to_data_map = {1: 1, 2: 3, 3: 6, 4: 8, 5: 9}

    # Collect filtered data for each dataset
    all_filtered_data = []

    # Cycle through Dataset1 to Dataset5 (inclusive)
    for i in range(1, 6):
        dataset_name = f'Dataset{i}'
        benchmark_path = f'ablation_study/{dataset_name}/ablated_baseline/benchmark.xlsx'
        
        if os.path.exists(benchmark_path):
            # Load the benchmark file and filter by is_training=True
            trained_genes_df = pd.read_excel(benchmark_path, index_col=0)
            training_genes_df = trained_genes_df[trained_genes_df['is_training'] == True]
            trained_genes_lower = set(training_genes_df.index.astype(str).str.lower())
            
            # Filter bench_data_full for this specific dataset
            data_number = dataset_to_data_map[i]
            data_name = f'Data{data_number}'
            dataset_bench_data = bench_data_full[bench_data_full['Image-based'] == data_name]
            
            # Find common genes between training genes and dataset bench data
            common_genes = set(dataset_bench_data.index).intersection(trained_genes_lower)
            dataset_filtered = dataset_bench_data.loc[sorted(common_genes)]
            
            all_filtered_data.append(dataset_filtered)
            print(f"Loaded {len(trained_genes_lower)} training genes from {dataset_name}")
            print(f"Found {len(common_genes)} common genes for {dataset_name}")
        else:
            print(f"Warning: {benchmark_path} not found")

    # Combine all filtered data
    if all_filtered_data:
        bench_data_train_only = pd.concat(all_filtered_data, axis=0)
    else:
        bench_data_train_only = pd.DataFrame()

    print(f"\nOriginal bench_data_full shape: {bench_data_full.shape}")
    print(f"Filtered bench_data_train_only shape: {bench_data_train_only.shape}")

    # Write the filtered dataframe (index already lowercase unified)
    bench_data_train_only.to_excel('benchmarking/bench_data_train_only.xlsx')
    print(f"\nSaved filtered data to benchmarking/bench_data_train_only.xlsx")


if __name__ == "__main__":
    main()
