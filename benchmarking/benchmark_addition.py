from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ROOT_DIR = Path(__file__).resolve().parents[1]
ADDITION_DIR = ROOT_DIR / "addition_study"
OUT_DIR = Path(__file__).resolve().parent

DEFAULT_DATASETS = [f"Dataset{i}" for i in range(1, 6)]
METRICS = ["PCC", "RMSE", "JS", "SSIM"]


def extract_metrics_from_df(df: pd.DataFrame):
    """Extract train/val metric values from a dataframe as lists.

    Returns a structure: {"train": {metric: [values...]}, "val": {metric: [values...]}}
    Assumptions: each Excel has columns `PCC`, `RMSE`, `JS`, `SSIM`.
    The file may indicate split via the index (labels 'train'/'val') or a
    column like 'split'/'set'. If neither is present, all rows are treated
    as validation (`val`) values (useful when benchmark.xlsx contains many runs).
    """
    out = {"train": {m: [] for m in METRICS}, "val": {m: [] for m in METRICS}}
    # normalize column names
    df = df.rename(columns=lambda s: str(s).strip())

    # quick bail-out if metrics columns aren't present
    if not all(m in df.columns for m in METRICS):
        return out

    # 1) If index contains 'train'/'val' rows, take those rows
    idx_lower = [str(i).strip().lower() for i in df.index]
    if any(x in ("train", "val") for x in idx_lower):
        for row_label in df.index:
            rl = str(row_label).strip().lower()
            if rl in ("train", "val"):
                split = "train" if rl == "train" else "val"
                for m in METRICS:
                    try:
                        v = df.loc[row_label, m]
                        out[split][m].append(float(v))
                    except Exception:
                        pass
        return out

    # 1b) If boolean columns exist like 'is_training'/'is_validation', use them
    col_names_lower = {str(c).strip().lower(): c for c in df.columns}
    if 'is_training' in col_names_lower or 'is_validation' in col_names_lower:
        train_col = col_names_lower.get('is_training')
        val_col = col_names_lower.get('is_validation')
        for _, row in df.iterrows():
            try:
                is_train = bool(row[train_col]) if train_col is not None else False
            except Exception:
                is_train = False
            try:
                is_val = bool(row[val_col]) if val_col is not None else False
            except Exception:
                is_val = False
            if is_train:
                for m in METRICS:
                    try:
                        out['train'][m].append(float(row[m]))
                    except Exception:
                        pass
            if is_val:
                for m in METRICS:
                    try:
                        out['val'][m].append(float(row[m]))
                    except Exception:
                        pass
        return out

    # 2) If there's an explicit split-like column, collect per-row
    split_col = None
    for cand in ("split", "set", "type", "phase"):
        if cand in df.columns:
            split_col = cand
            break
    if split_col is not None:
        for _, row in df.iterrows():
            split = str(row[split_col]).strip().lower()
            if split in ("train", "val"):
                for m in METRICS:
                    try:
                        out[split][m].append(float(row[m]))
                    except Exception:
                        pass
        return out

    # 3) Fallback: treat all rows as validation runs
    for _, row in df.iterrows():
        for m in METRICS:
            try:
                out["val"][m].append(float(row[m]))
            except Exception:
                pass
    return out


def collect_addition_data(root: Path, datasets, study_name: str = "tangram_optuna_study", include_ablation_baseline=True):
    """Traverse addition_study and collect metrics per dataset and addition configuration.
    
    Also optionally retrieves baseline, vanilla, and full configurations from ablation_study.

    Returns: dict dataset -> dict addition_label -> {"train":{}, "val":{}}
    """
    results = {}
    split_counts = {}  # {ds: {"train": int, "val": int}}
    
    # Get the ablation_study directory (sibling to addition_study)
    ablation_root = root.parent / "ablation_study" if include_ablation_baseline else None
    
    for ds in datasets:
        results[ds] = {}
        split_counts[ds] = {"train": None, "val": None}
        
        # First, try to get baseline, vanilla, full from ablation_study if available
        if include_ablation_baseline and ablation_root and ablation_root.exists():
            ablation_ds_dir = ablation_root / ds
            if ablation_ds_dir.exists():
                # Map ablation study results to addition study naming
                ablation_mapping = {
                    "ablated_baseline": "added_baseline",
                    "ablated_vanilla": "added_vanilla", 
                    "ablated_full": "added_full"
                }
                
                for ablation_name, addition_name in ablation_mapping.items():
                    ablation_sub = ablation_ds_dir / ablation_name
                    if ablation_sub.exists():
                        bench_file = ablation_sub / "benchmark.xlsx"
                        if bench_file.exists():
                            try:
                                x = pd.read_excel(bench_file, sheet_name=0)
                                # Update split counts
                                cols_lower = {str(c).strip().lower(): c for c in x.columns}
                                tr_col = cols_lower.get('is_training')
                                va_col = cols_lower.get('is_validation')
                                if tr_col is not None and x[tr_col].dtype.kind in ('b','i','u','f'):
                                    try:
                                        tr_count = int(pd.Series(x[tr_col]).astype(bool).sum())
                                        prev = split_counts[ds]["train"]
                                        split_counts[ds]["train"] = tr_count if prev is None else max(prev, tr_count)
                                    except Exception:
                                        pass
                                if va_col is not None and x[va_col].dtype.kind in ('b','i','u','f'):
                                    try:
                                        va_count = int(pd.Series(x[va_col]).astype(bool).sum())
                                        prev = split_counts[ds]["val"]
                                        split_counts[ds]["val"] = va_count if prev is None else max(prev, va_count)
                                    except Exception:
                                        pass
                                
                                extracted = extract_metrics_from_df(x)
                                results[ds][addition_name] = extracted
                                logging.info("Retrieved %s from ablation_study for %s", ablation_name, ds)
                            except Exception as e:
                                logging.error("Failed to read %s: %s", bench_file, e)
        
        # Now collect from addition_study
        ds_dir = root / ds / study_name  # Add study_name subdirectory
        if not ds_dir.exists():
            logging.warning("Dataset folder not found: %s", ds_dir)
            continue
        for sub in sorted(ds_dir.iterdir()):
            if sub.is_dir() and sub.name.startswith("added_"):
                bench_file = sub / "benchmark.xlsx"
                if not bench_file.exists():
                    logging.warning("Missing benchmark.xlsx in %s", sub)
                    continue
                try:
                    # read first sheet
                    x = pd.read_excel(bench_file, sheet_name=0)
                    # update per-split gene counts if boolean flags exist
                    cols_lower = {str(c).strip().lower(): c for c in x.columns}
                    tr_col = cols_lower.get('is_training')
                    va_col = cols_lower.get('is_validation')
                    if tr_col is not None and x[tr_col].dtype.kind in ('b','i','u','f'):
                        try:
                            tr_count = int(pd.Series(x[tr_col]).astype(bool).sum())
                            prev = split_counts[ds]["train"]
                            split_counts[ds]["train"] = tr_count if prev is None else max(prev, tr_count)
                        except Exception:
                            pass
                    if va_col is not None and x[va_col].dtype.kind in ('b','i','u','f'):
                        try:
                            va_count = int(pd.Series(x[va_col]).astype(bool).sum())
                            prev = split_counts[ds]["val"]
                            split_counts[ds]["val"] = va_count if prev is None else max(prev, va_count)
                        except Exception:
                            pass

                    extracted = extract_metrics_from_df(x)
                    results[ds][sub.name] = extracted
                except Exception as e:
                    logging.error("Failed to read %s: %s", bench_file, e)
    return results, split_counts

def make_and_save_plots(results: dict, split: str = "train", datasets=None, split_counts=None):
    """Create a figure with rows (metrics) x cols (datasets) and save PNG.

    split: 'train' or 'val'
    """
    if datasets is None:
        datasets = DEFAULT_DATASETS
    sns.set(style="whitegrid")
    cols = len(datasets)
    rows = len(METRICS)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.0), squeeze=False)

    # Build a consistent palette across the entire figure (all datasets)
    all_additions = []
    for ds in datasets:
        all_additions.extend(list(results.get(ds, {}).keys()))
    all_additions = sorted(set(all_additions))
    if len(all_additions) == 0:
        logging.warning("No addition configurations found; plots will be empty")
    palette_colors = sns.color_palette("tab20", n_colors=max(1, len(all_additions)))
    addition_palette = {ab: palette_colors[i % len(palette_colors)] for i, ab in enumerate(all_additions)}
    # Display labels without the leading 'added_' prefix for readability
    addition_display = {ab: (ab[6:] if ab.startswith('added_') else ab) for ab in all_additions}
    
    for j, ds in enumerate(datasets):
        # ensure same addition order across metrics: directory order
        additions = list(results.get(ds, {}).keys())
        
        # Collect all data for this dataset first
        all_rows_data = []
        for ab in additions:
            for metric in METRICS:
                vals = results[ds].get(ab, {}).get(split, {}).get(metric, [])
                # only include numeric values
                vals_num = [float(v) for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
                if vals_num:
                    for v in vals_num:
                        all_rows_data.append({"addition": ab, "metric": metric, "value": v})
        
        # Remove outliers that are more than 3 standard deviations from the mean
        filtered_rows_data = []
        if all_rows_data:
            df_all = pd.DataFrame(all_rows_data)
            for method in df_all["addition"].unique():
                for metric in df_all["metric"].unique():
                    subset = df_all[(df_all["addition"] == method) & (df_all["metric"] == metric)]
                    if len(subset) > 1:  # Need at least 2 points to calculate std
                        values = subset["value"]
                        mean_val = values.mean()
                        std_val = values.std()
                        if std_val > 0:  # Avoid division by zero
                            # Keep values within 3 standard deviations
                            z_scores = abs(values - mean_val) / std_val
                            subset_filtered = subset[z_scores <= 3.0]
                            filtered_rows_data.extend(subset_filtered.to_dict('records'))
                        else:
                            filtered_rows_data.extend(subset.to_dict('records'))
                    else:
                        filtered_rows_data.extend(subset.to_dict('records'))
        
        for i, metric in enumerate(METRICS):
            ax = axes[i][j]
            labels = list(additions)

            rows_data = [r for r in filtered_rows_data if r["metric"] == metric]

            if rows_data:
                dfp = pd.DataFrame(rows_data)
                try:
                    sns.boxplot(x="addition", y="value", data=dfp, order=labels, ax=ax, palette=addition_palette)
                except Exception:
                    # fallback: draw simple matplotlib boxplot
                    data_for_plot = [dfp.loc[dfp["addition"] == ab, "value"].values for ab in labels]
                    non_empty = [d for d in data_for_plot if len(d) > 0]
                    if non_empty:
                        ax.boxplot(non_empty, positions=np.arange(len(non_empty)))
                        ax.set_xticks(np.arange(len(labels)))
                        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            else:
                # no numeric rows for any addition configuration
                ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=9, color="#777777")
                ax.set_xticks([])

            # remove x-axis label (don't show 'addition')
            ax.set_xlabel("")
            # show x-axis labels only on bottom row, hide on other rows
            if i == len(METRICS) - 1:  # bottom row
                ax.set_xticklabels([addition_display.get(label, label) for label in labels], rotation=45, ha="right", fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(metric)
            # set column title with dataset and number of columns (genes) if available
            if i == 0:
                n_per_split = None
                if split_counts is not None:
                    n_per_split = split_counts.get(ds, {}).get(split)
                n_text = n_per_split if isinstance(n_per_split, int) and n_per_split >= 0 else "?"
                ax.set_title(f"{ds}\n(n={n_text})")

    # Normalize y-axis limits within each row (metric)
    for i, metric in enumerate(METRICS):
        # Collect all y-limits from this row
        y_mins, y_maxs = [], []
        for j in range(cols):
            ax = axes[i][j]
            y_min, y_max = ax.get_ylim()
            y_mins.append(y_min)
            y_maxs.append(y_max)
        
        # Set uniform limits for all subplots in this row
        if y_mins and y_maxs:
            global_y_min = min(y_mins)
            global_y_max = max(y_maxs)
            for j in range(cols):
                axes[i][j].set_ylim(global_y_min, global_y_max)

    # Add global legend and caption
    import matplotlib.patches as mpatches
    legend_handles = [mpatches.Patch(facecolor=addition_palette[ab], edgecolor='black', label=addition_display[ab]) for ab in all_additions]
    ncol_legend = min(len(legend_handles), 6) if len(legend_handles) > 0 else 1
    plt.tight_layout()
    fig.subplots_adjust(top=0.88, bottom=0.14)
    if legend_handles:
        fig.legend(handles=legend_handles, loc='upper center', ncol=ncol_legend, frameon=True, title='Addition models')
    # Add explanatory caption
    caption = (
        "Each box shows the distribution of the metric for that addition model configuration."
    )
    fig.text(0.5, 0.04, caption, ha="center", fontsize=9)
    out_name = f"boxplot_{split}_addition.png"
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    logging.info("Saved %s", out_path)

def parse_cli_datasets(datasets_arg):
    """Parse CLI args like `1 2 3` or `Dataset1 Dataset2` into dataset names list."""
    if not datasets_arg:
        return DEFAULT_DATASETS
    argv = datasets_arg  # datasets_arg is now a list from argparse
    ds = []
    for a in argv:
        a = str(a).strip()
        if a.isdigit():
            n = int(a)
            ds_name = f"Dataset{n}"
            ds.append(ds_name)
        elif a.lower().startswith("dataset") and a[7:].isdigit():
            ds.append(a)
        else:
            # tolerate bare names like '1,2,3' or comma-separated
            if "," in a:
                for part in a.split(","):
                    p = part.strip()
                    if p.isdigit():
                        ds.append(f"Dataset{int(p)}")
            else:
                # unknown token: include as-is and let existence check warn
                ds.append(a)
    # remove duplicates while preserving order
    seen = set()
    out = []
    for x in ds:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out if out else DEFAULT_DATASETS


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark addition study results")
    parser.add_argument('datasets', nargs='*', help='Dataset numbers or names (e.g., 1 2 3 or Dataset1 Dataset2)')
    parser.add_argument('--study-name', default='tangram_optuna_study', help='Study name subdirectory')
    parser.add_argument('--no-ablation-baseline', action='store_true', 
                       help='Do not retrieve baseline/vanilla/full from ablation_study')
    
    if argv is None:
        argv = sys.argv[1:]
    
    args = parser.parse_args(argv)
    datasets = parse_cli_datasets(args.datasets)
    include_ablation = not args.no_ablation_baseline
    
    logging.info("Processing datasets: %s (study: %s, include_ablation: %s)", 
                datasets, args.study_name, include_ablation)
    
    results, split_counts = collect_addition_data(ADDITION_DIR, datasets, args.study_name, include_ablation)
    make_and_save_plots(results, split="train", datasets=datasets, split_counts=split_counts)
    make_and_save_plots(results, split="val", datasets=datasets, split_counts=split_counts)


if __name__ == "__main__":
    main()
