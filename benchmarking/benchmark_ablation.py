from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ROOT_DIR = Path(__file__).resolve().parents[1]
ABLATION_DIR = ROOT_DIR / "ablation_study"
OUT_DIR = Path(__file__).resolve().parent

DEFAULT_DATASETS = [f"Dataset{i}" for i in range(1, 6)]
METRICS = ["PCC", "RMSE", "JS", "SSIM"]

# ---------------------------------------------------------
# Pretty λ notation mapping (same as polyfit_montage.py)
# ---------------------------------------------------------
PRETTY_NAMES = {
    "ablated_baseline": r"baseline",
    "ablated_vanilla": r"vanilla",
    "ablated_full": r"full",
    "ablated_lambda_g2": r"ablated $\lambda_{\mathrm{s/g}}$",
    "ablated_lambda_d": r"ablated $\lambda_{\mathrm{KL}}$",
    "ablated_lambda_geary": r"ablated $\lambda_{\mathrm{geary}}$",
    "ablated_lambda_getis_ord": r"ablated $\lambda_{\mathrm{getis\_ord}}$",
    "ablated_lambda_moran": r"ablated $\lambda_{\mathrm{moran}}$",
    "ablated_lambda_neighborhood_g1": r"ablated $\lambda_{G}$",
    "ablated_lambda_r": r"ablated $\lambda_{\mathrm{entropy}}$",
}

# ---------------------------------------------------------
# Ordering for consistency (same as polyfit_montage.py)
# ---------------------------------------------------------
ORDER = [
    "ablated_baseline",
    "ablated_vanilla",
    "ablated_full",
    "ablated_lambda_g2",
    "ablated_lambda_d",
    "ablated_lambda_r",
    "ablated_lambda_geary",
    "ablated_lambda_moran",
    "ablated_lambda_getis_ord",
    "ablated_lambda_neighborhood_g1",
]

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



def collect_ablation_data(root: Path, datasets):
    """Traverse ablation_study and collect metrics per dataset and ablation.

    Returns: dict dataset -> dict ablation_label -> {"train":{}, "val":{}}
    """
    results = {}
    split_counts = {}  # {ds: {"train": int, "val": int}}
    for ds in datasets:
        results[ds] = {}
        split_counts[ds] = {"train": None, "val": None}
        ds_dir = root / ds
        if not ds_dir.exists():
            logging.warning("Dataset folder not found: %s", ds_dir)
            continue
        for sub in sorted(ds_dir.iterdir()):
            if sub.is_dir() and sub.name.startswith("ablated_"):
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
    all_ablations = []
    for ds in datasets:
        all_ablations.extend(list(results.get(ds, {}).keys()))
    all_ablations = sorted(set(all_ablations))
    if len(all_ablations) == 0:
        logging.warning("No ablations found; plots will be empty")
    palette_colors = sns.color_palette("tab20", n_colors=max(1, len(all_ablations)))
    ablation_palette = {ab: palette_colors[i % len(palette_colors)] for i, ab in enumerate(all_ablations)}
    # Final display-name mapping
    # Keep only those in ORDER, preserving ORDER
    ordered_ablations = [ab for ab in ORDER if ab in all_ablations]

    ablation_display = {
        ab: PRETTY_NAMES.get(ab, ab)
        for ab in ordered_ablations
    }

    for j, ds in enumerate(datasets):
        # ensure same ablation order across metrics: directory order
        ablations = list(results.get(ds, {}).keys())
        
        # Collect all data for this dataset first
        all_rows_data = []
        for ab in ablations:
            for metric in METRICS:
                vals = results[ds].get(ab, {}).get(split, {}).get(metric, [])
                # only include numeric values
                vals_num = [float(v) for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
                if vals_num:
                    for v in vals_num:
                        all_rows_data.append({"ablation": ab, "metric": metric, "value": v})
        
        # Remove outliers that are more than 3 standard deviations from the mean
        filtered_rows_data = []
        if all_rows_data:
            df_all = pd.DataFrame(all_rows_data)
            for method in df_all["ablation"].unique():
                for metric in df_all["metric"].unique():
                    subset = df_all[(df_all["ablation"] == method) & (df_all["metric"] == metric)]
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
            # final ordered labels = intersection of ORDER with actual present ablations
            final_labels = [ab for ab in ORDER if ab in ablations]

            rows_data = [r for r in filtered_rows_data if r["metric"] == metric]

            if rows_data:
                dfp = pd.DataFrame(rows_data)
                sns.boxplot(x="ablation", y="value", data=dfp, order=ORDER, ax=ax, palette=ablation_palette, fliersize=False)
            else:
                # no numeric rows for any ablation
                ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=9, color="#777777")
                ax.set_xticks([])

            # remove x-axis label (don't show 'ablation')
            ax.set_xlabel("")
            # show x-axis labels only on bottom row, hide on other rows
            if i == len(METRICS) - 1:  # bottom row
                ax.set_xticks(range(len(final_labels)))
                ax.set_xticklabels(
                    [ablation_display.get(ab, ab) for ab in final_labels],
                    rotation=45,
                    ha="right",
                    fontsize=8,
                )
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

    # add overall caption explaining the ablated_* wildcard meaning
    # add global legend mapping colors to ablation names
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(
            facecolor=ablation_palette[ab],
            edgecolor="black",
            label=ablation_display.get(ab, ab)
        )
        for ab in final_labels
    ]
    ncol_legend = min(len(legend_handles), 5) if len(legend_handles) > 0 else 1
    plt.tight_layout()
    fig.subplots_adjust(top=0.88, bottom=0.14)
    if legend_handles:
        fig.legend(handles=legend_handles, loc='upper center', ncol=ncol_legend, frameon=True, title='Models')

    out_name = f"boxplot_{split}_ablation.png"
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    logging.info("Saved %s", out_path)

def parse_cli_datasets(argv):
    """Parse CLI args like `1 2 3` or `Dataset1 Dataset2` into dataset names list."""
    if not argv:
        return DEFAULT_DATASETS
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
    if argv is None:
        argv = sys.argv[1:]
    datasets = parse_cli_datasets(argv)
    logging.info("Selected datasets: %s", datasets)
    results, split_counts = collect_ablation_data(ABLATION_DIR, datasets)
    make_and_save_plots(results, split="train", datasets=datasets, split_counts=split_counts)
    make_and_save_plots(results, split="val", datasets=datasets, split_counts=split_counts)


if __name__ == "__main__":
    main()
