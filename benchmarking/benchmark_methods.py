from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import logging
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors



def rank_models_and_compute_acc(dataset_id: int) -> pd.DataFrame:
	"""
	Load DatasetX_validation_results.csv from ABLATION STUDY for the given dataset id (1..5),
	extract the four training metrics, rank ablated models per metric with the desired
	direction, normalize ranks by N, and compute ACC as the mean of normalized
	ranks across the four metrics.

	NOTE: This function only processes ablation study results, not addition study results.

	Ranking direction:
	- mean_train_SSIM: higher is better
	- mean_train_PCC: higher is better
	- mean_train_RMSE: lower is better
	- mean_train_JS: lower is better

	Rank scale:
	- Worst model gets 1, best model gets N (number of models)
	- Normalized rank = rank / N (in (0, 1])
	- ACC = (1/4) * sum(normalized ranks) = mean of normalized ranks

	Returns a DataFrame with index = `ablated_term` (if present) and
	columns = `ablated_term`, normalized ranks per metric (same metric names)
	plus an 'ACC' column. The `ablated_term` column is always included in the
	output if present in the CSV.
	"""

	if not isinstance(dataset_id, int) or dataset_id < 1:
		raise ValueError("dataset_id must be an integer >= 1 (expected 1..5)")

	repo_root = Path(__file__).resolve().parent.parent
	csv_path = (
		repo_root
		/ "ablation_study"
		/ f"Dataset{dataset_id}"
		/ f"Dataset{dataset_id}_validation_results.csv"
	)

	if not csv_path.exists():
		raise FileNotFoundError(f"Validation results not found: {csv_path}")

	ablation_results_df = pd.read_csv(csv_path)

	# Use `ablated_term` as identifier when present in ablation results
	if "ablated_term" in ablation_results_df.columns:
		ablation_results_df = ablation_results_df.set_index("ablated_term")

	# Define metrics and their ranking direction (higher/lower is better)
	training_metrics: Dict[str, str] = {
		"mean_train_SSIM": "higher",
		"mean_train_PCC": "higher",
		"mean_train_RMSE": "lower",
		"mean_train_JS": "lower",
	}

	# Check that all required metrics are present in ablation results
	missing_metrics = [col for col in training_metrics if col not in ablation_results_df.columns]
	if missing_metrics:
		raise KeyError(
			"Missing expected columns in ablation CSV: " + ", ".join(missing_metrics)
		)

	# Extract only the metric columns for ranking
	metrics_df = ablation_results_df[list(training_metrics.keys())].copy()

	num_models = len(metrics_df)
	if num_models == 0:
		raise ValueError("No ablated models found in the CSV for ranking")

	# Compute ranks per metric: worst model gets rank 1, best gets rank N
	model_ranks = {}
	for metric_col, better_direction in training_metrics.items():
		# For "higher is better" metrics: ascending=True (smallest value gets rank 1)
		# For "lower is better" metrics: ascending=False (largest value gets rank 1)
		rank_ascending = (better_direction == "higher")
		model_ranks[metric_col] = metrics_df[metric_col].rank(
			method="average", ascending=rank_ascending
		)

	ranks_df = pd.DataFrame(model_ranks, index=metrics_df.index)

	# Normalize ranks to [0, 1] range
	normalized_ranks_df = ranks_df / float(num_models)

	# ACC (Aggregate Composite Criterion) = mean of normalized ranks
	normalized_ranks_df["ACC"] = normalized_ranks_df.mean(axis=1)

	# Ensure `ablated_term` is included as a column in the output (if present)
	if normalized_ranks_df.index.name == "ablated_term":
		output_df = normalized_ranks_df.copy()
		# Add ablated_term as first column for easier reading
		output_df.insert(0, "ablated_term", output_df.index)
		return output_df

	return normalized_ranks_df


def load_other_methods(dataset_id: int) -> pd.DataFrame:
	"""
	Load `benchmarking/bench_data_train_only.xlsx` and keep only rows for the given dataset.
	Uses the Dataset-Data mapping to filter by 'Image-based' column.
	"""

	if not isinstance(dataset_id, int) or dataset_id < 1:
		raise ValueError("dataset_id must be an integer >= 1 (expected 1..5)")

	# Mapping from Dataset number to Data number
	dataset_to_data_map = {1: 1, 2: 3, 3: 6, 4: 8, 5: 9}

	repo_root = Path(__file__).resolve().parent.parent
	xlsx_path = repo_root / "benchmarking" / "bench_data_train_only.xlsx"
	
	df = pd.read_excel(xlsx_path)

	# Filter by Image-based column
	data_number = dataset_to_data_map[dataset_id]
	target_value = f"Data{data_number}"
	
	mask = df["Image-based"].astype(str).str.strip() == target_value
	filtered = df.loc[mask].copy()
	return filtered


__all__ = ["rank_models_and_compute_acc", "load_other_methods"]


# ------------------------
# ------------------------
# CLI helpers and entrypoint
# NOTE: This benchmarking script focuses on ABLATION STUDY results only
# ------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
PLOT_OUT = Path(__file__).resolve().parent / "boxplot_all_methods.png"
METRICS = ["PCC", "RMSE", "JS", "SSIM"]
ABLATION_COLOR = "#e45756"  # distinct color to highlight best ablation model


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	df.columns = [str(c).strip() for c in df.columns]
	# unify metric columns casing if needed
	col_map = {}
	for m in METRICS:
		for c in df.columns:
			if c.lower() == m.lower():
				col_map[c] = m
	# unify 'tool' column casing
	for c in df.columns:
		if str(c).strip().lower() == "tool":
			col_map[c] = "tool"
	if col_map:
		df = df.rename(columns=col_map)
	return df


def _collect_other_methods_records(df: pd.DataFrame, dataset_label: str) -> List[dict]:
	"""Collect raw metric values per 'tool' to enable boxplots (no averaging)."""
	records: List[dict] = []
	df = _standardize_columns(df)
	if "tool" not in df.columns:
		return records
	present_metrics = [m for m in METRICS if m in df.columns]
	if not present_metrics:
		return records
	for _, row in df.iterrows():
		method = str(row["tool"]) if pd.notna(row.get("tool")) else None
		if not method:
			continue
		for m in present_metrics:
			try:
				v = float(row[m])
			except Exception:
				continue
			if pd.notna(v):
				records.append({
					"dataset": dataset_label,
					"method": method,
					"metric": m,
					"value": v,
					"source": "other",
				})
	return records


def parse_cli_dataset_ids(argv: List[str] | None) -> List[int]:
	"""
	Parse CLI args like `1 2 3` or `Dataset1 Dataset2` or `1,2,3` into a list of ints.
	If none provided, default to [1, 2, 3, 4, 5].
	"""
	if not argv:
		return [1, 2, 3, 4, 5]
	out: List[int] = []
	for a in argv:
		a = str(a).strip()
		if "," in a:
			for part in a.split(","):
				p = part.strip()
				if p.isdigit():
					n = int(p)
					if n not in out:
						out.append(n)
				elif p.lower().startswith("dataset") and p[7:].isdigit():
					n = int(p[7:])
					if n not in out:
						out.append(n)
		else:
			if a.isdigit():
				n = int(a)
				if n not in out:
					out.append(n)
			elif a.lower().startswith("dataset") and a[7:].isdigit():
				n = int(a[7:])
				if n not in out:
					out.append(n)
	return out if out else [1, 2, 3, 4, 5]


def main(argv: List[str] | None = None) -> None:
	logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
	if argv is None:
		argv = sys.argv[1:]
	dataset_ids = parse_cli_dataset_ids(argv)
	logging.info("Selected datasets: %s", dataset_ids)

	# Prepare figure layout ahead of data loading: rows=metrics, cols=datasets
	sns.set(style="whitegrid")
	datasets_labels = [f"Dataset{did}" for did in dataset_ids]
	cols = len(datasets_labels)
	rows = len(METRICS)
	fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 3.0), squeeze=False)

	# Stable color function to ensure consistent colors across the figure
	# Use 10 distinct, moderately bright colors
	palette_base = [
		"#1f77b4",  # blue
		"#ff7f0e",  # orange
		"#2ca02c",  # green
		"#d62728",  # red
		"#9467bd",  # purple
		"#8c564b",  # brown
		"#e377c2",  # pink
		"#7f7f7f",  # gray
		"#bcbd22",  # olive
		"#17becf"   # cyan
	]
	def color_for_method(name: str) -> tuple:
		if name is None:
			return (0.7, 0.7, 0.7)
		# Highlight ablation models with distinct color
		if isinstance(name, str) and name.strip() != "" and name.lower().startswith("ablated_"):
			return mcolors.to_rgb(ABLATION_COLOR)
		# use hash-based stable index with 10-color palette
		idx = abs(hash(name)) % len(palette_base)
		return mcolors.to_rgb(palette_base[idx])

	methods_seen: set[str] = set()
	ds_training_counts: dict[str, int] = {}

	# Process each dataset: rank ablation study models and compare with other methods
	for dataset_id in dataset_ids:
		try:
			# Load and rank ablation study results for this dataset
			ablation_ranks_df = rank_models_and_compute_acc(dataset_id)

			# Save ablation model rankings to CSV
			dataset_dir = ROOT_DIR / "ablation_study" / f"Dataset{dataset_id}"
			dataset_dir.mkdir(parents=True, exist_ok=True)
			ranks_output_csv = dataset_dir / f"Dataset{dataset_id}_ranks.csv"
			ablation_ranks_df.to_csv(ranks_output_csv, index=False)
			logging.info("Saved ablation ranks CSV: %s", ranks_output_csv)


			# Collect plotting data for this dataset
			plot_records: List[dict] = []  # Format: {dataset, method, metric, value, source}

			# Load other baseline methods for comparison
			other_methods_df = load_other_methods(dataset_id)
			logging.info(
				"Loaded other methods for Dataset %s with shape %s",
				dataset_id,
				tuple(other_methods_df.shape),
			)
			# Save other methods data alongside ablation ranks
			other_methods_csv = dataset_dir / f"Dataset{dataset_id}_other_methods.csv"
			other_methods_df.to_csv(other_methods_csv, index=False)
			logging.info("Saved other methods CSV: %s", other_methods_csv)
			# Collect raw metric values from other methods
			plot_records.extend(_collect_other_methods_records(other_methods_df, f"Dataset{dataset_id}"))

			# Find best ablation model by ACC score
			if "ACC" in ablation_ranks_df.columns and len(ablation_ranks_df) > 0:
				# Prefer ablated_term column for identification if available
				identifier_col = "ablated_term" if "ablated_term" in ablation_ranks_df.columns else None
				best_model_idx = ablation_ranks_df["ACC"].idxmax()
				best_acc_score = float(ablation_ranks_df.loc[best_model_idx, "ACC"]) if best_model_idx is not None else float("nan")
				best_model_name = (
					ablation_ranks_df.loc[best_model_idx, identifier_col]
					if (identifier_col is not None and best_model_idx is not None)
					else str(best_model_idx)
				)
				logging.info("Dataset %s best ablation model by ACC: %s (ACC=%.4f)", dataset_id, best_model_name, best_acc_score)

			# Read benchmark.xlsx from the best ablation folder
			ablation_folder_name = best_model_name if str(best_model_name).startswith("ablated_") else f"ablated_{best_model_name}"
			best_model_dir = dataset_dir / ablation_folder_name
			benchmark_file = best_model_dir / "benchmark.xlsx"
			
			benchmark_df = pd.read_excel(benchmark_file)
			logging.info(
				"Loaded benchmark.xlsx for Dataset %s best model '%s' with shape %s",
				dataset_id,
				ablation_folder_name,
				tuple(benchmark_df.shape),
			)
			
			# Record training gene count
			training_gene_count = int(pd.Series(benchmark_df['is_training']).astype(bool).sum())
			ds_training_counts[f"Dataset{dataset_id}"] = training_gene_count
			
			# Extract metric values for the ablation best model
			standardized_benchmark = _standardize_columns(benchmark_df)
			for metric in METRICS:
				if metric in standardized_benchmark.columns:
					metric_series = pd.to_numeric(standardized_benchmark[metric], errors="coerce").dropna()
					for value in metric_series.tolist():
						plot_records.append({
							"dataset": f"Dataset{dataset_id}",
							"method": ablation_folder_name.replace("ablated_", "", 1),
							"metric": metric,
							"value": float(value),
							"source": "ablation",
						})
			else:
				logging.warning("ACC column missing or empty DataFrame for Dataset %s", dataset_id)


			# Plot this dataset column
			plot_df_ds = pd.DataFrame(plot_records)
			
			# Remove outliers that are more than 3 standard deviations from the mean
			if not plot_df_ds.empty:
				filtered_records = []
				for method in plot_df_ds["method"].unique():
					for metric in plot_df_ds["metric"].unique():
						subset = plot_df_ds[(plot_df_ds["method"] == method) & (plot_df_ds["metric"] == metric)]
						if len(subset) > 1:  # Need at least 2 points to calculate std
							values = subset["value"]
							mean_val = values.mean()
							std_val = values.std()
							if std_val > 0:  # Avoid division by zero
								# Keep values within 3 standard deviations
								z_scores = abs(values - mean_val) / std_val
								subset_filtered = subset[z_scores <= 3.0]
								filtered_records.append(subset_filtered)
							else:
								filtered_records.append(subset)
						else:
							filtered_records.append(subset)
				
				if filtered_records:
					plot_df_ds = pd.concat(filtered_records, ignore_index=True)
			
			methods_present = sorted(plot_df_ds["method"].dropna().unique().tolist()) if not plot_df_ds.empty else []
			methods_seen.update(methods_present)
			# Determine order: other methods then ablation last
			a_methods = [m for m in methods_present if isinstance(m, str) and m.lower().startswith("ablated_")]
			other_methods_present = [m for m in methods_present if m not in a_methods]
			ordered_methods = sorted(other_methods_present) + sorted([m for m in a_methods if m not in other_methods_present])
			# palette mapping for present methods
			palette_ds = {m: color_for_method(m) for m in ordered_methods}

			j = datasets_labels.index(f"Dataset{dataset_id}")
			for i, metric in enumerate(METRICS):
				ax = axes[i][j]
				mf = plot_df_ds[plot_df_ds["metric"] == metric]

				present = [m for m in ordered_methods if m in mf["method"].unique().tolist()]
				if present:
					sns.boxplot(data=mf, x="method", y="value", ax=ax, order=present, showcaps=True, fliersize=2, palette=palette_ds)
					ax.set_xlabel("")
					if j == 0:
						ax.set_ylabel(metric)
					else:
						ax.set_ylabel("")
					# show x-axis labels only on bottom row, hide on other rows
					if i == len(METRICS) - 1:  # bottom row
						ax.set_xticklabels([m for m in present], rotation=45, ha="right", fontsize=8)
					else:
						ax.set_xticklabels([])
				else:
					ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=9, color="#777777")
					ax.set_xticks([])
					if j == 0:
						ax.set_ylabel(metric)
				
				# set column title at top row with training gene count
				if i == 0:
					n_text = ds_training_counts.get(f"Dataset{dataset_id}", "?")
					ax.set_title(f"Dataset{dataset_id}\n(n={n_text})")

		except Exception as e:
			logging.error("Dataset %s top-level failure: %s", dataset_id, e)

	# After processing all datasets, finalize legend/caption and save the figure
	# Build legend mapping colors to methods seen
	ordered_methods_global = sorted(m for m in methods_seen if m is not None)
	# ensure any ablation-labeled method goes last
	ablation_methods = [m for m in ordered_methods_global if isinstance(m, str) and m.lower().startswith("ablated_")]
	other_methods = [m for m in ordered_methods_global if m not in ablation_methods]
	ordered_methods_global = other_methods + ablation_methods
	legend_handles = [mpatches.Patch(facecolor=color_for_method(m), edgecolor='black', label=m) for m in ordered_methods_global]
	fig.tight_layout()
	fig.subplots_adjust(top=0.88, bottom=0.15)
	if legend_handles:
		ncol_legend = min(len(legend_handles), 6)
		fig.legend(handles=legend_handles, loc='upper center', ncol=ncol_legend, frameon=True, title='Methods')
	caption = (
		"Each box shows the distribution of the metric across runs per method. "
		"Colors label methods (legend above). The ablation entry is the best ablated model "
		"for each dataset."
	)
	fig.text(0.5, 0.04, caption, ha="center", fontsize=9)
	fig.savefig(PLOT_OUT, dpi=200, bbox_inches="tight")
	logging.info("Saved %s", PLOT_OUT)


if __name__ == "__main__":
	main()

