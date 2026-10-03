import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, chi2_contingency

def run_conditional_validation(real_path, synthetic_path, output_dir_base):

    # ============================================================
    # Helper: make directory
    # ============================================================
    os.makedirs(output_dir_base, exist_ok=True)
    print("Base output folder:", output_dir_base)

    # ============================================================
    # Load datasets
    # ============================================================
    real = pd.read_csv(real_path, low_memory=False)
    synthetic = pd.read_csv(synthetic_path, low_memory=False)

    # Strip whitespace from column names so Mach_number / Time_in_cycles match cleanly
    real.columns = [c.strip() for c in real.columns]
    synthetic.columns = [c.strip() for c in synthetic.columns]

    print("\nLoaded datasets:")
    print("Real columns:", list(real.columns))
    print("Synthetic columns:", list(synthetic.columns))

    # ============================================================
    # Enforce numeric dtypes for key axes (if present)
    # ============================================================
    common_cols = [c for c in real.columns if c in synthetic.columns]
    if not common_cols:
        raise ValueError("No common columns found between real and synthetic datasets.")

    real = real[common_cols].copy()
    synthetic = synthetic[common_cols].copy()

    for col in common_cols:
        real[col] = pd.to_numeric(real[col], errors="ignore")
        synthetic[col] = pd.to_numeric(synthetic[col], errors="ignore")

    common_numeric = [
        col for col in common_cols
        if pd.api.types.is_numeric_dtype(real[col]) and pd.api.types.is_numeric_dtype(synthetic[col])
    ]

    if not common_numeric:
        raise ValueError("No shared numeric columns found.")

    print("\nShared numeric columns available for filtering:")
    for i, col in enumerate(common_numeric, start=1):
        print(f"{i}. {col}")


    # ============================================================
    # Interactive multi-condition filter (Altitude, Mach_number, TRA)
    # ============================================================

    def ask_float(prompt_text):
        while True:
            try:
                return float(input(prompt_text).strip())
            except ValueError:
                print("Enter a valid number.")

    def parse_selected_columns(user_input, available_cols):
        selected = []
        if not user_input.strip():
            return selected

        for item in user_input.split(","):
            item = item.strip()

            if item.isdigit():
                idx = int(item) - 1
                if 0 <= idx < len(available_cols):
                    col_name = available_cols[idx]
                    if col_name not in selected:
                        selected.append(col_name)
            else:
                if item in available_cols and item not in selected:
                    selected.append(item)

        return selected

    print("\nConditional Filter Selection")
    print("Enter column numbers or names, comma-separated.")
    print("Example: 1,3 or Altitude,Mach_number")

    selected_input = input("Columns to filter by (press Enter for no filter): ")
    selected_filter_cols = parse_selected_columns(selected_input, common_numeric)

    if not selected_filter_cols:
        print("No filters selected. Using full datasets.")
        real_filt = real.copy()
        synthetic_filt = synthetic.copy()
        filter_desc = "unfiltered"
    else:
        filter_settings = {}

        for col in selected_filter_cols:
            center = ask_float(f"Target value for {col}: ")
            tol = ask_float(f"Tolerance for {col}: ")
            filter_settings[col] = (center, tol)

        mask_real = pd.Series(True, index=real.index)
        mask_syn = pd.Series(True, index=synthetic.index)

        for col, (center, tol) in filter_settings.items():
            low, high = center - tol, center + tol
            mask_real &= real[col].between(low, high)
            mask_syn &= synthetic[col].between(low, high)

        real_filt = real[mask_real].copy()
        synthetic_filt = synthetic[mask_syn].copy()

        desc_parts = [f"{col}[{center:.3g}±{tol:.3g}]" for col, (center, tol) in filter_settings.items()]
        filter_desc = "_".join(desc_parts)

    print("\nFiltered dataset sizes:")
    print(f"Real filtered shape:       {real_filt.shape}")
    print(f"Synthetic filtered shape:  {synthetic_filt.shape}")

    if real_filt.empty or synthetic_filt.empty:
        raise ValueError("Filtered real or synthetic dataset is empty. Try different centers/tolerances or filters.")

    # ============================================================
    # Align common columns & consistent order (after filtering)
    # ============================================================
    common_cols = [c for c in real_filt.columns if c in synthetic_filt.columns]

    key_first = [c for c in ["unit", "Unit", "engine_id", "Engine_ID", "Time_in_cycles", "RUL"]
                if c in common_cols]
    the_rest = [c for c in common_cols if c not in key_first]
    common_cols = key_first + the_rest

    real_filt = real_filt[common_cols].copy()
    synthetic_filt = synthetic_filt[common_cols].copy()

    # Downsample to have same number of rows
    if len(real_filt) > len(synthetic_filt):
        real_filt = real_filt.sample(n=len(synthetic_filt), random_state=42).reset_index(drop=True)
    elif len(real_filt) < len(synthetic_filt):
        synthetic_filt = synthetic_filt.sample(n=len(real_filt), random_state=42).reset_index(drop=True)

    print("\n After alignment/downsampling:")
    print(f"Real filtered shape:       {real_filt.shape}")
    print(f"Synthetic filtered shape:  {synthetic_filt.shape}")

    # ============================================================
    # Set condition-specific output dir
    # ============================================================
    # Sanitize filter description for folder name
    safe_filter_desc = filter_desc.replace(" ", "").replace("[", "").replace("]", "").replace("±", "_")
    if not safe_filter_desc:
        safe_filter_desc = "unfiltered"

    output_dir = os.path.join(output_dir_base, safe_filter_desc)
    os.makedirs(output_dir, exist_ok=True)
    print("Condition-specific outputs will be saved in:", output_dir)

    # Helper: numeric-aware sorting for labels like s1, s2, ..., s10
    num_label_regex = re.compile(r"^([A-Za-z_]*)(\d+)$")
    def sort_key(label):
        s = str(label)
        m = num_label_regex.match(s)
        if m:
            prefix, num = m.group(1), int(m.group(2))
            return (prefix, num)
        return (s, float("inf"))

    # ============================================================
    # 1. Summary statistics (numeric only, FILTERED)
    # ============================================================
    print("\n=== Summary Statistics Comparison (Filtered) ===")

    num_cols = real_filt.select_dtypes(include="number").columns.intersection(
        synthetic_filt.select_dtypes(include="number").columns
    )

    compare = pd.DataFrame({
        "real_mean": real_filt[num_cols].mean(),
        "synthetic_mean": synthetic_filt[num_cols].mean(),
        "real_std": real_filt[num_cols].std(),
        "synthetic_std": synthetic_filt[num_cols].std(),
        "real_min": real_filt[num_cols].min(),
        "synthetic_min": synthetic_filt[num_cols].min(),
        "real_max": real_filt[num_cols].max(),
        "synthetic_max": synthetic_filt[num_cols].max(),
    })
    compare.to_csv(os.path.join(output_dir, "summary_filtered.csv"))

    print("Conditional validation saved to:", output_dir)
        
    print(compare)
 
    print("Saved summary comparison as summary_comparison_filtered.csv")

    # ============================================================
    # 2. Distribution comparison (numeric; shared bins, FILTERED)
    # ============================================================
    print("\n=== Plotting distribution comparisons (filtered; colored outlines, no tables) ===")

    def pretty_sensor_title(col):
        m = re.match(r"^[A-Za-z_]*?(\d+)$", str(col))
        return f"Sensor {m.group(1)}" if m else str(col)

    COL_REAL = "tab:blue"
    COL_SYN  = "tab:orange"

    for col in num_cols:
        x = real_filt[col].dropna()
        y = synthetic_filt[col].dropna()
        if x.empty or y.empty:
            continue

        xmin = min(x.min(), y.min())
        xmax = max(x.max(), y.max())
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
            continue

        bins = np.linspace(xmin, xmax, 50)

        fig, ax = plt.subplots(figsize=(7.4, 4.8))

        # Real
        ax.hist(
            x, bins=bins, alpha=0.45, label="Real",
            color=COL_REAL, edgecolor=COL_REAL, linewidth=0.6
        )
        ax.hist(
            x, bins=bins, histtype="step", linewidth=1.6,
            color=COL_REAL, label="_real_outline"
        )

        # Synthetic
        ax.hist(
            y, bins=bins, alpha=0.45, label="Synthetic",
            color=COL_SYN, edgecolor=COL_SYN, linewidth=0.6
        )
        ax.hist(
            y, bins=bins, histtype="step", linewidth=1.6,
            color=COL_SYN, label="_synthetic_outline"
        )

        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        title_suffix = filter_desc if filter_desc else "unfiltered"
        ax.set_title(f"{pretty_sensor_title(col)}  |  Filter: {title_suffix}")
        ax.grid(True, ls="--", alpha=.3)
        ax.legend()

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"dist_filtered_{col}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

    print("Saved filtered distribution plots (colored outlines, no tables).")

    # ============================================================
    # 3. Correlation comparison (numeric; sorted axes, FILTERED)
    # ============================================================
    print("\n=== Correlation Comparison (Filtered) ===")
    real_corr = real_filt[num_cols].corr()
    synthetic_corr = synthetic_filt[num_cols].corr()
    corr_diff = (real_corr - synthetic_corr).abs()

    sorted_idx = sorted(corr_diff.index, key=sort_key)
    sorted_cols = sorted(corr_diff.columns, key=sort_key)
    corr_diff = corr_diff.reindex(index=sorted_idx, columns=sorted_cols)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_diff, cmap="coolwarm", center=0)
    plt.title(f"Correlation Difference (|real - synthetic|)  |  Filter: {filter_desc}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_diff_filtered.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved filtered correlation difference heatmap as correlation_diff_filtered.png")

    # ============================================================
    # 4. Optional: RUL trends (if present), FILTERED
    # ============================================================
    if {"RUL", "Time_in_cycles"}.issubset(real_filt.columns) and {"RUL", "Time_in_cycles"}.issubset(synthetic_filt.columns):
        xr = real_filt[["Time_in_cycles", "RUL"]].dropna()
        xs = synthetic_filt[["Time_in_cycles", "RUL"]].dropna()
        if not xr.empty and not xs.empty:
            plt.figure(figsize=(6, 4))
            plt.scatter(xr["Time_in_cycles"], xr["RUL"], alpha=0.4, label="Real", s=10)
            plt.scatter(xs["Time_in_cycles"], xs["RUL"], alpha=0.4, label="Synthetic", s=10)
            plt.xlabel("Time in Cycles")
            plt.ylabel("Remaining Useful Life (RUL)")
            plt.title(f"RUL Trends (Filtered: {filter_desc})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "RUL_trends_filtered.png"), dpi=150, bbox_inches="tight")
            plt.close()
            print("Saved filtered RUL trend comparison as RUL_trends_filtered.png")

    # ============================================================
    # 5. Statistical tests (KS / Chi-square), FILTERED
    # ============================================================
    print("\n=== Statistical Tests (KS / Chi-square, Filtered) ===")
    stats_results = []

    shared_cols = [c for c in real_filt.columns if c in synthetic_filt.columns]

    for col in shared_cols:
        rcol = real_filt[col]
        scol = synthetic_filt[col]

        if pd.api.types.is_numeric_dtype(rcol) and pd.api.types.is_numeric_dtype(scol):
            rx = rcol.dropna()
            sx = scol.dropna()
            if rx.empty or sx.empty:
                continue
            stat, p = ks_2samp(rx, sx)
            result = "Different" if p < 0.05 else "Similar"
            stats_results.append((col, "KS", stat, p, result))
        else:
            r_counts = rcol.astype("category").value_counts()
            s_counts = scol.astype("category").value_counts()
            table = pd.concat([r_counts, s_counts], axis=1).fillna(0)
            table.columns = ["real", "synthetic"]
            if table.shape[0] < 2:
                continue
            chi2_stat, p_val, dof, expected = chi2_contingency(table.values)
            result = "Different" if p_val < 0.05 else "Similar"
            stats_results.append((col, "Chi-square", chi2_stat, p_val, result))

    stats_df = pd.DataFrame(stats_results, columns=["Column", "Test", "Statistic", "p_value", "Result"])
    print(stats_df)
    stats_df.to_csv(os.path.join(output_dir, "statistical_tests_filtered.csv"), index=False)
    print("Saved filtered statistical test results as statistical_tests_filtered.csv")

    print("\nAll condition-filtered checks complete. Files saved in:", output_dir)

if __name__ == "__main__":
    print("Run this module from validation.py")