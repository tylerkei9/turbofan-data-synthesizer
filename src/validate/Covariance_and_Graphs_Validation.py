import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.decomposition import PCA
import os
 

def run_covariance_and_graphs(real_path, synthetic_path, output_dir):


    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print("✅ Saving all outputs to:", output_dir)
    
    # === Load datasets ===
    real = pd.read_csv(real_path)
    synthetic = pd.read_csv(synthetic_path)
    # Drop the last column 'Generated_Cluster' if it exists
    if 'Generated_Cluster' in synthetic.columns:
        synthetic = synthetic.drop(columns=['Generated_Cluster'])
    
    # Downsample the larger dataset so both have the same number of rows
    n = min(len(real), len(synthetic))
    real = real.sample(n=n, random_state=42).reset_index(drop=True)
    synthetic = synthetic.sample(n=n, random_state=42).reset_index(drop=True)   
    
    print("\n✅ Loaded datasets:")
    print(f"Real shape: {real.shape}")
    print(f"Synthetic shape: {synthetic.shape}")
    
    # === 1. Summary statistics comparison ===
    print("\n=== Summary Statistics Comparison ===")
    compare = pd.DataFrame({
        "real_mean": real.mean(numeric_only=True),
        "synthetic_mean": synthetic.mean(numeric_only=True),
        "real_std": real.std(numeric_only=True),
        "synthetic_std": synthetic.std(numeric_only=True),
        "real_min": real.min(numeric_only=True),
        "synthetic_min": synthetic.min(numeric_only=True),
        "real_max": real.max(numeric_only=True),
        "synthetic_max": synthetic.max(numeric_only=True),
    })
    print(compare)
    compare.to_csv(os.path.join(output_dir, "summary_comparison.csv"))
    print("📂 Saved summary comparison as summary_comparison.csv")
    
    # === 2. Distribution comparison ===
    print("\n=== Plotting distribution comparisons ===")
    for col in real.columns:
        if col in synthetic.columns and pd.api.types.is_numeric_dtype(real[col]):
            plt.figure(figsize=(6, 4))
            plt.hist(real[col], bins=50, alpha=0.5, label="Real")
            plt.hist(synthetic[col], bins=50, alpha=0.5, label="Synthetic")
            plt.title(f"Distribution Comparison for Sensor '{col}'", fontsize=12, fontweight="bold")
            plt.xlabel(f"Sensor {col} Reading", fontsize=10)
            plt.ylabel("Frequency", fontsize=10)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"dist_{col}.png"))
            plt.close()
    print("📂 Saved distribution plots for all numeric columns.")
    
    # === 3. Correlation comparison ===
    print("\n=== Correlation Comparison ===")
    real_corr = real.corr()
    synthetic_corr = synthetic.corr()
    corr_diff = (real_corr - synthetic_corr).abs()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_diff, cmap="coolwarm", center=0)
    plt.title("Correlation Difference (|real - synthetic|)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_diff.png"))
    plt.close()
    print("📂 Saved correlation difference heatmap as correlation_diff.png")
    
    # === 4. PCA Comparison ===
    print("\n=== PCA Comparison ===")
    # Get numeric columns that exist in both datasets
    numeric_cols = [col for col in real.columns 
                    if col in synthetic.columns 
                    and pd.api.types.is_numeric_dtype(real[col])]

    if len(numeric_cols) > 1:
        # Prepare data
        real_numeric = real[numeric_cols].fillna(real[numeric_cols].median())
        synthetic_numeric = synthetic[numeric_cols].fillna(synthetic[numeric_cols].median())
        
        # Fit PCA on real data
        pca = PCA(n_components=2)
        real_pca = pca.fit_transform(real_numeric)
        
        # Transform synthetic data using the same PCA
        synthetic_pca = pca.transform(synthetic_numeric)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Overlay comparison
        ax = axes[0]
        ax.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.5, s=20, label="Real", color='steelblue', edgecolors='darkblue', linewidth=0.3)
        ax.scatter(synthetic_pca[:, 0], synthetic_pca[:, 1], alpha=0.5, s=20, label="Synthetic", color='coral', edgecolors='darkred', linewidth=0.3)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=11, fontweight='bold')
        ax.set_title('PCA Comparison: Real vs Synthetic (Overlay)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, markerscale=2)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Side-by-side comparison
        ax = axes[1]
        ax.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.6, s=20, label="Real", color='steelblue', edgecolors='darkblue', linewidth=0.3)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=11, fontweight='bold')
        ax.set_title('Real Data in PCA Space', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Principal Component Analysis Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "pca_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("📂 Saved PCA comparison as pca_comparison.png")
        
        # Print variance explained
        print(f"  Variance explained by PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
        print(f"  Variance explained by PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")
        print(f"  Total variance explained: {(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])*100:.2f}%")
    else:
        print("⚠️ Not enough numeric columns for PCA analysis")
    
    # === 5. Optional: RUL trends ===
    if "RUL" in real.columns and "RUL" in synthetic.columns and \
    "Time_in_cycles" in real.columns and "Time_in_cycles" in synthetic.columns:
        print("\n=== Plotting RUL trends ===")
        plt.figure(figsize=(6, 4))
        plt.scatter(real["Time_in_cycles"], real["RUL"], alpha=0.5, label="Real")
        plt.scatter(synthetic["Time_in_cycles"], synthetic["RUL"], alpha=0.5, label="Synthetic")
        plt.title("RUL vs Time in Cycles", fontsize=12, fontweight="bold")
        plt.xlabel("Time in Cycles", fontsize=10)
        plt.ylabel("Remaining Useful Life (RUL)", fontsize=10)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "RUL_trends.png"))
        plt.close()
        print("📂 Saved RUL trend comparison as RUL_trends.png")
    
    # === 6. Statistical tests ===
    print("\n=== Statistical Tests (KS / Chi-square) ===")
    stats_results = []
    
    for col in real.columns:
        if col not in synthetic.columns:
            continue
    
        if pd.api.types.is_numeric_dtype(real[col]):
            stat, p = ks_2samp(real[col].dropna(), synthetic[col].dropna())
            result = "Different" if p < 0.05 else "Similar"
            stats_results.append((col, "KS", stat, p, result))
        else:
            real_counts = real[col].value_counts()
            synth_counts = synthetic[col].value_counts()
            combined = pd.concat([real_counts, synth_counts], axis=1).fillna(0)
            chi2, p, _, _ = chi2_contingency(combined)
            result = "Different" if p < 0.05 else "Similar"
            stats_results.append((col, "Chi-square", chi2, p, result))
    
    stats_df = pd.DataFrame(stats_results, columns=["Column", "Test", "Statistic", "p_value", "Result"])
    print(stats_df)
    stats_df.to_csv(os.path.join(output_dir, "statistical_tests.csv"), index=False)
    print("📂 Saved statistical test results as statistical_tests.csv")
    
    print("\n✅ All checks complete. Files saved in:", output_dir)

    # --------------------------------------------------
# Allows standalone testing
# --------------------------------------------------
if __name__ == "__main__":
    print("Run this module from validation.py")
