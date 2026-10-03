import pandas as pd
import os 

def run_descriptive_stats(real_path, synthetic_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    real = pd.read_csv(real_path)
    synthetic = pd.read_csv(synthetic_path)

    # Quick compare
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

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "descriptive_stats.csv")
    compare.to_csv(out_file)

    print(f"[OK] Descriptive statistics saved to {out_file}")
