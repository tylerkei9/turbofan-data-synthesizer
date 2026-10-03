import argparse
from Descriptive_Stats_Validation import run_descriptive_stats
from Covariance_and_Graphs_Validation import run_covariance_and_graphs
from Conditional_Validation_Script import run_conditional_validation

def main():

    parser = argparse.ArgumentParser(
        description="Synthetic Data Validation Pipeline"
    )

    parser.add_argument("--real", required=True,
                        help="Path to real dataset")

    parser.add_argument("--synthetic", required=True,
                        help="Path to synthetic dataset")

    parser.add_argument("--out", default="validation_outputs",
                        help="Output directory")

    args = parser.parse_args()

    run_descriptive_stats(args.real, args.synthetic, args.out + "/descriptive")
    run_covariance_and_graphs(args.real, args.synthetic, args.out + "/unconditional")
    run_conditional_validation(args.real, args.synthetic, args.out + "/conditional")

    print("ALL VALIDATION COMPLETE")

if __name__ == "__main__":
    print("Run this module from validation.py")
    main()
