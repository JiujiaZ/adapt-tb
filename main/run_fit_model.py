import argparse
import os
from src.fit_model import get_model_parameter
def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Run model fitting with specified r and d values.")

    # Optional arguments for passing r and d values directly (if not on a cluster)
    parser.add_argument("--r", type=int, help="Ratio of s+ to s- (must be 1, 2, 3, 4, or 5)")
    parser.add_argument("--d", type=float, help="d value (must be 0.33 or 0.43)")

    # Parse the arguments
    args = parser.parse_args()

    # Default values for r and d arrays
    r_values = [1, 2, 3, 4, 5]
    d_values = [0.33, 0.43, 0.53, 0.63, 0.73]

    # Check if the script is running in an SGE environment (i.e., SGE_TASK_ID)
    if "SGE_TASK_ID" in os.environ:
        task_id = int(os.environ["SGE_TASK_ID"])

        # Use the task ID to index into r and d arrays
        r = r_values[(task_id - 1) % len(r_values)]
        d = d_values[(task_id - 1) // len(r_values)]
        print(f"Running model on cluster with r={r}, d={d} (SGE_TASK_ID={task_id})")

    else:
        # Use provided arguments (fallback for local runs)
        if args.r is None or args.d is None:
            raise ValueError("You must provide both r and d values when not running in a cluster environment.")

        r = args.r
        d = args.d

        # Validate r and d
        if r not in r_values:
            raise ValueError(f"r must be one of {r_values}.")

        if d not in d_values:
            raise ValueError(f"d must be one of {d_values}.")

        print(f"Running model locally with r={r}, d={d}")

    # Run the model fitting with the specified r and d
    get_model_parameter(r=r, d=d)


if __name__ == "__main__":
    main()
