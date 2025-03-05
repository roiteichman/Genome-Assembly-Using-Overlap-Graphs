import os
import pandas as pd
import re

def create_paths(paths):
    """
    Create directories for saving results and plots.

    Parameters:
        paths (list): List of tuples (path, experiment_name) to create directories for each experiment.

    Returns:
        list: List of paths to save results and plots.
    """
    new_paths = []
    for i, (path, experiment_name) in enumerate(paths):
        new_path = os.path.join(path, experiment_name)
        directory = os.path.dirname(new_path)
        try:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory {directory}: {e}")
            return
        new_paths.insert(i, new_path)

    return new_paths


def save_results(results, experiment_name, path="results"):
    """
    Save experiment results to a CSV file.

    Parameters:
        results (list): List of result dictionaries.
        experiment_name (str): Name of the experiment.
        path (str): Path to save the results.

    Returns:
        None
    """
    df = pd.DataFrame(results)

    # Save full results
    df.to_csv(f"{path}/results.csv", index=False)

    # Save only the averaged results for easier analysis
    df_filtered = df[
        [col for col in df.columns if
         "avg" in col or col in ["num_reads", "read_length", "error_prob", "expected_coverage"]]
    ]
    df_filtered.to_csv(f"{path}/summary.csv", index=False)

    print(f"Results saved to {path}/results.csv")
    print(f"Summary results saved to {path}/summary.csv")


def load_coverage_results_from_csv(path_to_csvs, name_pattern):
    """
    Load experimental results from CSV files in the specified directory.

    Parameters:
        path_to_csvs (str): Path to the directory containing CSV result files.
        name_pattern (str): The base name pattern to search for in filenames.

    Returns:
        dict: A dictionary with coverage levels as keys and lists of results as values.
    """
    all_coverage_results = {}

    # Iterate through all CSV files in the directory
    for filename in os.listdir(path_to_csvs):
        if filename.endswith('.csv') and name_pattern in filename:
            # Extract coverage level using regex
            match = re.search(rf'{re.escape(name_pattern)}_(\d+)', filename)
            if not match:
                print(f"Skipping file {filename} - could not extract coverage")
                continue

            coverage = float(match.group(1))
            full_path = os.path.join(path_to_csvs, filename)

            # Read the CSV file
            df = pd.read_csv(full_path)
            if df.empty:
                print(f"Warning: {filename} is empty! Skipping.")
                continue

            # Convert DataFrame to list of dictionaries
            results = df.to_dict('records')

            # Store results for this coverage level
            all_coverage_results[coverage] = results

    return all_coverage_results