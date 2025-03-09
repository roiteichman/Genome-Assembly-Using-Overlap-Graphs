import os
import pandas as pd
import re
import ast


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


def load_results_from_csv(file_path):
    """
    Load experiment results from a CSV file into a list of dictionaries.

    Parameters:
        file_path (str): Path to the CSV result file.

    Returns:
        list: A list of result dictionaries.
    """
    try:
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"Warning: {file_path} is empty.")
        return
    except Exception as e:
        print(f"An error occurred while loading {file_path}: {e}")
        return


def load_coverage_results_from_csv(base_path, name_pattern):
    """
    Load experimental results from 'results.csv' files in the specified directory structure.

    Parameters:
        base_path (str): The base path "results\experiment_const_coverage".
        name_pattern (str): The pattern for the C_{number} directories (e.g., "C_").

    Returns:
        dict: A dictionary with coverage levels as keys and lists of results as values.
    """
    all_coverage_results = {}

    for dir_name in os.listdir(base_path):
        if dir_name.startswith(name_pattern):
            # Extract coverage level from directory name
            try:
                coverage = float(dir_name.split('_')[1])
            except (IndexError, ValueError):
                print(f"Skipping directory {dir_name} - could not extract coverage")
                continue

            # Construct full path to results.csv
            results_path = os.path.join(base_path, dir_name, "results.csv")

            # Load results
            results = load_results_from_csv(results_path)

            all_coverage_results[coverage] = results

    return all_coverage_results


def load_and_clean_results(folder_path):
    """
    Loads results from a CSV file, cleans the data, and returns it as a list of dictionaries.

    Parameters:
        folder_path (str): Path to the folder containing the results.csv file.

    Returns:
        list: List of cleaned result dictionaries suitable for plotting.
    """
    try:
        # Construct the full file path
        file_path = os.path.join(folder_path, "results.csv")

        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Identify columns with "raw" in their suffix
        raw_columns = [col for col in df.columns if col.endswith("raw")]

        # Convert "raw" columns from string representation of lists to actual lists and handle numpy types
        for col in raw_columns:
            def convert_list(x):
                try:
                    lst = ast.literal_eval(x)
                    if not isinstance(lst, list):
                        return lst  # return non list values as is.

                    if col in ["num_reads raw", "read_length raw", "Number of Contigs raw", "N50 raw"]:
                        return [int(val) for val in lst]
                    elif col in ["error_prob raw", "Mismatch Rate Aligned Regions raw",
                                 "Mismatch Rate Genome Level raw", "expected_coverage raw", "Genome Coverage raw"]:
                        return [float(val) for val in lst]
                    else:
                        return lst  # return other lists as is.
                except (ValueError, SyntaxError) as e:
                    print(f"Error converting column '{col}': {e} - value: {x}")
                    return []  # return empty list on error.

            df[col] = df[col].apply(convert_list)

        # Convert DataFrame to a list of dictionaries
        cleaned_results = df.to_dict('records')

        return cleaned_results

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"Warning: {file_path} is empty.")
        return
    except Exception as e:
        print(f"An error occurred while loading or cleaning {file_path}: {e}")
        return
