import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pandas as pd
import os
import time
from functools import partial
from multiprocessing import Pool
from generateErrorFreeReads import read_genome_from_fasta, generate_error_free_reads
from generateErrorProneReads import generate_error_prone_reads
from overlapGraphs import assemble_contigs_using_overlap_graphs
from performanceMeasures import calculate_essential_performance_measures
from testAssembly import test_assembly


def run_experiments(file_path="sequence.fasta"):
    """
    Main function to run all experimentation scenarios.

    Parameters:
        file_path (str): Path to the FASTA file containing the reference genome.

    Returns:
        None (Saves results to files and generates plots)
    """
    # Create directories for results if they don't exist
    path_to_save_csvs = "results"
    path_to_save_plots = "plots"
    os.makedirs(path_to_save_csvs, exist_ok=True)
    os.makedirs(path_to_save_plots, exist_ok=True)
    paths = [path_to_save_csvs, path_to_save_plots]

    # Read the reference genome from the FASTA file
    #genome = read_genome_from_fasta(file_path) #TODO: uncomment this line
    toy_genome = "ATGCGTACGTTAGCATGCGTACGTTAGC"  #TODO: remove this line

    genome_length = len(toy_genome)  #TODO: update this value

    # Define common parameters
    min_error_prob = 0.001
    error_probs = [0.001, 0.01, 0.05, 0.1]

    small_l = 5  #50 #TODO: update this value
    large_l = 25  #150 #TODO: update this value

    small_n = 25 #100 #TODO: update this value
    large_n = 1000  #TODO: update this value

    # calculate C for small n (50*100)/5386 = 0.93
    c_smaller_than_1_small_n = ((small_l * small_n) / len(toy_genome))  # TODO: update to genome length

    small_coverage_targets = [c_smaller_than_1_small_n, *list(range(2, 4))]  # TODO: update this value
    large_coverage_targets = [*(list(range(10, 55, 15)))]  # TODO: update this value

    # Calculate N values for different coverage depths
    small_n_values_small_l = [int(np.ceil(coverage * genome_length / small_l)) for coverage in small_coverage_targets]
    large_n_values_small_l = [int(np.ceil(coverage * genome_length / small_l)) for coverage in large_coverage_targets]

    # Calculate l values for different coverage depths
    small_l_values_small_n = sorted([int(np.ceil(coverage * genome_length / small_n))
                                     for coverage in small_coverage_targets
                                     if int(np.ceil(coverage * genome_length / small_n)) < large_l])
    large_l_values_small_n = sorted([int(np.ceil(coverage * genome_length / small_n))
                                     for coverage in large_coverage_targets
                                     if int(np.ceil(coverage * genome_length / small_n)) < large_l])

    # Calculate expected coverage
    expected_coverage_small_n = [n * small_l / genome_length for n in small_n_values_small_l]
    expected_coverage_large_n = [n * small_l / genome_length for n in large_n_values_small_l]

    expected_coverage_small_l = [small_n * l / genome_length for l in small_l_values_small_n]
    expected_coverage_large_l = [small_n * l / genome_length for l in large_l_values_small_n]


    # Verify N - small C
    """experiment_varying_value(toy_genome, small_n_values_small_l, [small_l], [min_error_prob],
                             expected_coverage_small_n, "experiment1_varying_N_small_range", paths)"""

    # Verify N - large C
    """experiment_varying_value(toy_genome, large_n_values_small_l, [small_l], [min_error_prob],
                             expected_coverage_small_n, "experiment1_varying_N_large_range", paths)"""


    # Verify l - small C
    """experiment_varying_value(toy_genome, [small_n], small_l_values_small_n, [min_error_prob],
                             expected_coverage_small_l, "experiment2_varying_l_small_range", paths)"""


    # Verify l - large C
    """experiment_varying_value(toy_genome, [small_n], large_l_values_small_n, [min_error_prob],
                             expected_coverage_large_l, "experiment2_varying_l_large_range", paths)"""


    # Verify p - small C
    experiment_varying_value(toy_genome, small_n_values_small_l, [small_l], error_probs,
                             expected_coverage_small_n, "experiment3_varying_p_N_small_range", paths)

    """experiment_varying_value(toy_genome, [small_n], small_l_values_small_n, error_probs,
                             expected_coverage_small_l, "experiment3_varying_p_l_small_range", paths)"""

    # Verify p - large C - TODO: update this value
    """experiment_varying_value(toy_genome, large_n_values_small_l, [small_l], error_probs,
                             expected_coverage_small_n, "experiment3_varying_p_N_large_range", paths)"""
    # TODO: update this value
    """experiment_varying_value(toy_genome, [small_n], large_l_values_small_n, error_probs,
                                expected_coverage_large_l, "experiment3_varying_p_l_large_range", paths)"""


    # TODO - more...
    """
    const_coverage_target = 10
    n_values_const_coverage = [10, 28, 56, 70, 140] # TODO - update this value

    # Verify N and l - constant C
    experiment_const_coverage(toy_genome, const_coverage_target, error_probs, n_values_const_coverage,
                              x_axis_var="n", experiment_name="experiment4_const_coverage", paths=paths)
    """

    print("All experiments completed!")


def experiment_varying_value(reference_genome, n_values, l_values, p_values, expected_coverage,
                             experiment_name, paths, num_iterations=5):
    """
    Experiment: Vary variable values to achieve different coverage depths.

    Parameters:
        reference_genome (str): Reference genome sequence.
        n_values (list): List of N values to test.
        l_values (list): List of l values to test.
        p_values (list): List of p values to test.
        expected_coverage (list): List of expected coverage values corresponding to N values.
        experiment_name (str): Name of the experiment.
        paths (list): List of paths to save results and plots.
        num_iterations (int): Number of iterations to run for each parameter combination.

    Returns:
        None (Saves results to files and generates plots)
    """
    print(f"Running Experiment: {experiment_name}")

    print(f"reference_genome: {reference_genome}")
    print(f"n_values: {n_values}")
    print(f"l_values: {l_values}")
    print(f"p_values: {p_values}")
    print(f"expected_coverage: {expected_coverage}\n")

    # Create parameter combinations
    params = []  # list of dictionaries

    """
    variable = None
    labels = ["N (Number of Reads)", "l (Read Length)", "p (Error Probability)"]
    label = None
    x_keys = ["num_reads", "read_length", "error_prob"]
    x_key = None

    if len(p_values) > 1:
        label = labels[0] if len(n_values) > 1 else labels[1]
        x_key = x_keys[0] if len(n_values) > 1 else x_keys[1]
    elif len(n_values) > 1:
        label = labels[0]
        x_key = x_keys[0]
    elif len(l_values) > 1:
        label = labels[1]
        x_key = x_keys[1]
    """

    for i, p in enumerate(p_values):
        for j, n in enumerate(n_values):
            for k, l in enumerate(l_values):
                params.append({
                    'num_reads': n,
                    'read_length': l,
                    'error_prob': p,
                    'reference_genome': reference_genome,
                    'expected_coverage': expected_coverage[j] if len(n_values) > 1 else expected_coverage[k],
                    'experiment_name': experiment_name
                })

    # Create folders for the experiment
    path_to_save_csvs = f"{paths[0]}/{experiment_name}"
    os.makedirs(path_to_save_csvs, exist_ok=True)
    path_to_save_plots = f"{paths[1]}/{experiment_name}"
    os.makedirs(path_to_save_plots, exist_ok=True)

    # Run simulations
    results = run_simulations_num_iteration(params, num_iterations, path=path_to_save_plots)

    # Save results
    save_results(results, experiment_name, path=path_to_save_csvs)

    # Plot results
    #if len(p_values) > 1:
    if len(n_values) > 1:
        plot_experiment_results_by_p_values(results, x_key="num_reads",
                                            coverage_key="expected_coverage", path=path_to_save_plots,
                                            num_iterations=num_iterations)
    elif len(l_values) > 1:
        plot_experiment_results_by_p_values(results, x_key="read_length",
                                            coverage_key="expected_coverage", path=path_to_save_plots,
                                            num_iterations=num_iterations)
    #else:
        #plot_experiment_results(results, label, experiment_name, x_key=x_key, coverage_key="expected_coverage",
        #                        path=path_to_save_plots)


def run_simulations_num_iteration(params_list, num_iterations=5, path="plots"):
    """
    Run simulations with the given parameter combinations, repeating each simulation
    num_iterations times and returning the average results with standard deviation.

    Parameters:
        params_list (list): List of parameter dictionaries.
        num_iterations (int): Number of iterations to run for each parameter combination.
        path (str): Path to save the plots.

    Returns:
        list: List of result dictionaries
    """
    results = []

    for params in params_list:
        print(
            f"Running {params['experiment_name']} simulation with N={params['num_reads']}, l={params['read_length']}, "
            f"p={params['error_prob']}, expected coverage={params['expected_coverage']:.2f}x"
        )
        # Create folders for the experiment
        experiment_folder = f"{path}/test_assembly/N={params['num_reads']}_l={params['read_length']}_p={params['error_prob']}"
        os.makedirs(experiment_folder, exist_ok=True)

        all_iteration_results_error_free = []
        all_iteration_results_error_prone = []

        for i in range(num_iterations):
            results_ef, results_ep = run_simulations([params], num_iteration=i + 1, path=experiment_folder)  # first iteration is 1
            all_iteration_results_error_free.append(results_ef[0])  # Get first dict from the list
            all_iteration_results_error_prone.append(results_ep[0])  # Get first dict from the list

        # Extract **only numeric keys** for averaging
        numeric_keys = [
            key for key in all_iteration_results_error_free[0].keys()
            if isinstance(all_iteration_results_error_free[0][key], (int, float, np.number))
        ]
        # Calculate averages and standard deviations for all performance measures
        avg_results_error_free = {
            key: np.mean([r[key] for r in all_iteration_results_error_free]) for key in numeric_keys
        }
        std_results_error_free = {
            key: np.std([r[key] for r in all_iteration_results_error_free]) for key in numeric_keys
        }

        avg_results_error_prone = {
            key: np.mean([r[key] for r in all_iteration_results_error_prone]) for key in numeric_keys
        }
        std_results_error_prone = {
            key: np.std([r[key] for r in all_iteration_results_error_prone]) for key in numeric_keys
        }

        # Rename keys for clarity
        formatted_results = {
            **params,
            **{f"{key} (EF) avg": avg_results_error_free[key] for key in avg_results_error_free},
            **{f"{key} (EF) std": std_results_error_free[key] for key in std_results_error_free},
            **{f"{key} (EP) avg": avg_results_error_prone[key] for key in avg_results_error_prone},
            **{f"{key} (EP) std": std_results_error_prone[key] for key in std_results_error_prone},
            **{f"{key} (EP) raw": [r[key] for r in all_iteration_results_error_prone] for key in numeric_keys},
            **{f"{key} (EF) raw": [r[key] for r in all_iteration_results_error_free] for key in numeric_keys}
        }

        results.append(formatted_results)

    return results


def run_simulations(params_list, num_iteration, path="plots"):
    """
    Run simulations with the given parameter combinations.

    Parameters:
        params_list (list): List of parameter dictionaries.
        num_iteration (int): The number of the specific iteration.
        path (str): Path to save the plots.

    Returns:
        tuple: Tuple of lists of result dictionaries for error-free and error-prone assemblies.
    """
    results_error_free = []
    results_error_prone = []

    for params in params_list:
        measures = test_assembly(params['reference_genome'], params['read_length'], params['num_reads'],
                                 params['error_prob'], params['experiment_name'], num_iteration, path)

        # Add parameters to results

        result_error_free = {**params, **measures["error_free"]}
        result_error_prone = {**params, **measures["error_prone"]}
        results_error_free.append(result_error_free)
        results_error_prone.append(result_error_prone)

    return results_error_free, results_error_prone


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
    df.to_csv(f"{path}/{experiment_name}_results.csv", index=False)

    # Save only the averaged results for easier analysis
    df_filtered = df[
        [col for col in df.columns if
         "avg" in col or col in ["num_reads", "read_length", "error_prob", "expected_coverage"]]
    ]
    df_filtered.to_csv(f"{path}/{experiment_name}_summary.csv", index=False)

    print(f"Results saved to {path}/{experiment_name}_results.csv")
    print(f"Summary results saved to {path}/{experiment_name}_summary.csv")


def plot_experiment_results(results, x_label, experiment_name, x_key="num_reads", coverage_key="expected_coverage",
                            path="plots", log_scale=False):
    """
    Plot experiment results.

    Parameters:
        results (list): List of result dictionaries.
        x_label (str): Label for x-axis.
        experiment_name (str): Name of the experiment.
        x_key (str): Dictionary key for x-axis values.
        coverage_key (str): Dictionary key for coverage values or None.
        path (str): Path to save the plots.
        log_scale (bool): Whether to use logarithmic scale for x-axis.

    Returns:
        None
    """
    # Create dataframe from results
    df = pd.DataFrame(results)

    # Metrics to plot
    metrics = ["Number of Contigs", "Genome Coverage", "N50", "Mismatch Rate"]
    metric_labels = ["Number of Contigs", "Genome Coverage (%)", "N50", "Mismatch Rate"]

    # Loop for both EF (error-free) and EP (error-prone) results
    for error_type in ["EF", "EP"]:

        error_type_str = "Error-Free" if error_type == "EF" else "Error-Prone"

        # Create path for the experiment
        full_path = f"{path}/{error_type_str}"
        os.makedirs(full_path, exist_ok=True)

        for include_raw in [True, False]:  # First plot without raw, then with
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()

            # Plot each metric
            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                ax = axes[i]

                # Sort by x_key
                df_sorted = df.sort_values(by=x_key)
                x_values = df_sorted[x_key].values
                # Extract metric values
                metric_avg = df_sorted[f"{metric} ({error_type}) avg"].values
                metric_std = df_sorted[f"{metric} ({error_type}) std"].values

                if coverage_key:
                    # Create labels with coverage information
                    coverage_values = df_sorted[coverage_key].values
                    x_labels = [f"{x}\n(C={c:.1f}x)" for x, c in zip(x_values, coverage_values)]
                else:
                    x_labels = [str(x) for x in x_values]

                # Plot data
                ax.errorbar(x_values, metric_avg, yerr=metric_std, fmt='o-', label=f"{error_type_str}", capsize=5)

                # Overlay raw data points if requested
                if include_raw:
                    raw_data = df_sorted[f"{metric} ({error_type}) raw"].values
                    for j, raw_vals in enumerate(raw_data):
                        ax.scatter([x_values[j]] * len(raw_vals), raw_vals, alpha=0.7, color='gray', s=12,
                                   label="Raw Data" if j == 0 else "")

                # Set x-axis labels
                ax.set_xticks(x_values)
                ax.set_xticklabels(x_labels, rotation=45)

                # Set y-axis to log scale if needed for better visualization
                if metric == "num_contigs" or metric == "n50":
                    ax.set_yscale('log')

                # Set x-axis to log scale if requested
                if log_scale:
                    ax.set_xscale('log')

                # Trend line
                trend = np.polyfit(x_values, metric_avg, len(x_values) - 2)  # one degree below perfect fit
                trend_y = np.polyval(trend, x_values)  # Compute trendline values

                # Plot trend line
                ax.plot(x_values, trend_y, 'k--', label=f"Trend Line")

                # Add labels and title
                ax.set_xlabel(x_label)
                ax.set_ylabel(label)
                ax.set_title(f"{label} vs. {x_label}")
                ax.grid(True, alpha=0.3)

                # Add legend
                ax.legend()

            # Adjust layout
            plt.tight_layout()
            suffix = "_with_raw" if include_raw else ""
            # Save plot
            plt.savefig(f"{full_path}/plot{suffix}.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Plots saved to {full_path}/plot{suffix}.png")


def plot_experiment_results_by_p_values(results, x_key="num_reads", coverage_key="expected_coverage",
                                        path="plots", log_scale=False, num_iterations=5):
    """
    Plot experiment results with x_key values as x-axis and different error probabilities as separate series.
    Now includes individual plots for each p value with trend lines, and an average trend line for combined plots.

    Parameters:
        results (list): List of result dictionaries.
        x_key (str): Dictionary key for x-axis values (typically "num_reads" or "read_length").
        coverage_key (str): Dictionary key for coverage values or None.
        path (str): Path to save the plots.
        log_scale (bool): Whether to use logarithmic scale for x-axis.
        num_iterations (int): Number of iterations to run for each parameter combination.

    Returns:
        None
    """
    # Create dataframe from results
    df = pd.DataFrame(results)

    # Get unique error probabilities and x values
    p_values = sorted(df['error_prob'].unique())
    x_values = sorted(df[x_key].unique())

    # Metrics to plot
    metrics = ["Number of Contigs", "Genome Coverage", "N50", "Mismatch Rate"]
    metric_labels = ["Number of Contigs", "Genome Coverage (%)", "N50", "Mismatch Rate"]

    # Colors for different p values
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    light_colors = ['#a6cee3', '#fdbf6f', '#b2df8a', '#fb9a99', '#cab2d6', '#d2b48c']

    # Set x-axis label based on x_key
    if x_key == "num_reads":
        x_axis_label = "N (Number of Reads)"
    elif x_key == "read_length":
        x_axis_label = "l (Read Length)"
    else:
        x_axis_label = x_key

    # Loop for error-prone results
    error_type_str = "Error-Prone"
    for error_type in ["EP"]:
        # Create path for the experiment
        full_path = f"{path}/{error_type_str}"
        os.makedirs(full_path, exist_ok=True)

        # 1. Plot combined graph with all p values
        for include_raw in [False, True]:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()

            # Plot each metric
            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                ax = axes[i]

                # Data for average trend line
                all_x = []
                all_y = []

                # For each error probability, plot a separate line
                for p_idx, p in enumerate(p_values):
                    # Filter data for this p value
                    df_p = df[df['error_prob'] == p].sort_values(by=x_key)

                    if df_p.empty:
                        continue

                    x_values = df_p[x_key].values
                    metric_avg = df_p[f"{metric} ({error_type}) avg"].values
                    metric_std = df_p[f"{metric} ({error_type}) std"].values

                    # Collect data for overall trend line
                    all_x.extend(x_values)
                    all_y.extend(metric_avg)

                    # Plot data with error bars
                    ax.errorbar(x_values, metric_avg, yerr=metric_std, fmt='o-',
                                label=f"p={p}", color=colors[p_idx % len(colors)],
                                capsize=5, markersize=6)

                    # Overlay raw data points if requested
                    if include_raw:
                        raw_data = df_p[f"{metric} ({error_type}) raw"].values
                        for j, raw_vals in enumerate(raw_data):
                            ax.scatter([x_values[j]] * len(raw_vals), raw_vals,
                                       alpha=0.7, color=light_colors[p_idx % len(light_colors)], s=20,
                                       marker='o')

                # Add coverage information to x-axis if available
                if coverage_key:
                    # Get all unique x_value and coverage combinations
                    x_coverage_map = {}
                    for x in sorted(df[x_key].unique()):
                        # Find corresponding coverage for this x value (should be same for all p)
                        coverage = df[df[x_key] == x][coverage_key].iloc[0]
                        x_coverage_map[x] = coverage

                    # Create x-tick labels with coverage info
                    x_ticks = sorted(x_coverage_map.keys())
                    x_labels = [f"{x}\n(C={x_coverage_map[x]:.1f}x)" for x in x_ticks]

                    ax.set_xticks(x_ticks)
                    ax.set_xticklabels(x_labels, rotation=45)

                # Add average trend line if we have enough data points
                if len(all_x) > 2:
                    # Sort by x for proper line plotting
                    sorted_indices = np.argsort(all_x)
                    sorted_x = np.array(all_x)[sorted_indices]
                    sorted_y = np.array(all_y)[sorted_indices]

                    # Fit a polynomial (degree = min(3, len(unique x) - 1))
                    degree = min(3, len(set(all_x)) - 1)
                    if degree > 0:  # Need at least 2 points for a line
                        trend = np.polyfit(sorted_x, sorted_y, degree)
                        trend_y = np.polyval(trend, sorted_x)
                        ax.plot(sorted_x, trend_y, 'k--', linewidth=2, label="Average Trend")

                # Set x-axis to log scale if requested
                if log_scale:
                    ax.set_xscale('log')

                # Add labels and title
                ax.set_xlabel(x_axis_label)
                ax.set_ylabel(label)
                ax.set_title(f"{label} vs. {x_axis_label} for Different p ({num_iterations} iterations)")
                ax.grid(True, alpha=0.3)

                # Add legend
                ax.legend(title="Error Probability (p)", bbox_to_anchor=(1.05, 1), loc='upper left')

            # Adjust layout
            plt.tight_layout()
            suffix = "_with_raw" if include_raw else ""

            # Save plot
            output_path = f"{full_path}/plot_by_p_values{suffix}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Combined plot saved to {output_path}")

        # 2. Create individual plots for each p value
        for p_idx, p in enumerate(p_values):
            for include_raw in [False, True]:
                # Create figure with subplots
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.flatten()

                # Filter data for this p value
                df_p = df[df['error_prob'] == p].sort_values(by=x_key)

                if df_p.empty:
                    continue

                # Plot each metric
                for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                    ax = axes[i]

                    # Get data
                    x_values = df_p[x_key].values
                    metric_avg = df_p[f"{metric} ({error_type}) avg"].values
                    metric_std = df_p[f"{metric} ({error_type}) std"].values

                    # Plot data with error bars
                    ax.errorbar(x_values, metric_avg, yerr=metric_std, fmt='o-',
                                color=colors[p_idx % len(colors)],
                                capsize=5, markersize=6)

                    # Overlay raw data points if requested
                    if include_raw:
                        raw_data = df_p[f"{metric} ({error_type}) raw"].values
                        for j, raw_vals in enumerate(raw_data):
                            ax.scatter([x_values[j]] * len(raw_vals), raw_vals,
                                       alpha=0.7, color=light_colors[p_idx % len(light_colors)], s=20)

                    # Add coverage information to x-axis if available
                    if coverage_key:
                        # Create x-tick labels with coverage info
                        x_ticks = x_values
                        x_labels = [f"{x}\n(C={df_p[df_p[x_key] == x][coverage_key].iloc[0]:.1f}x)" for x in x_ticks]

                        ax.set_xticks(x_ticks)
                        ax.set_xticklabels(x_labels, rotation=45)

                    # Add trend line if we have enough points
                    if len(x_values) > 1:
                        # For better trends, sort the values by x
                        sorted_indices = np.argsort(x_values)
                        sorted_x = np.array(x_values)[sorted_indices]
                        sorted_y = np.array(metric_avg)[sorted_indices]

                        # Calculate degree for polynomial fit
                        degree = min(len(set(x_values)) - 1, 3)  # At most cubic, at least linear if possible
                        degree = max(degree, 1)  # Ensure at least linear if we have 2+ points

                        trend = np.polyfit(sorted_x, sorted_y, degree)
                        x_for_trend = np.linspace(min(sorted_x), max(sorted_x), 100)
                        trend_y = np.polyval(trend, x_for_trend)

                        ax.plot(x_for_trend, trend_y, 'k--', linewidth=2, label="Trend Line")

                    # Set x-axis to log scale if requested
                    if log_scale:
                        ax.set_xscale('log')

                    # Add labels and title
                    ax.set_xlabel(x_axis_label)
                    ax.set_ylabel(label)
                    ax.set_title(f"{label} vs. {x_axis_label} (p={p}, {num_iterations} iterations)")
                    ax.grid(True, alpha=0.3)
                    if len(x_values) > 1:  # Only add legend if we have trend line
                        ax.legend()

                # Adjust layout
                plt.tight_layout()
                suffix = "_with_raw" if include_raw else ""

                # Save plot
                output_file = f"{full_path}/plot_by_{x_key}_p_{p}{suffix}.png"
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Individual plot saved to {output_file}")


def experiment_const_coverage(reference_genome, coverage_target, error_probs,
                              n_values=None, l_values=None, x_axis_var="n",
                              experiment_name="const_coverage", paths=None, num_iterations=5):
    """
    Experiment: Vary N and l while keeping coverage constant.
    Either n_values or l_values should be provided, and the other will be calculated.

    Parameters:
        reference_genome (str): Reference genome sequence.
        coverage_target (float): Target coverage to maintain (e.g., 10X).
        error_probs (list): List of p values to test.
        n_values (list, optional): List of N values to test. If provided, l will be calculated.
        l_values (list, optional): List of l values to test. If provided, N will be calculated.
        x_axis_var (str): Variable to use on x-axis ('n' or 'l').
        experiment_name (str): Name of the experiment.
        paths (list): List of paths to save results and plots.
        num_iterations (int): Number of iterations to run for each parameter combination.

    Returns:
        None (Saves results to files and generates plots)
    """
    if paths is None:
        paths = ["results", "plots"]

    print(f"Running Experiment with Constant Coverage: {experiment_name}")

    genome_length = len(reference_genome)

    # Create parameter combinations
    params = []  # list of dictionaries

    # Calculate the dependent variable based on which one was provided
    if n_values is not None and l_values is None:
        # Calculate l values that maintain constant coverage for each n
        l_values = [int(np.ceil(coverage_target * genome_length / n)) for n in n_values]
        x_axis_var = "n"  # Force x_axis_var to match provided values
    elif l_values is not None and n_values is None:
        # Calculate n values that maintain constant coverage for each l
        n_values = [int(np.ceil(coverage_target * genome_length / l)) for l in l_values]
        x_axis_var = "l"  # Force x_axis_var to match provided values
    else:
        raise ValueError("Either n_values or l_values must be provided, but not both")

    # Create a list of expected coverage values (should all be approximately the same)
    expected_coverage = [n * l / genome_length for n, l in zip(n_values, l_values)]

    for i, p in enumerate(error_probs):
        for j, (n, l) in enumerate(zip(n_values, l_values)):
            params.append({
                'num_reads': n,
                'read_length': l,
                'error_prob': p,
                'reference_genome': reference_genome,
                'expected_coverage': expected_coverage[j],
                'experiment_name': experiment_name
            })

    # Create folders for the experiment
    csvs_path = f"{paths[0]}/{experiment_name}"
    os.makedirs(csvs_path, exist_ok=True)
    plots_path = f"{paths[1]}/{experiment_name}"
    os.makedirs(plots_path, exist_ok=True)

    # Run simulations
    results = run_simulations_num_iteration(params, num_iterations, path=plots_path)

    # Save results
    save_results(results, experiment_name, path=csvs_path)

    # Plot results
    plot_const_coverage_results(results, experiment_name,
                                coverage_target=coverage_target,
                                x_axis_var=x_axis_var,
                                path=plots_path)


def plot_const_coverage_results(results, coverage_target, x_axis_var="n", path="plots", num_iterations=5):
    """
    Plot experiment results with constant coverage but varying N and l.

    Parameters:
        results (list): List of result dictionaries.
        coverage_target (float): The target coverage value.
        x_axis_var (str): Variable to use on x-axis ('n' or 'l').
        path (str): Path to save the plots.
        num_iterations (int): Number of iterations to run for each parameter combination.

    Returns:
        None
    """
    # Create dataframe from results
    df = pd.DataFrame(results)

    # Get unique error probabilities
    p_values = sorted(df['error_prob'].unique())

    # Metrics to plot
    metrics = ["Number of Contigs", "Genome Coverage", "N50", "Mismatch Rate"]
    metric_labels = ["Number of Contigs", "Genome Coverage (%)", "N50", "Mismatch Rate"]

    # Colors for different p values
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    light_colors = ['#a6cee3', '#fdbf6f', '#b2df8a', '#fb9a99', '#cab2d6', '#d2b48c']

    # "Loop" for EP (error-prone) results
    error_type_str = "Error-Prone"
    for error_type in ["EP"]:
        # Create path for the experiment
        full_path = f"{path}/{error_type_str}/coverage_{coverage_target:.1f}x" #TODO: if not work remove the coverage_target from here and return down as claude did
        os.makedirs(full_path, exist_ok=True)

        # 1. Plot combined graph with all p values
        for include_raw in [False, True]:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()

            # Select x-axis variable and label according to x_axis_var
            if x_axis_var.lower() == 'n':
                x_key = 'num_reads'
                x_label = "N (Number of Reads)"
                y_key = 'read_length'
                y_label = "l (Read Length)"
            else:  # x_axis_var == 'l'
                x_key = 'read_length'
                x_label = "l (Read Length)"
                y_key = 'num_reads'
                y_label = "N (Number of Reads)"

            # Plot each metric
            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                ax = axes[i]

                # Data for average trend line
                all_x = []
                all_y = []

                # For each error probability, plot a separate line
                for p_idx, p in enumerate(p_values):
                    # Filter data for this p value
                    df_p = df[df['error_prob'] == p].sort_values(by=x_key)

                    if df_p.empty:
                        continue

                    # Get x values and corresponding y values
                    x_values = df_p[x_key].values
                    y_values = df_p[y_key].values
                    metric_avg = df_p[f"{metric} ({error_type}) avg"].values
                    metric_std = df_p[f"{metric} ({error_type}) std"].values

                    # Collect data for overall trend line
                    all_x.extend(x_values)
                    all_y.extend(metric_avg)

                    # Plot data with error bars
                    ax.errorbar(x_values, metric_avg, yerr=metric_std, fmt='o-',
                                label=f"p={p}", color=colors[p_idx % len(colors)],
                                capsize=5, markersize=6)

                    # Overlay raw data points if requested
                    if include_raw:
                        raw_data = df_p[f"{metric} ({error_type}) raw"].values
                        for j, raw_vals in enumerate(raw_data):
                            ax.scatter([x_values[j]] * len(raw_vals), raw_vals,
                                       alpha=0.7, color=light_colors[p_idx % len(light_colors)], s=20,
                                       marker='o')

                # Create x-tick labels with the other variable values
                x_ticks = sorted(df[x_key].unique())
                x_labels = [f"{x}\n({y_label[0]}={df[df[x_key] == x][y_key].iloc[0]})" for x in x_ticks]

                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_labels, rotation=45)

                # Add average trend line if we have enough data points
                if len(all_x) > 2:
                    # Sort by x for proper line plotting
                    sorted_indices = np.argsort(all_x)
                    sorted_x = np.array(all_x)[sorted_indices]
                    sorted_y = np.array(all_y)[sorted_indices]

                    # Fit a polynomial (degree = min(3, len(unique x) - 1))
                    degree = min(3, len(set(all_x)) - 1) #TODO: do you think we can take the len(set(all_x)) - 2 as the degree? to get almost the best fit
                    if degree > 0:  # Need at least 2 points for a line
                        trend = np.polyfit(sorted_x, sorted_y, degree)
                        trend_y = np.polyval(trend, sorted_x)
                        ax.plot(sorted_x, trend_y, 'k--', linewidth=2, label="Average Trend")

                # Add labels and title
                ax.set_xlabel(x_label)
                ax.set_ylabel(label)
                ax.set_title(f"{label} vs. {x_label} (C={coverage_target:.1f}x, {num_iterations} iterations)")
                ax.grid(True, alpha=0.3)

                # Add legend
                ax.legend(title="Error Probability", bbox_to_anchor=(1.05, 1), loc='upper left')

            # Adjust layout
            plt.tight_layout()
            suffix = "_with_raw" if include_raw else ""

            # Save plot
            output_file = f"{full_path}/ordered_by_{x_axis_var}{suffix}.png" #TODO: if not work return the coverage_target to here and remove from up as claude did
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Plots saved to {output_file}")

        # 2. Create individual plots for each p value
        for p_idx, p in enumerate(p_values):
            for include_raw in [False, True]:
                # Create figure with subplots
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.flatten()

                # Filter data for this p value
                df_p = df[df['error_prob'] == p].sort_values(by=x_key)

                if df_p.empty:
                    continue

                # Plot each metric
                for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                    ax = axes[i]

                    # Get data
                    x_values = df_p[x_key].values
                    y_values = df_p[y_key].values
                    metric_avg = df_p[f"{metric} ({error_type}) avg"].values
                    metric_std = df_p[f"{metric} ({error_type}) std"].values

                    # Plot data with error bars
                    ax.errorbar(x_values, metric_avg, yerr=metric_std, fmt='o-',
                                color=colors[p_idx % len(colors)],
                                capsize=5, markersize=6)

                    # Overlay raw data points if requested
                    if include_raw:
                        raw_data = df_p[f"{metric} ({error_type}) raw"].values
                        for j, raw_vals in enumerate(raw_data):
                            ax.scatter([x_values[j]] * len(raw_vals), raw_vals,
                                       alpha=0.7, color=light_colors[p_idx % len(light_colors)], s=20)

                    # Create x-tick labels
                    x_ticks = x_values
                    x_labels = [f"{x}\n({y_label[0]}={y})" for x, y in zip(x_values, y_values)]

                    ax.set_xticks(x_ticks)
                    ax.set_xticklabels(x_labels, rotation=45)

                    # Add trend line if we have enough points
                    if len(x_values) > 1:
                        # For better trends, sort the values by x
                        sorted_indices = np.argsort(x_values)
                        sorted_x = np.array(x_values)[sorted_indices]
                        sorted_y = np.array(metric_avg)[sorted_indices]

                        # Calculate degree for polynomial fit
                        degree = min(len(set(x_values)) - 1, 3)  # At most cubic, at least linear if possible
                        degree = max(degree, 1)  # Ensure at least linear if we have 2+ points

                        trend = np.polyfit(sorted_x, sorted_y, degree)
                        x_for_trend = np.linspace(min(sorted_x), max(sorted_x), 100)
                        trend_y = np.polyval(trend, x_for_trend)

                        ax.plot(x_for_trend, trend_y, 'k--', linewidth=2, label="Trend Line")

                    # Add labels and title
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(label)
                    ax.set_title(f"{label} vs. {x_label} (p={p}, C={coverage_target:.1f}x, {num_iterations} iterations)")
                    ax.grid(True, alpha=0.3)
                    if len(x_values) > 1:  # Only add legend if we have trend line
                        ax.legend()

                # Adjust layout
                plt.tight_layout()
                suffix = "_with_raw" if include_raw else ""

                # Save plot
                output_file = f"{full_path}/coverage_{coverage_target}_by_{x_axis_var}_p_{p}{suffix}.png"
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Individual plot saved to {output_file}")


# Main function to run all experiments
if __name__ == "__main__":
    run_experiments("sequence.fasta")
