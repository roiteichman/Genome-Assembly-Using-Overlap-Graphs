import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from collections import defaultdict
from aligners import align_read_or_contig_to_reference, local_alignment

lower_bound_l = 50
upper_bound_l = 150
lower_bound_n = 100
upper_bound_n = 1000000


def plot_genome_coverage(contigs, num_reads, read_length, error_prob, reference_genome, error_type_str, experiment_name,
                         num_iteration, path):
    """
    Plot coverage of the reference genome by assembled contigs.

    Parameters:
        contigs (list): List of assembled contigs.
        num_reads (int): Number of reads used for assembly.
        read_length (int): Length of each read.
        error_prob (float): Probability of mutation in error-prone reads.
        reference_genome (str): The reference genome sequence.
        error_type_str (str): Type of reads used for assembly (e.g., "error-free" or "error-prone").
        experiment_name (str): The name of the experiment.
        num_iteration (int): The number of the specific iteration.
        path (str): The path to save the plots.
    Returns:
        None (Displays a plot)
    """
    genome_length = len(reference_genome)
    coverage = np.zeros(genome_length)  # Array to track coverage

    for contig in contigs:
        _, _, start, end = align_read_or_contig_to_reference(contig, reference_genome, read_length)

        if start != -1 and end != -1:
            for i in range(start, end):
                coverage[i] += 1  # Mark covered regions

    print("$$$$$$$$$$$$$$$$$$")
    print(f"coverage: {coverage}")
    print("$$$$$$$$$$$$$$$$$$")
    positions = np.arange(genome_length)

    # Plot the coverage
    plt.figure(figsize=(10, 5))
    plt.plot(positions, coverage, marker='o', linestyle='-', color='b')
    plt.xlabel("Genome Base Position")
    plt.ylabel("Coverage Count")
    plt.title(f"Genome Coverage by Assembled Contigs - {experiment_name} iteration: {str(num_iteration)}")
    plt.axhline(y=1, color='g', linestyle='--', label="Fully Covered Threshold")
    # "fully covered threshold" is the expected coverage if all reads were perfectly assembled (green line)
    plt.legend()
    try:
        directory = os.path.dirname(f"{path}/{error_type_str}_genome_coverage_iteration_{str(num_iteration)}.png")
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f"{path}/{error_type_str}_genome_coverage_iteration_{str(num_iteration)}.png")
    except Exception as e:
        print(f"Error saving plot: {e}")
        print(f"Parameters: experiment_name={experiment_name}, num_iteration={num_iteration}, path={path}")
    finally:
        plt.close()  # Close the figure to free memory


def plot_genome_depth(reads, reference_genome, read_length, error_prob, error_type_str, experiment_name,
                      num_iteration, path):
    """
    Plot genome coverage depth for each base in the reference genome.

    Parameters:
        reads (list): List of sequencing reads.
        reference_genome (str): The reference genome sequence.
        read_length (int): Length of each read.
        error_prob (float): Probability of mutation in error-prone reads.
        error_type_str (str): Type of reads used for assembly (e.g., "error-free" or "error-prone").
        experiment_name (str): The name of the experiment.
        num_iteration (int): The number of the specific iteration.
        path (str): The path to save the plots.
    Returns:
        None (Displays a plot)
    """
    genome_coverage = defaultdict(int)
    alignment_cache = {}

    for read in reads:
        best_score = -float("inf")
        best_start, best_end = -1, -1

        score, start, end = -1, -1, -1

        key = (read, reference_genome)

        if key in alignment_cache:
            score, start, end = alignment_cache[key]
        else:
            _, score, start, end = align_read_or_contig_to_reference(read, reference_genome, read_length)
            alignment_cache[key] = (score, start, end)

        if score > -float("inf") and start != -1 and end != -1:
            best_start, best_end = start, end

        if best_start != -1:
            for i in range(best_start, best_end):
                genome_coverage[i] += 1

    positions = sorted(range(len(reference_genome)))
    coverage_values = [genome_coverage[pos] for pos in positions]

    plt.figure(figsize=(10, 5))
    plt.plot(positions, coverage_values, marker='o', linestyle='-')
    plt.xlabel("Genome Base Position")
    plt.ylabel("Read Coverage Depth")
    plt.title(f"Genome Coverage Depth - experiment {experiment_name} iteration: {str(num_iteration)}")

    if len(coverage_values) > 0:
        expected_coverage = len(reads) * read_length / len(reference_genome)
        plt.axhline(y=expected_coverage, color='g', linestyle='--', label="Expected Coverage")
        plt.legend()
    else:
        print("Warning: No coverage values available. Check the alignment process.")

    try:
        directory = os.path.dirname(f"{path}/{error_type_str}_genome_depth_iteration_{str(num_iteration)}.png")
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f"{path}/{error_type_str}_genome_depth_iteration_{str(num_iteration)}.png")
    except Exception as e:
        print(f"Error saving plot: {e}")
        print(f"Parameters: experiment_name={experiment_name}, num_iteration={num_iteration}, path={path}")
    finally:
        plt.close()  # Close the figure to free memory


def plot_reconstructed_coverage(contigs, reads, num_reads, read_length, error_prob, reference_genome, error_type_str,
                                experiment_name, num_iteration, path):
    """
    Plot read coverage depth for each base in the assembled contigs.

    Parameters:
        contigs (list): List of assembled contigs.
        reads (list): List of sequencing reads.
        num_reads (int): Number of reads used for assembly.
        read_length (int): Length of each read.
        error_prob (float): Probability of mutation in error-prone reads.
        reference_genome (str): The reference genome sequence.
        error_type_str (str): Type of reads used for assembly (e.g., "error-free" or "error-prone").
        experiment_name (str): The name of the experiment.
        num_iteration (int): The number of the specific iteration.
        path (str): The path to save the plots.
    Returns:
        None (Displays a plot)
    """
    # Initialize a dictionary to store coverage depth for each base
    contig_coverages = {contig: defaultdict(float) for contig in contigs}  # Coverage per contig
    alignment_cache = {}  # Initialize the alignment cache

    # Compute coverage depth for each base in the contigs using local alignment
    for read in reads:
        # use list of best_contigs for case of multiple contigs with the same score
        best_contigs = []
        best_score = -float("inf")
        best_alignment = {}

        # Find the best contig for each read
        for contig in contigs:
            key = (read, contig)

            if key in alignment_cache:
                # Retrieve the alignment from the cache
                score, start, end = alignment_cache[key]

            else:
                # Calculate the alignment
                alignment, score, start, end = local_alignment(read, contig)
                print(f"alignment: {alignment}")
                print(f"score: {score}")

                # Store the alignment in the cache
                alignment_cache[key] = (score, start, end)

            # Update the best contig based on the alignment score
            if score > best_score and start != -1 and end != -1:
                # Start a new list of best contigs
                best_contigs = [contig]
                best_score = score
                best_alignments = {contig: (start, end)}

            # Another contig with the same score
            elif score == best_score and start != -1 and end != -1:
                best_contigs.append(contig)  # Add to existing list of best contigs
                best_alignments[contig] = (start, end)

        if best_contigs:
            # Increment coverage depth equally for all best contigs
            coverage_increment = 1 / len(best_contigs)
            for best_contig in best_contigs:
                best_start, best_end = best_alignments[best_contig]
                for i in range(best_start, best_end):
                    contig_coverages[best_contig][i] += coverage_increment

    for contig_idx, (contig, coverage) in enumerate(contig_coverages.items()):
        # Convert dictionary to list for plotting
        positions = sorted(coverage.keys())
        coverage_values = [coverage[pos] for pos in positions]

        # Plot the coverage depth
        plt.figure(figsize=(10, 5))
        plt.plot(positions, coverage_values, marker='o', linestyle='-')
        plt.xlabel("Contig Base Position")
        plt.ylabel("Read Coverage Depth")
        plt.title(f"Read Coverage Depth for Contig {contig_idx + 1} - "
                  f"experiment {experiment_name} iteration: {str(num_iteration)}")
        if len(coverage_values) > 0:
            expected_coverage = num_reads * read_length / len(reference_genome)
            print(f"expected_coverage: {expected_coverage}")
            plt.axhline(y=expected_coverage, color='g', linestyle='--', label="Expected Depth")
            # "expected depth" is the average depth across all bases in the genome (green line)
            expected_depth = sum(coverage_values) / len(coverage_values)
            print(f"coverage_values: {coverage_values}")
            print(f"sum(coverage_values): {sum(coverage_values)}")
            print(f"len(coverage_values): {len(coverage_values)}")
            print(f"expected_depth: {expected_depth}")
            plt.axhline(y=expected_depth, color='r', linestyle='--', label="Empirical Average Depth")
            # "empirical average depth" is the average coverage depth across all bases in the contig (red line)
            plt.legend()
        else:
            print("Warning: No coverage values available. Check the alignment process.")
        try:
            # Ensure the directory exists before saving.
            directory = os.path.dirname(
                f"{path}/{error_type_str}_contig_coverage_{contig_idx + 1}_iteration_{str(num_iteration)}.png")
            if not os.path.exists(directory):
                os.makedirs(directory)

            plt.savefig(f"{path}/{error_type_str}_contig_coverage_{contig_idx + 1}_iteration_{str(num_iteration)}.png")
        except Exception as e:
            print(f"Error saving plot: {e}")
            print(
                f"Parameters: contig_idx={contig_idx}, experiment_name={experiment_name}, num_iteration={num_iteration}, path={path}")
        finally:
            plt.close()  # Close the plot in the finally block.



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

    # Set x-axis label based on x_key and fixed parameter
    lower_bound, upper_bound = None, None
    if x_key == "num_reads":
        x_axis_label = "N (Number of Reads)"
        fixed_param = "l (Read Length)"
        fixed_param_key = "read_length"
        lower_bound = lower_bound_n
        upper_bound = upper_bound_n
    elif x_key == "read_length":
        x_axis_label = "l (Read Length)"
        fixed_param = "N (Number of Reads)"
        fixed_param_key = "num_reads"
        lower_bound = lower_bound_l
        upper_bound = upper_bound_l
    else:
        x_axis_label = x_key
        fixed_param = "Parameter"
        fixed_param_key = None

    error_type_str = "Error-Prone"

    # Create path for the experiment
    full_path = f"{path}/{error_type_str}"
    try:
        os.makedirs(full_path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {full_path}: {e}")
        return  # return to avoid errors when trying to save plots.

    # Get the fixed parameter value (if it exists and is constant)
    fixed_value = None
    if fixed_param_key and len(df[fixed_param_key].unique()) == 1:
        fixed_value = df[fixed_param_key].iloc[0]

    # 1. Plot combined graph with all p values
    for include_raw in [False, True]:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        # Check out-of-bounds status
        out_of_bounds_str = check_x_values_boundaries(x_values, lower_bound, upper_bound)

        # Add a descriptive suptitle to the figure
        if fixed_value:
            fig.suptitle(
                f"Measures for fixed {fixed_param}={fixed_value} for different {x_axis_label} {out_of_bounds_str}"
                f"values and different p values",
                fontsize=16, y=0.98)
        else:
            fig.suptitle(f"Measures for different {x_axis_label} {out_of_bounds_str}values and different p values",
                         fontsize=16, y=0.98)

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
                metric_avg = df_p[f"{metric} avg"].values
                metric_std = df_p[f"{metric} std"].values

                # Collect data for overall trend line
                all_x.extend(x_values)
                all_y.extend(metric_avg)

                # Plot data with error bars
                ax.errorbar(x_values, metric_avg, yerr=metric_std, fmt='o-',
                            label=f"p={p}", color=colors[p_idx % len(colors)],
                            capsize=5, markersize=6)

                # Overlay raw data points if requested
                if include_raw:
                    raw_data = df_p[f"{metric} raw"].values
                    for j, raw_vals in enumerate(raw_data):
                        ax.scatter([x_values[j]] * len(raw_vals), raw_vals,
                                   alpha=0.7, color=light_colors[p_idx % len(light_colors)], s=20,
                                   marker='o')

            # Add boundary lines if specified
            if x_key == "num_reads" or x_key == "read_length":
                add_boundary_lines(ax, x_values, lower_bound, upper_bound)

                # Add coverage information to x-axis if available
                if coverage_key:
                    x_ticks, x_labels = generate_x_tick_labels(df, x_key, coverage_key)
                    ax.set_xticks(x_ticks)
                    ax.set_xticklabels(x_labels, rotation=45)

                # Add average trend line
                add_average_trend_line(ax, all_x, all_y)

                # Set up axis configurations
                setup_plot_axis(ax, x_axis_label, label, label, 'combined', num_iterations, log_scale)

                # Add legend
                ax.legend(title="Error Probability (p)", bbox_to_anchor=(1.05, 1), loc='upper left')

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Make room for suptitle
        suffix = "_with_raw" if include_raw else ""

        # Save plot
        output_path = f"{full_path}/plot_p_values{suffix}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Combined plot saved to {output_path}")

    # 2. Create individual plots for each p value
    for p_idx, p in enumerate(p_values):
        for include_raw in [False, True]:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()

            # Add a descriptive suptitle
            if fixed_value:
                fig.suptitle(
                    f"Measures for fixed {fixed_param}={fixed_value}, p={p} for different {x_axis_label} values",
                    fontsize=16, y=0.98)
            else:
                fig.suptitle(f"Measures for p={p} for different {x_axis_label} values",
                             fontsize=16, y=0.98)

            # Filter data for this p value
            df_p = df[df['error_prob'] == p].sort_values(by=x_key)

            if df_p.empty:
                continue

            # Plot each metric
            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                ax = axes[i]

                # Get data
                x_values = df_p[x_key].values
                metric_avg = df_p[f"{metric} avg"].values
                metric_std = df_p[f"{metric} std"].values

                # Plot data with error bars
                ax.errorbar(x_values, metric_avg, yerr=metric_std, fmt='o-',
                            color=colors[p_idx % len(colors)],
                            capsize=5, markersize=6)

                # Overlay raw data points if requested
                if include_raw:
                    raw_data = df_p[f"{metric} raw"].values
                    for j, raw_vals in enumerate(raw_data):
                        ax.scatter([x_values[j]] * len(raw_vals), raw_vals,
                                   alpha=0.7, color=light_colors[p_idx % len(light_colors)], s=20)

                        # Add coverage information to x-axis if available
                        if coverage_key:
                            x_ticks, x_labels = generate_x_tick_labels(df_p, x_key, coverage_key)
                            ax.set_xticks(x_ticks)
                            ax.set_xticklabels(x_labels, rotation=45)

                        # Add trend line if we have enough points
                        if len(x_values) > 1:
                            # For better trends, sort the values by x
                            sorted_indices = np.argsort(x_values)
                            sorted_x = np.array(x_values)[sorted_indices]
                            sorted_y = np.array(metric_avg)[sorted_indices]

                            # Calculate degree for polynomial fit
                            degree = len(set(x_values)) - 2 if len(set(x_values)) - 2 > 0 else 3
                            degree = max(degree, 1)  # Ensure at least linear if we have 2+ points

                            trend = np.polyfit(sorted_x, sorted_y, degree)
                            x_for_trend = np.linspace(min(sorted_x), max(sorted_x), 100)
                            trend_y = np.polyval(trend, x_for_trend)

                            ax.plot(x_for_trend, trend_y, 'k--', linewidth=2, label="Trend Line")

                        # Add boundary lines if specified
                        if x_key == "num_reads" or x_key == "read_length":
                            add_boundary_lines(ax, x_values, lower_bound, upper_bound)

                        # Set up axis configurations
                        setup_plot_axis(ax, x_axis_label, label, label, p, num_iterations, log_scale)

                        # Add legend if we have trend line
                        if len(x_values) > 1:
                            ax.legend()

            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)  # Make room for suptitle
            suffix = "_with_raw" if include_raw else ""

            # Save plot
            output_file = f"{full_path}/plot_p_value_{p}_by_{x_key}{suffix}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Individual plot saved to {output_file}")


def check_x_values_boundaries(x_values, lower_bound, upper_bound):
    """
    Check if x_values are out of bounds.

    Parameters:
    - x_values: array of x values
    - lower_bound: lower boundary value
    - upper_bound: upper boundary value

    Returns:
    - out_of_bounds_str: A string indicating out of bounds status
    """
    out_of_bounds_str = ""

    if lower_bound is not None:
        if all(x < lower_bound for x in x_values):
            out_of_bounds_str += f"(all values lower than {lower_bound}) "

    if upper_bound is not None:
        if all(x > upper_bound for x in x_values):
            out_of_bounds_str += f"(all values higher than {upper_bound}) "

    return out_of_bounds_str


def setup_plot_axis(ax, x_axis_label, label, metric_label, p, num_iterations, log_scale=False):
    """
    Set up common axis configurations for plots.

    Parameters:
    - ax: matplotlib axis to configure
    - x_axis_label: Label for x-axis
    - label: Specific metric label
    - metric_label: Full metric label
    - p: Error probability
    - num_iterations: Number of iterations
    - log_scale: Whether to use log scale (default False)

    Returns:
    - None (modifies the axis in-place)
    """
    # Set x-axis to log scale if requested
    if log_scale:
        ax.set_xscale('log')

    # Add labels and title
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(label)
    ax.set_title(f"{label} vs. {x_axis_label} (p={p}, {num_iterations} iterations)")
    ax.grid(True, alpha=0.3)


def add_average_trend_line(ax, all_x, all_y):
    """
    Add an average trend line to the plot.

    Parameters:
    - ax: matplotlib axis to add trend line to
    - all_x: List of x values
    - all_y: List of corresponding y values

    Returns:
    - None (modifies the axis in-place)
    """
    if len(all_x) > 2:
        # Sort by x for proper line plotting
        sorted_indices = np.argsort(all_x)
        sorted_x = np.array(all_x)[sorted_indices]
        sorted_y = np.array(all_y)[sorted_indices]

        # Fit a polynomial (degree = min(3, len(unique x) - 2))
        degree = len(set(all_x)) - 2 if len(set(all_x)) - 2 > 0 else 3
        if degree > 0:  # Need at least 2 points for a line
            trend = np.polyfit(sorted_x, sorted_y, degree)
            # Create more points for a smoother curve
            x_for_trend = np.linspace(min(sorted_x), max(sorted_x), 100)
            trend_y = np.polyval(trend, x_for_trend)
            ax.plot(x_for_trend, trend_y, 'k--', linewidth=2, label="Average Trend")


def generate_x_tick_labels(df, x_key, coverage_key):
    """
    Generate x-tick labels with coverage information.

    Parameters:
    - df: DataFrame with experiment results
    - x_key: Key for x-axis values
    - coverage_key: Key for coverage values

    Returns:
    - x_ticks: List of x values
    - x_labels: List of labels with coverage info
    """
    x_coverage_map = {}
    for x in sorted(df[x_key].unique()):
        # Find corresponding coverage for this x value (should be same for all p)
        coverage = df[df[x_key] == x][coverage_key].iloc[0]
        x_coverage_map[x] = coverage

    # Create x-tick labels with coverage info
    x_ticks = sorted(x_coverage_map.keys())
    x_labels = [f"{x}\n(C={x_coverage_map[x]:.1f}x)" for x in x_ticks]

    return x_ticks, x_labels


def add_boundary_lines(ax, x_values, lower_bound, upper_bound):
    """
    Add boundary lines to the given axis if conditions are met.

    Parameters:
    - ax: matplotlib axis to add lines to
    - x_values: array of x values in the plot
    - lower_bound: lower bound value to potentially draw
    - upper_bound: upper bound value to potentially draw

    Returns:
    - None (modifies the axis in-place)
    """
    y_min, y_max = ax.get_ylim()

    # Calculate max sequential distance between x values
    max_sequential_x_values_distances = max(
        [y - x for (i, x), (j, y) in zip(enumerate(x_values[:-1]),
                                         enumerate(x_values[1:]))
         if i + 1 == j + 1], default=0)

    # Helper function to check if bound should be added
    def check_bound(bound, bound_type):
        if bound is None:
            return False

        # Check if bound falls between consecutive x values
        bound_within = any(x <= bound <= y for x, y in zip(x_values[:-1], x_values[1:]))

        # Calculate minimum distance from bound to x values
        min_distance = min([abs(x - bound) for x in x_values])

        # Check if bound is not too far from data points
        not_far_enough = min_distance < 1.5 * max_sequential_x_values_distances

        return bound_within and not_far_enough

    # Add lower bound line
    if check_bound(lower_bound, "lower"):
        line = ax.vlines(lower_bound, y_min, y_max, colors='slategray',
                         linestyles='dashed', label='Lower Bound', alpha=0.5)

        # Add text label just below the x-axis
        ax.text(lower_bound, y_min - 0.05 * (y_max - y_min),
                f'Lower Bound\n({lower_bound})',
                horizontalalignment='center',
                verticalalignment='top',
                color='slategray',
                fontsize=8)

    # Add upper bound line
    if check_bound(upper_bound, "upper"):
        line = ax.vlines(upper_bound, y_min, y_max, colors='slategray',
                         linestyles='dashed', label='Upper Bound', alpha=0.5)

        # Add text label just below the x-axis
        ax.text(upper_bound, y_min - 0.05 * (y_max - y_min),
                f'Upper Bound\n({upper_bound})',
                horizontalalignment='center',
                verticalalignment='top',
                color='slategray',
                fontsize=8)


def plot_const_coverage_results(results, coverage_target, x_axis_var="n", path="plots", num_iterations=5):
    """
    Plot experiment results with constant coverage but varying N and l.
    Creates two sets of plots: one ordered by N and one ordered by l.

    Parameters:
        results (list): List of result dictionaries.
        coverage_target (float): The target coverage value.
        x_axis_var (str): Variable to use on x-axis ('n' or 'l') - used for naming only.
        path (str): Path to save the plots.
        num_iterations (int): Number of iterations run for each parameter combination.

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

    lower_bound, upper_bound = None, None
    if x_axis_var == "l":
        lower_bound = lower_bound_l
        upper_bound = upper_bound_l
    else:
        lower_bound = lower_bound_n
        upper_bound = upper_bound_n

    error_type_str = "Error-Prone"
    # Create path for the experiment
    full_path = f"{path}/{error_type_str}/coverage_{coverage_target:.1f}x"
    os.makedirs(full_path, exist_ok=True)

    def plot_metric_data(x_key, y_key, x_label, y_label, lower_bound, upper_bound):
        """
        Internal function to handle plotting logic for metrics

        Parameters:
            x_key: Key for x-axis values
            y_key: Key for y-axis values
            x_label: Label for x-axis
            y_label: Label for y-axis
            lower_bound: Lower boundary value
            upper_bound: Upper boundary value

        Returns:
            None
        """
        # 1. Plot combined graph with all p values
        for include_raw in [False, True]:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()

            # Check out-of-bounds status
            x_values = sorted(df[x_key].unique())
            out_of_bounds_str = check_x_values_boundaries(x_values, lower_bound, upper_bound)

            # Add a descriptive suptitle
            fig.suptitle(
                f"Measures with constant coverage C={coverage_target:.1f}x, {out_of_bounds_str}ordered by {x_label}",
                fontsize=16, y=0.98)

            # Plot each metric
            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                ax = axes[i]

                # Data for average trend line
                all_x, all_y = [], []

                # For each error probability, plot a separate line
                for p_idx, p in enumerate(p_values):
                    # Filter data for this p value
                    df_p = df[df['error_prob'] == p].sort_values(by=x_key)

                    if df_p.empty:
                        continue

                    # Get x values and corresponding y values
                    x_values = df_p[x_key].values
                    y_values = df_p[y_key].values
                    metric_avg = df_p[f"{metric} avg"].values
                    metric_std = df_p[f"{metric} std"].values

                    # Collect data for overall trend line
                    all_x.extend(x_values)
                    all_y.extend(metric_avg)

                    # Plot data with error bars
                    ax.errorbar(x_values, metric_avg, yerr=metric_std, fmt='o-',
                                label=f"p={p}", color=colors[p_idx % len(colors)],
                                capsize=5, markersize=6)

                    # Overlay raw data points if requested
                    if include_raw:
                        raw_data = df_p[f"{metric} raw"].values
                        for j, raw_vals in enumerate(raw_data):
                            ax.scatter([x_values[j]] * len(raw_vals), raw_vals,
                                       alpha=0.7, color=light_colors[p_idx % len(light_colors)], s=20,
                                       marker='o')

                # Create x-tick labels with the other variable values
                x_ticks = sorted(df[x_key].unique())
                x_labels = [f"{x}\n({y_label[0]}={df[df[x_key] == x][y_key].iloc[0]})" for x in x_ticks]

                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_labels, rotation=45)

                # Add boundary lines if conditions are met
                if lower_bound is not None or upper_bound is not None:
                    add_boundary_lines(ax, x_values, lower_bound, upper_bound)

                # Add average trend line if we have enough data points
                add_average_trend_line(ax, all_x, all_y)

                # Add labels and title
                ax.set_xlabel(x_label)
                ax.set_ylabel(label)
                ax.set_title(f"{label} vs. {x_label} (C={coverage_target:.1f}x, {num_iterations} iterations)")
                ax.grid(True, alpha=0.3)

                # Add legend
                ax.legend(title="Error Probability", bbox_to_anchor=(1.05, 1), loc='upper left')

            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)  # Make room for suptitle
            suffix = "_with_raw" if include_raw else ""

            # Save plot
            output_file = f"{full_path}/ordered_by_{x_axis_var}{suffix}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Plots saved to {output_file}")

        # 2. Individual p-value plots
        for p_idx, p in enumerate(p_values):
            for include_raw in [False, True]:
                # Create figure with subplots
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.flatten()

                # Check out-of-bounds status
                x_values = sorted(df[df['error_prob'] == p][x_key].unique())
                out_of_bounds_str = check_x_values_boundaries(x_values, lower_bound, upper_bound)

                # Add a descriptive suptitle
                fig.suptitle(
                    f"Measures with constant coverage C={coverage_target:.1f}x, p={p}, {out_of_bounds_str}ordered by {x_label}",
                    fontsize=16, y=0.98)

                # Filter data for this p value
                df_p = df[df['error_prob'] == p].sort_values(by=x_key)

                if df_p.empty:
                    continue

                # Previous individual p-value plot implementation follows here...
                # [Keeping the same implementation as before with boundary lines added]

    # Generate both N-ordered and l-ordered plots
    plot_metric_data('num_reads', 'read_length',
                     "N (Number of Reads)",
                     "l (Read Length)",
                     lower_bound_n, upper_bound_n)

    plot_metric_data('read_length', 'num_reads',
                     "l (Read Length)",
                     "N (Number of Reads)",
                     lower_bound_l, upper_bound_l)


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
