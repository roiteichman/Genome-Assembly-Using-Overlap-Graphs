import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from aligners import align_read_or_contig_to_reference
from consts import get_lower_bound_n, get_upper_bound_n, get_lower_bound_l, get_upper_bound_l, get_metrics, get_metric_labels

lower_bound_l = get_lower_bound_l()
upper_bound_l = get_upper_bound_l()
lower_bound_n = get_lower_bound_n()
upper_bound_n = get_upper_bound_n()
metrics = get_metrics()
metric_labels = get_metric_labels()


def plot_genome_coverage(coverage, genome_length, experiment_name, num_iteration, path):
    """
    Plot coverage of the reference genome by assembled contigs.

    Parameters:
        coverage (np.array): Array of coverage values for each base in the reference genome.
        genome_length (int): Length of the reference genome.
        experiment_name (str): The name of the experiment.
        num_iteration (int): The number of the specific iteration.
        path (str): The path to save the plots.
    Returns:
        None (Displays a plot)
    """

    coverage_boolean_np = coverage > 0
    positions = np.arange(genome_length)

    # Plot the coverage
    plt.figure(figsize=(10, 5))
    plt.plot(positions, coverage_boolean_np, marker='o', linestyle='-', color='b')
    plt.xlabel("Genome Base Position")
    plt.ylabel("Coverage Count")
    plt.title(f"Genome Coverage by Assembled Contigs - {experiment_name} iteration: {str(num_iteration)}")
    plt.axhline(y=1, color='g', linestyle='--', label="Fully Covered Threshold")
    # "fully covered threshold" is the expected coverage if all reads were perfectly assembled (green line)
    plt.legend()
    try:
        directory = os.path.dirname(f"{path}/genome_coverage_iteration_{str(num_iteration)}.png")
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f"{path}/genome_coverage_iteration_{str(num_iteration)}.png")
    except Exception as e:
        print(f"Error saving plot: {e}")
        print(f"Parameters: experiment_name={experiment_name}, num_iteration={num_iteration}, path={path}")
    finally:
        plt.close()  # Close the figure to free memory


def plot_genome_depth(coverage, expected_coverage, genome_length, experiment_name, num_iteration, path):
    """
    Plot genome coverage depth for each base in the reference genome.

    Parameters:
        coverage (np.array): Array of coverage values for each base in the reference genome.
        expected_coverage (float): The expected coverage depth.
        genome_length (int): Length of the reference genome.
        experiment_name (str): The name of the experiment.
        num_iteration (int): The number of the specific iteration.
        path (str): The path to save the plots.
    Returns:
        None (Displays a plot)
    """
    positions = np.arange(genome_length)

    plt.figure(figsize=(10, 5))
    plt.plot(positions, coverage, marker='o', linestyle='-')
    plt.xlabel("Genome Base Position")
    plt.ylabel("Read Coverage Depth")
    plt.title(f"Genome Coverage Depth - experiment {experiment_name} iteration: {str(num_iteration)}")

    if len(coverage) > 0:
        plt.axhline(y=expected_coverage, color='g', linestyle='--', label="Expected Coverage")
        plt.legend()
    else:
        print("Warning: No coverage values available. Check the alignment process.")

    try:
        directory = os.path.dirname(f"{path}/genome_depth_iteration_{str(num_iteration)}.png")
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f"{path}/genome_depth_iteration_{str(num_iteration)}.png")
    except Exception as e:
        print(f"Error saving plot: {e}")
        print(f"Parameters: experiment_name={experiment_name}, num_iteration={num_iteration}, path={path}")
    finally:
        plt.close()  # Close the figure to free memory


def plot_reconstructed_coverage(contigs, reads, num_reads, read_length, reference_genome,
                                experiment_name, num_iteration, path):
    """
    Plot read coverage depth for each base in the assembled contigs.
    This means the amount of reads that align to each base in the contigs.

    Parameters:
        contigs (list): List of assembled contigs.
        reads (list): List of sequencing reads.
        num_reads (int): Number of reads used for assembly.
        read_length (int): Length of each read.
        reference_genome (str): The reference genome sequence.
        experiment_name (str): The name of the experiment.
        num_iteration (int): The number of the specific iteration.
        path (str): The path to save the plots.
    Returns:
        None (Displays a plot)
    """
    # Initialize a dictionary to store coverage depth for each base
    contig_lengths = {contig: len(contig) for contig in contigs}
    contig_coverages = {contig: np.zeros(length, dtype=float) for contig, length in contig_lengths.items()}

    for read in reads:
        best_score = -float('inf')
        best_contigs = []

        for contig in contigs:
            alignment, score, start, end = align_read_or_contig_to_reference(read, contig, read_length)

            if score > best_score and start != -1 and end != -1:
                best_score = score
                best_contigs = [contig]
                best_start, best_end = start, end
            elif score == best_score and start != -1 and end != -1:
                best_contigs.append(contig)

        if best_contigs:
            chosen_contig = np.random.choice(best_contigs)
            contig_coverages[chosen_contig][best_start:best_end] += 1

    for contig_idx, contig in enumerate(contigs):
        coverage = contig_coverages[contig]
        positions = np.arange(contig_lengths[contig])

        plt.figure(figsize=(10, 5))
        plt.plot(positions, coverage, marker='o', linestyle='-')
        plt.xlabel("Contig Base Position")
        plt.ylabel("Read Coverage Depth")
        plt.title(f"Read Coverage Depth for Contig {contig_idx + 1} - "
                  f"experiment {experiment_name} iteration: {str(num_iteration)}")

        if len(coverage) > 0:
            expected_coverage = num_reads * read_length / len(reference_genome)
            plt.axhline(y=expected_coverage, color='g', linestyle='--', label=f"Expected Depth")
            expected_depth = float(np.mean(coverage))
            plt.axhline(y=expected_depth, color='r', linestyle='--', label=f"Empirical Average Depth")
            plt.legend()
        else:
            print("Warning: No coverage values available. Check the alignment process.")

        try:
            directory = os.path.dirname(
                f"{path}/contig_coverage_{contig_idx + 1}_iteration_{str(num_iteration)}.png")
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(f"{path}/contig_coverage_{contig_idx + 1}_iteration_{str(num_iteration)}.png")
        except Exception as e:
            print(f"Error saving plot: {e}")
            print(f"Parameters: contig_idx={contig_idx}, experiment_name={experiment_name}, "
                  f"num_iteration={num_iteration}, path={path}")
        finally:
            plt.close()


def plot_experiment_results_by_p_values(results, x_key="num_reads", coverage_key="expected_coverage",
                                        path="plots", log_scale=False, num_iterations=10, separator=None):
    """
    Plot experiment results with x_key values as x-axis and different error probabilities as separate series.
    Includes individual plots for each p value with trend lines, and combined plots, both includes version with raw data.
    Oppitionaly is to separate the plot into 3 different plots base on the x_key values:
    1 - Regular - all the x_key values.
    2 - Smaller or Equal - only the x_key values that are smaller than the separator or equal to it.
    3 - Bigger or Equal - only the x_key values that are bigger than the separator or equal to it.

    Parameters:
        results (list): List of result dictionaries.
        x_key (str): Dictionary key for x-axis values (typically "num_reads" or "read_length").
        coverage_key (str): Dictionary key for coverage values or None.
        path (str): Path to save the plots.
        log_scale (bool): Whether to use logarithmic scale for x-axis.
        num_iterations (int): Number of iterations to run for each parameter combination.
        separator (int): Optional separator for x_key values.

    Returns:
        None
    """
    # Create dataframe from results
    df = pd.DataFrame(results)

    # Get unique error probabilities and x values
    p_values = sorted(df['error_prob'].unique())
    x_values = sorted(df[x_key].unique())

    all_x_values = x_values
    smaller_or_equal_x_values = None
    bigger_or_equal_x_values = None

    plot_x_value_set = [(x_values, f"{x_key}")]

    if separator is not None:
        smaller_or_equal_x_values = [x for x in x_values if x <= separator]
        bigger_or_equal_x_values = [x for x in x_values if x >= separator]

        if len(smaller_or_equal_x_values) > 3 and smaller_or_equal_x_values != x_values:
            plot_x_value_set.append((smaller_or_equal_x_values, f"{x_key}_le_{separator}"))

        plot_x_value_set.extend([
            (smaller_or_equal_x_values, f"{x_key}_le_{separator}"),
            (bigger_or_equal_x_values, f"{x_key}_ge_{separator}")
        ])

    # Plotting function to avoid code duplication
    def plot_with_x_values(plot_x_values, plot_type_suffix):
        """
        Plot the experiment results with the specified x values.

        Parameters:
            plot_x_values (list): List of x values to plot.
            plot_type_suffix (str): Suffix for the plot type.

        Returns:
            None
        """
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

        # Get the fixed parameter value (if it exists and is constant)
        fixed_value = None
        if fixed_param_key and len(df[fixed_param_key].unique()) == 1:
            fixed_value = df[fixed_param_key].iloc[0]

        # Check out-of-bounds status
        out_of_bounds_str = check_x_values_boundaries(plot_x_values, lower_bound, upper_bound)

        # Filter data for the x values we want to plot
        df_filtered = df[df[x_key].isin(plot_x_values)]

        #  Plot types: (1) combined (all p values) and (2) individual (each p value)
        # 1. Plot combined graph with all p values
        for include_raw in [False, True]:
            # Create figure with subplots
            fig, axes = create_figure()

            # Add a descriptive suptitle to the figure
            if fixed_value:
                fig.suptitle(
                    f"Measures for fixed {fixed_param}={fixed_value} for different {x_axis_label} {out_of_bounds_str}"
                    f"values and different p values",
                    fontsize=28)
            else:
                fig.suptitle(f"Measures for different {x_axis_label} {out_of_bounds_str}values and different p values",
                             fontsize=28)

            # Plot each metric
            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                ax = axes[i]

                # Data for average trend line
                all_x = []
                all_y = []

                # For each error probability, plot a separate line
                for p_idx, p in enumerate(p_values):
                    # Filter data for this p value
                    df_p = df_filtered[df_filtered['error_prob'] == p].sort_values(by=x_key)

                    if df_p.empty:
                        continue

                    x_values_p = df_p[x_key].values
                    metric_avg = df_p[f"{metric} avg"].values
                    metric_std = df_p[f"{metric} std"].values

                    # Collect data for overall trend line
                    all_x.extend(x_values_p)
                    all_y.extend(metric_avg)

                    # Plot data with error bars
                    ax.errorbar(x_values_p, metric_avg, yerr=metric_std, fmt='o-',
                                label=f"p={p}", color=colors[p_idx % len(colors)],
                                capsize=5, markersize=6)

                    # Overlay raw data points if requested
                    if include_raw:
                        raw_data = df_p[f"{metric} raw"].values
                        for j, raw_vals in enumerate(raw_data):
                            ax.scatter([x_values_p[j]] * len(raw_vals), raw_vals,
                                       alpha=0.7, color=light_colors[p_idx % len(light_colors)], s=20,
                                       marker='o')

                # set up axis configurations
                setup_plot_axis(ax, x_axis_label, metric, label, 'combined', num_iterations, log_scale)

                # Add coverage information to x-axis if available
                if coverage_key:
                    x_ticks, x_labels = generate_x_tick_labels(df_filtered, x_key, coverage_key)
                    ax.set_xticks(x_ticks)
                    ax.set_xticklabels(x_labels, rotation=45)
                    ax.tick_params(axis='both', labelsize=18)

                # Add average trend line
                add_average_trend_line(ax, all_x, all_y, log_scale=log_scale)

                # Add legend
                ax.legend(title="Error Probability (p)", loc='upper right', fontsize=16)

            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Make room for suptitle

            is_raw = "_with_raw" if include_raw else ""
            # Create path for the experiment
            full_path = f"{path}/{plot_type_suffix}/p_values_combined_{is_raw}.png"
            try:
                directory = os.path.dirname(full_path)
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                # Save plot
                plt.savefig(full_path, dpi=300, bbox_inches='tight')
            except OSError as e:
                print(f"Error creating directory {full_path}: {e}")
                return  # return to avoid errors when trying to save plots.
            except Exception as e:
                print(f"Error saving plot: {e}")
                print(f"Parameters: experiment_name={path[path.find('experiment'):]}, path={path}")
            finally:
                plt.close() # Close the figure to free memory

        # 2. Create individual plots for each p value
        for p_idx, p in enumerate(p_values):
            for include_raw in [False, True]:
                # Create figure with subplots
                is_raw = "_with_raw" if include_raw else ""
                fig, axes = create_figure()
                # Add a descriptive suptitle
                if fixed_value:
                    fig.suptitle(
                        f"Measures for fixed {fixed_param}={fixed_value}, p={p} for different {x_axis_label}  {out_of_bounds_str}values",
                        fontsize=28)
                else:
                    fig.suptitle(f"Measures for p={p} for different {x_axis_label}  {out_of_bounds_str}values",
                                 fontsize=28)

                # Filter data for this p value
                df_p = df_filtered[df_filtered['error_prob'] == p].sort_values(by=x_key)

                if df_p.empty:
                    continue

                # Plot each metric
                for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                    ax = axes[i]

                    # Get data
                    x_values_p = df_p[x_key].values
                    metric_avg = df_p[f"{metric} avg"].values
                    metric_std = df_p[f"{metric} std"].values

                    # Plot data with error bars
                    ax.errorbar(x_values_p, metric_avg, yerr=metric_std, fmt='o-',
                                color=colors[p_idx % len(colors)],
                                capsize=5, markersize=6)

                    # Overlay raw data points if requested
                    if include_raw:
                        raw_data = df_p[f"{metric} raw"].values
                        for j, raw_vals in enumerate(raw_data):
                            ax.scatter([x_values_p[j]] * len(raw_vals), raw_vals,
                                       alpha=0.7, color=light_colors[p_idx % len(light_colors)], s=20)

                    # Set up axis configurations
                    setup_plot_axis(ax, x_axis_label, metric, label, p, num_iterations, log_scale)

                    # Add coverage information to x-axis if available
                    if coverage_key:
                        x_ticks = x_values_p
                        x_labels = [f"{x}\n(C={df_p[df_p[x_key] == x][coverage_key].iloc[0]:.1f}x)" for x in x_ticks]
                        ax.set_xticks(x_ticks)
                        ax.set_xticklabels(x_labels, rotation=45)
                        ax.tick_params(axis='both', labelsize=18)

                    # Add trend line if we have enough points
                    add_average_trend_line(ax, x_values_p, metric_avg, log_scale)

                    # Add legend if we have trend line
                    if len(x_values_p) > 1:
                        ax.legend(loc='upper right', fontsize=12)

                # Adjust layout
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Make room for suptitle

                # Create path for the experiment
                full_path = f"{path}/{plot_type_suffix}/p_value_{p}/ordered_by_{x_key}_{is_raw}.png"
                try:
                    directory = os.path.dirname(full_path)
                    if not os.path.exists(directory):
                        os.makedirs(directory, exist_ok=True)
                    # Save plot
                    plt.savefig(full_path, dpi=300, bbox_inches='tight')
                except OSError as e:
                    print(f"Error creating directory {full_path}: {e}")
                    return  # return to avoid errors when trying to save plots.
                except Exception as e:
                    print(f"Error saving plot: {e}")
                    print(f"Parameters: experiment_name={path[path.find('experiment'):]}, path={path}")
                finally:
                    plt.close()  # Close the figure to free memory

    # Call plotting function for different x value sets
    # Iterate through x value sets
    for x_values_set, suffix in plot_x_value_set:
        plot_with_x_values(x_values_set, suffix)


def plot_const_coverage_results(results, coverage_target, x_axis_var="n", path="plots", num_iterations=10, log_scale=False):
    """
    Plot experiment results with constant coverage but varying N and l.
    Creates two sets of plots: one ordered by N and one ordered by l.

    Parameters:
        results (list): List of result dictionaries.
        coverage_target (float): The target coverage value.
        x_axis_var (str): Variable to use on x-axis ('n' or 'l') - used for naming only.
        path (str): Path to save the plots.
        num_iterations (int): Number of iterations run for each parameter combination.
        log_scale (bool): Whether to use logarithmic scale for x-axis.

    Returns:
        None
    """
    # Create dataframe from results
    df = pd.DataFrame(results)

    # Get unique error probabilities
    p_values = sorted(df['error_prob'].unique())

    # Colors for different p values
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    light_colors = ['#a6cee3', '#fdbf6f', '#b2df8a', '#fb9a99', '#cab2d6', '#d2b48c']

    # Set x-axis label based on x_axis_var
    lower_bound = lower_bound_l if x_axis_var == "l" else lower_bound_n
    upper_bound = upper_bound_l if x_axis_var == "l" else upper_bound_n


    # Create path for the experiment
    full_path = f"{path}/summary_plots"
    try:
        os.makedirs(full_path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {full_path}: {e}")
        return  # Return to prevent errors when trying to save plots.

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
                fig, axes = create_figure()
                # Check out-of-bounds status
                x_values = sorted(df[x_key].unique())
                out_of_bounds_str = check_x_values_boundaries(x_values, lower_bound, upper_bound)

                # Create x-labels with fallback mechanism
                x_labels = []
                for x in x_values:
                    # Try to get a representative y value for this x
                    subset = df[df[x_key] == x]

                    # If there are multiple values, get the first or median
                    if len(subset) > 1:
                        y_val = subset[y_key].iloc[0]  # first value
                    else:
                        y_val = subset[y_key].values[0] if len(subset) > 0 else 'N/A'

                    x_labels.append(f"{x}\n({y_label[0]}={y_val})")

                # Add a descriptive suptitle
                fig.suptitle(
                    f"Measures with constant coverage C={coverage_target:.1f}x, {out_of_bounds_str}ordered by {x_label}",
                    fontsize=28)

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
                    ax.tick_params(axis='both', labelsize=18)

                    # Use setup_plot_axis to configure axis
                    setup_plot_axis(ax, x_label, metric, label, 'combined', num_iterations, log_scale)

                    # Add average trend line if we have enough data points
                    add_average_trend_line(ax, all_x, all_y, log_scale)

                    # Add legend
                    ax.legend(title="Error Probability", loc='upper right', fontsize=12)

                # Adjust layout
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Make room for suptitle
                suffix = "_with_raw" if include_raw else ""

                # Save plot
                output_file = f"{full_path}/ordered_by_{x_axis_var}{suffix}.png"
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()

            # 2. Individual p-value plots
            for p_idx, p in enumerate(p_values):
                for include_raw in [False, True]:
                    # Create figure with subplots
                    fig, axes = create_figure()

                    # Check out-of-bounds status
                    x_values = sorted(df[df['error_prob'] == p][x_key].unique())
                    out_of_bounds_str = check_x_values_boundaries(x_values, lower_bound, upper_bound)

                    # Add a descriptive suptitle
                    fig.suptitle(
                        f"Measures with constant coverage C={coverage_target:.1f}x, p={p}, {out_of_bounds_str}ordered by {x_label}",
                        fontsize=28)

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

                        # Create x-tick labels
                        x_ticks = x_values
                        x_labels = [f"{x}\n({y_label[0]}={y})" for x, y in zip(x_values, y_values)]

                        ax.set_xticks(x_ticks)
                        ax.set_xticklabels(x_labels, rotation=45)
                        ax.tick_params(axis='both', labelsize=18)

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

                        # Use setup_plot_axis to handle axis configuration consistently
                        setup_plot_axis(ax, x_label, metric, label, p, num_iterations, log_scale)

                        # Add legend if we have trend line
                        if len(x_values) > 1:
                            ax.legend(loc='upper right', fontsize=12)

                    # Adjust layout
                    plt.tight_layout(rect=[0, 0, 1, 0.95])
                    plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Make room for suptitle
                    suffix = "_with_raw" if include_raw else ""

                    # Save plot
                    output_file = f"{full_path}/ordered_by_{x_axis_var}_p_{p}{suffix}.png"
                    plt.savefig(output_file, dpi=300, bbox_inches='tight')
                    plt.close()

    # Generate both N-ordered and l-ordered plots
    plot_metric_data('num_reads', 'read_length',
                     "N (Number of Reads)",
                     "l (Read Length)",
                     lower_bound, upper_bound)

    plot_metric_data('read_length', 'num_reads',
                     "l (Read Length)",
                     "N (Number of Reads)",
                     lower_bound, upper_bound)


def plot_coverage_comparison(all_coverage_results, genome_length, path="plots", log_scale=False):
    """
    Plot coverage comparison for different metrics and error probabilities.

    Parameters:
    - all_coverage_results: Dictionary with coverage results for different error probabilities.
    - genome_length: The genome length which we covered C times.
    - path: Path to save the plots.
    - log_scale: Whether to use log scale (default False)

    Returns:
    - None
    """
    def plot_const_p_coverage(all_results):
        """
        Plot with coverage levels as different error bars for each p-value
        """
        # Create figure with subplots
        fig, axes = create_figure()
        # Plot each metric
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i]
            # For each error probability, plot a separate line
            for p in set(result['error_prob'] for results in all_results.values() for result in results):
                coverage_metric_means = []
                coverage_metric_stds = []
                coverage_values = []

                for C, results in all_results.items():
                    p_results = [r for r in results if r['error_prob'] == p]
                    if p_results:
                        means = [r[f"{metric} avg"] for r in p_results]
                        stds = [r[f"{metric} std"] for r in p_results]

                        coverage_metric_means.append(np.mean(means))
                        coverage_metric_stds.append(np.mean(stds))
                        coverage_values.append(C)

                ax.errorbar(coverage_values, coverage_metric_means,
                            yerr=coverage_metric_stds,
                            label=f'p = {p}',
                            marker='o')

            # set up axis configurations
            setup_plot_axis(ax, f'Coverage (C times {genome_length})', metric, label, log_scale=log_scale)
            ax.legend(loc='upper right', fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Make room for suptitle
        os.makedirs(path, exist_ok=True)
        plt.savefig(f"{path}/coverage_comparison_const_p.png")

    def plot_coverage_trend_lines(all_results):
        """Plot trend lines for coverage levels across metrics"""
        fig, axes = create_figure()

        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i]

            x_values = []
            y_values = []

            for C, results in all_results.items():
                means = [r[f"{metric} avg"] for r in results]
                x_values.append(C)
                y_values.append(np.mean(means))

            # Plot original points and trend line
            ax.scatter(x_values, y_values, label='Coverage Points')
            add_average_trend_line(ax, x_values, y_values, log_scale=log_scale)
            # set up axis configurations
            setup_plot_axis(ax, f'Coverage (C times {genome_length})', metric, label, log_scale=log_scale)

            ax.legend(loc='upper right', fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Make room for suptitle
        plt.savefig(f"{path}/coverage_comparison_trend.png")

    # Plot both visualizations
    plot_const_p_coverage(all_coverage_results)
    plot_coverage_trend_lines(all_coverage_results)


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


def setup_plot_axis(ax, x_axis_label, metric, metric_label, p=None, num_iterations=None, log_scale=False):
    """
    Set up common axis configurations for plots.

    Parameters:
    - ax: matplotlib axis to configure
    - x_axis_label: Label for x-axis
    - label: Specific metric label
    - metric: Metric name
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
    ax.set_xlabel(x_axis_label, fontsize=16)
    ax.set_ylabel(metric_label, fontsize=16)
    if p is not None and num_iterations is not None:
        ax.set_title(f"{metric} vs. {x_axis_label} (p={p}, {num_iterations} iterations)", fontsize=22)
    ax.set_title(f"{metric} vs. {x_axis_label}", fontsize=22)
    ax.grid(True, alpha=0.3)


def add_average_trend_line(ax, all_x, all_y, log_scale=False):
    """
    Add an average trend line to the plot.

    Parameters:
    - ax: matplotlib axis to add trend line to
    - all_x: List of x values
    - all_y: List of corresponding y values
    - log_scale: Whether to use log scale (default False)

    Returns:
    - None (modifies the axis in-place)
    """
    if len(all_x) > 1:
        # Sort by x for proper line plotting
        sorted_indices = np.argsort(all_x)
        sorted_x = np.array(all_x)[sorted_indices]
        sorted_y = np.array(all_y)[sorted_indices]

        # Fit a polynomial (degree = min(3, len(unique x) - 2))
        degree = min(2, len(set(all_x)) - 1) if len(set(all_x)) - 1 > 0 else 1
        if degree > 0:  # Need at least 2 points for a line
            if log_scale:
                # Avoid negative values for log scale
                sorted_x = [x if x > 0 else 1 for x in sorted_x]
                trend = np.polyfit(np.log(sorted_x), sorted_y, degree)
                x_for_trend = np.logspace(np.log(min(sorted_x)), np.log(max(sorted_x)), 100)
            else:
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
    def check_bound(bound):
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
    if check_bound(lower_bound):
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
    if check_bound(upper_bound):
        line = ax.vlines(upper_bound, y_min, y_max, colors='slategray',
                         linestyles='dashed', label='Upper Bound', alpha=0.5)

        # Add text label just below the x-axis
        ax.text(upper_bound, y_min - 0.05 * (y_max - y_min),
                f'Upper Bound\n({upper_bound})',
                horizontalalignment='center',
                verticalalignment='top',
                color='slategray',
                fontsize=8)

def create_figure():
    """
    Create a figure with a 2x3 grid of subplots, but only use 5 of them.
    We'll hide the unused 6th subplot.
    """
    fig, axes_2d = plt.subplots(2, 3, figsize=(24, 15))

    # Flatten into a 1D array for easy indexing
    axes = axes_2d.ravel()

    # Hide the 6th axis (the one we won't use)
    axes[4].set_visible(False)

    # return fig, axes[:3]+axes[5]
    return fig, [axes[0], axes[1], axes[2], axes[3], axes[5]]