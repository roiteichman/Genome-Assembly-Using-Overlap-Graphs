import numpy as np
import os
from generateErrorFreeReads import read_genome_from_fasta
from testAssembly import test_assembly
from plots import plot_experiment_results_by_other_values, plot_const_coverage_results, plot_coverage_comparison, plot_experiment_results_by_two_values
from createAndSave import save_results, create_paths, load_coverage_results_from_csv, load_and_clean_results, load_and_combine_results
from collections import defaultdict
from consts import get_lower_bound_l, get_upper_bound_l, get_lower_bound_n, get_upper_bound_n, get_big_n, get_lower_bound_p, get_upper_bound_p
from joblib import Parallel, delayed


# Constants
lower_bound_l = get_lower_bound_l()
upper_bound_l = get_upper_bound_l()
lower_bound_n = get_lower_bound_n()
upper_bound_n = get_upper_bound_n()
lower_bound_p = get_lower_bound_p()
upper_bound_p = get_upper_bound_p()
big_n = get_big_n()


def run_experiments(file_path="sequence.fasta", path_to_save_csvs="results_k2", path_to_save_plots="plots_k2",
                    skip_1=False, skip_2=False, skip_3=False, data_replace_experiment=None):
    """
    Main function to run all experimentation scenarios.

    Parameters:
        file_path (str): Path to the FASTA file containing the reference genome.
        path_to_save_csvs (str): Path to save the CSV result files.
        path_to_save_plots (str): Path to save the plots.
        skip_1 (bool): Whether to skip the first experiment.
        skip_2 (bool): Whether to skip the second experiment.
        skip_3 (bool): Whether to skip the third experiment.
        data_replace_experiment (int): if not None, it will replace the experiment with the given index with the data

    Returns:
        None (Saves results to files and generates plots)
    """

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""                           PREPARATIONS FOR THE EXPERIMENTS                           """""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Read the reference genome from the FASTA file
    genome = read_genome_from_fasta(file_path) #"ATGCGTACGTTAGC"
    genome_length = len(genome)

    c_smaller_than_1 = round(((lower_bound_n * lower_bound_l) / genome_length), 3) #10/14

    total_coverage_targets = [c_smaller_than_1, 2, 5, 10, 30] #[2, 5] # TODO - return to all values
    n_values = np.unique(np.logspace(np.log10(lower_bound_n), np.log10(big_n), 5).astype(int)) # np.array([2, 5])
    l_values = np.unique(np.linspace(lower_bound_l, upper_bound_l, 3).astype(int)) #np.array([5, 10])
    error_probs = np.unique(np.logspace(np.log10(get_lower_bound_p()), np.log10(get_upper_bound_p()), 3))
    k_values = np.unique(np.array([2,5,10,15])) #np.array([0, 1]) TODO - I give up on k=0, just if there will be enough time
    paths_comparison_fixed_k = []
    paths_comparison_fixed_p = []

    # if skip_1 is True, then its 0, else it will not be used so nevermind the value
    exp_1_idx = 0
    # if skip_1 is True, then if skip_2 is True, its 0, else it will not be used so nevermind the value
    exp_2_idx = 1 if skip_1 is False else 0
    # if skip_1 is True, then if skip_2 is True, its 1, else it will not be used so nevermind the value
    exp_3_other_option_idx = 1 if skip_1 is False else 0
    exp_3_idx = 2 if (skip_1 is False and skip_2 is False) else exp_3_other_option_idx

    path_to_loaded_data = None
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""                       FIRST EXPERIMENT - VARYING COVERAGE TARGETS (C)                """""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # For fixed C we will find N and l that keep the number of reads and read length constant
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    all_coverage_results_fixed_k = {}
    all_coverage_results_fixed_p = {}

    if not skip_1 or data_replace_experiment == 1:
        print("Experiment #1 started!")

        for C in total_coverage_targets:
            # Create paths for saving CSV files and plots
            experiment_name = f"experiment_const_coverage/C_{C}"
            paths_c = create_paths([(path_to_save_csvs, experiment_name), (path_to_save_plots, experiment_name)])
            prefix_comparison = f"experiment_const_coverage/comparison"
            paths_comparison_fixed_k.insert(exp_1_idx, create_paths([(path_to_save_plots, f"{prefix_comparison}/fixed_k")])[0])
            paths_comparison_fixed_p.insert(exp_1_idx, create_paths([(path_to_save_plots, f"{prefix_comparison}/fixed_p")])[0])

            results = []
            if C in total_coverage_targets:
                results = experiment_const_coverage(genome, C, error_probs, k_values,
                                                    l_values=l_values,
                                                    x_axis_var="l",
                                                    experiment_name=experiment_name,
                                                    paths=paths_c, return_results=True)
            else:
                results = load_and_clean_results(f"results/experiment_const_coverage/C_{C}")
                print(f"Load results for C={C}")
            res_fixed_k = filter_results(results, 'k', k_values)
            all_coverage_results_fixed_k[C] = res_fixed_k
            res_fixed_p = filter_results(results, 'error_prob', error_probs)
            all_coverage_results_fixed_p[C] = res_fixed_p


        print("Experiment #1 completed!")

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""       SECOND EXPERIMENT - VARYING READS LENGTH FOR FIXED N AND FOR ALL C VALUES      """""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    results_vary_l_fixed_k = {}
    results_vary_l_fixed_p = {}

    if not skip_2 or data_replace_experiment == 2:
        print("Experiment #2 started!")

        # Fixed N
        for n in n_values:
            suffix = f"experiment_varying_l/fixed_n_{n}"
            paths_vary_l = create_paths([(path_to_save_csvs, suffix), (path_to_save_plots, suffix)])

            prefix_comparison = f"experiment_varying_l/comparison"
            paths_comparison_fixed_k.insert(exp_2_idx, create_paths([(path_to_save_plots, f"{prefix_comparison}/fixed_k")])[0])
            paths_comparison_fixed_p.insert(exp_2_idx, create_paths([(path_to_save_plots, f"{prefix_comparison}/fixed_p")])[0])

            median_l = int(l_values[len(l_values) // 2])

            if data_replace_experiment != 2:
                results = experiment_varying_value(genome, [n], l_values, error_probs, k_values,
                                                   expected_coverage=total_coverage_targets,
                                                   experiment_name=f"experiment_varying_l_fixed_n_{n}",
                                                   paths=paths_vary_l, separator=median_l, return_results=True)

                res_fixed_k = filter_results(results, 'k', k_values)
                r_fixed_k = filter_results(res_fixed_k, 'k', k_values)
                results_vary_l_fixed_k[n] = r_fixed_k
                res_fixed_p = filter_results(results, 'error_prob', error_probs)
                r_fixed_p = filter_results(res_fixed_p, 'error_prob', error_probs)
                results_vary_l_fixed_p[n] = r_fixed_p

            else:
                path_to_loaded_data = f"{path_to_save_csvs}/{suffix}"

        print("Experiment #2 completed!")

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""      THIRD EXPERIMENT - VARYING NUMBER OF READS FOR FIXED L AND FOR ALL C VALUES     """""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    results_vary_n_fixed_k = {}
    results_vary_n_fixed_p = {}

    if not skip_3 or data_replace_experiment == 3:

        print("Experiment #3 started!")
        # Fixed l - log_scale=True
        for l in l_values:
            common_prefix = "experiment_varying_n"
            suffix = f"{common_prefix}/fixed_l_{l}"
            paths_vary_n = create_paths([(path_to_save_csvs, suffix), (path_to_save_plots, suffix)])

            prefix_comparison = f"{common_prefix}/comparison"
            paths_comparison_fixed_k.insert(exp_2_idx, create_paths([(path_to_save_plots, f"{prefix_comparison}/fixed_k")])[0])
            paths_comparison_fixed_p.insert(exp_2_idx, create_paths([(path_to_save_plots, f"{prefix_comparison}/fixed_p")])[0])

            median_n = int(n_values[len(n_values) // 2])

            if data_replace_experiment != 3:
                results = experiment_varying_value(genome, n_values, [l], error_probs, k_values,
                                                   expected_coverage=total_coverage_targets,
                                                   experiment_name=f"{common_prefix}_fixed_l_{l}", paths=paths_vary_n,
                                                   separator=median_n, return_results=True, log_scale=True)

                res_fixed_k = filter_results(results, 'k', k_values)
                r_fixed_k = filter_results(res_fixed_k, 'k', k_values)
                results_vary_l_fixed_k[l] = r_fixed_k
                res_fixed_p = filter_results(results, 'error_prob', error_probs)
                r_fixed_p = filter_results(res_fixed_p, 'error_prob', error_probs)
                results_vary_l_fixed_p[l] = r_fixed_p

            else:
                path_to_loaded_data = f"{path_to_save_csvs}/{suffix}"

        print("Experiment #3 completed!")

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""           OPTIONAL - LOAD DATA INSTEAD OF RUNNING THE EXPERIMENTS AGAIN              """""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Optional - Load results from CSV files
    results_fixed_k, results_fixed_p = [], []
    if data_replace_experiment is not None:

        # Load the list of dictionaries return from expriements functions
        #pattern = "C_" if data_replace_experiment == 1 else "fixed_n_" if data_replace_experiment == 2 else "fixed_l_"
        #results = load_coverage_results_from_csv(path_to_loaded_data, pattern)
        results = load_and_clean_results(path_to_loaded_data)

        print(f"\nresults:\n{results}\n")

        # Filter it according the group by values
        for k in k_values:
            results_fixed_k.append(filter_results(results, 'k', k))

        print(f"\nall_coverage_results_fixed_k:\n{all_coverage_results_fixed_k}\n")

        for p in error_probs:
            results_fixed_p.append(filter_results(results, 'error_prob', p))

        print(f"\nall_coverage_results_fixed_p:\n{all_coverage_results_fixed_p}\n")

    # Plot comparison of all experiments
    print("Plotting Comparison Graphs...")

    if not skip_1 or data_replace_experiment == 1:
        res_k = results_fixed_k if data_replace_experiment == 1 else all_coverage_results_fixed_k
        res_p = results_fixed_p if data_replace_experiment == 1 else all_coverage_results_fixed_p
        plot_coverage_comparison(res_k, genome_length, path=paths_comparison_fixed_k[exp_1_idx], grouping_value='error_prob')
        plot_coverage_comparison(res_p, genome_length, path=paths_comparison_fixed_p[exp_1_idx], grouping_value='k')
    if not skip_2 or data_replace_experiment == 2:
        res_k = results_vary_l_fixed_k if data_replace_experiment == 2 else all_coverage_results_fixed_k
        res_p = results_vary_l_fixed_p if data_replace_experiment == 2 else all_coverage_results_fixed_p
        plot_coverage_comparison(res_k, genome_length, path=paths_comparison_fixed_k[exp_2_idx])
        plot_coverage_comparison(res_p, genome_length, path=paths_comparison_fixed_p[exp_2_idx])
    if not skip_3 or data_replace_experiment == 3:
        res_k = results_vary_n_fixed_k if data_replace_experiment == 3 else all_coverage_results_fixed_k
        res_p = results_vary_n_fixed_p if data_replace_experiment == 3 else all_coverage_results_fixed_p
        plot_coverage_comparison(results_vary_n_fixed_k, genome_length, path=paths_comparison_fixed_k[exp_3_idx])
        plot_coverage_comparison(results_vary_n_fixed_p, genome_length, path=paths_comparison_fixed_p[exp_3_idx])

    print("All experiments completed!")


def experiment_const_coverage(reference_genome, coverage_target, error_probs, k_values, n_values=None, l_values=None,
                              x_axis_var="n", experiment_name=None, paths=None, num_iterations=10,
                              log_scale=False, return_results=False):
    """
    Experiment: Vary N and l while keeping coverage constant.
    Either n_values or l_values should be provided, and the other will be calculated.

    Parameters:
        reference_genome (str): Reference genome sequence.
        coverage_target (float): Target coverage to maintain (e.g., 10X).
        error_probs (np.array): List of p values to test.
        k_values (np.array): List of k values to test.
        n_values (np.array, optional): List of N values to test. If provided, l will be calculated.
        l_values (np.array, optional): List of l values to test. If provided, N will be calculated.
        x_axis_var (str): Variable to use on x-axis ('n' or 'l').
        experiment_name (str): Name of the experiment.
        paths (list): List of paths to save results and plots.
        num_iterations (int): Number of iterations to run for each parameter combination.
        log_scale (bool): Whether to use log scale for x-axis.
        return_results (bool): Whether to return the results.

    Returns:
        None (Saves results to files and generates plots)
    """
    if paths is None:
        paths = ["results", "plots"]

    print(f"Running Experiment with Constant Coverage: {experiment_name}")

    print(f"reference_genome: {reference_genome}")
    values = l_values if x_axis_var == "l" else n_values
    print(f"{x_axis_var}_values: {values}")
    print(f"p_values: {error_probs}")
    print(f"k_values: {k_values}")
    print(f"expected_coverage: {coverage_target}\n")

    genome_length = len(reference_genome)

    # Create parameter combinations
    params = []  # list of dictionaries

    # Calculate the dependent variable based on which one was provided
    if n_values is not None and l_values is None:
        # Calculate l values that maintain constant coverage for each n
        l_values = [int(np.ceil(coverage_target * genome_length / n)) for n in n_values]
        x_axis_var = "num_reads"  # Force x_axis_var to match provided values
    elif l_values is not None and n_values is None:
        # Calculate n values that maintain constant coverage for each l
        n_values = [int(np.ceil(coverage_target * genome_length / l)) for l in l_values]
        x_axis_var = "read_length"  # Force x_axis_var to match provided values
    else:
        raise ValueError("Either n_values or l_values must be provided, but not both")

    # Create a list of expected coverage values (should all be approximately the same)
    expected_coverage = [n * l / genome_length for n, l in zip(n_values, l_values)]

    for i, p in enumerate(error_probs):
        for m, k in enumerate(k_values):
            for j, (n, l) in enumerate(zip(n_values, l_values)):
                if k==2: # TODO just for adding k=2
                    params.append({
                        'num_reads': n,
                        'read_length': l,
                        'error_prob': p,
                        'k': k,
                        'reference_genome': reference_genome,
                        'expected_coverage': expected_coverage[j],
                        'experiment_name': experiment_name,
                        'num_iterations': num_iterations,
                        'contigs': None
                    })

    suffix = ""
    C_is_smaller_than_1 = 1>=(1-expected_coverage[0])>=0 if 1>expected_coverage[0] else 1>(expected_coverage[0]-1)
    C_2 = 1>=(2-expected_coverage[0])>=0 if 2>expected_coverage[0] else 1>(expected_coverage[0]-2)
    C_5 = 1>=(5-expected_coverage[0])>=0 if 5>expected_coverage[0] else 1>(expected_coverage[0]-5)
    C_10 = 1>=(10-expected_coverage[0])>=0 if 10>expected_coverage[0] else 1>(expected_coverage[0]-10)
    C_30 = 1>=(30-expected_coverage[0])>=0 if 30>expected_coverage[0] else 1>(expected_coverage[0]-30)

    if C_is_smaller_than_1:
        suffix = "C_0.928"
    elif C_2:
        suffix = "C_2"
    elif C_5:
        suffix = "C_5"
    elif C_10:
        suffix = "C_10"
    elif C_30:
        suffix = "C_30"

    res_all_other_k = load_and_clean_results(f"results/experiment_const_coverage/{suffix}")
    if res_all_other_k:
        print(f"load results for {suffix}")
    # TODO - put the old result in results/experiment_const_coverage/C_num where num choosen by upper if-else

    # Run simulations
    results = run_simulations_parallel(params, path=paths[1])  # Returns list of dictionaries

    results.extend(res_all_other_k)

    # Save results
    os.makedirs(paths[0], exist_ok=True)
    save_results(results, experiment_name, path=paths[0])

    group_by_str = ['fixed_p', 'fixed_k']
    folders_name = create_paths([(paths[1], f"{name}") for name in group_by_str])

    # Collect filtered results for k
    all_filtered_results_k = []
    for k in k_values:
        filtered_results_k = [r for r in results if r['k'] == k]
        all_filtered_results_k.extend(filtered_results_k)

    # Plot combined results for k
    print(f"Plotting Measures...")
    plot_const_coverage_results(all_filtered_results_k, coverage_target=coverage_target, x_axis_var=x_axis_var,
                                path=folders_name[0], log_scale=log_scale, grouping_value='k',
                                num_iterations=len(results))

    # Collect filtered results for p
    all_filtered_results_p = []
    for p in error_probs:
        filtered_results_p = [r for r in results if r['error_prob'] == p]
        all_filtered_results_p.extend(filtered_results_p)

    # Plot combined results for p
    plot_const_coverage_results(all_filtered_results_p, coverage_target=coverage_target, x_axis_var=x_axis_var,
                                path=folders_name[1], log_scale=log_scale, grouping_value='error_prob',
                                num_iterations=len(results))

    plot_experiment_results_by_two_values(results, x_key=x_axis_var, group_key_1="error_prob", group_key_2="k",
                                          coverage_key="expected_coverage", path=paths[1], log_scale=log_scale,
                                          num_iterations=num_iterations)

    if return_results:
        return results


def experiment_varying_value(reference_genome, n_values, l_values, p_values, k_values, expected_coverage,
                             experiment_name, paths, num_iterations=10, log_scale=False, separator=None,
                             return_results=False):
    """
    Experiment: Vary variable values to achieve different coverage depths.

    Parameters:
        reference_genome (str): Reference genome sequence.
        n_values (np.array): List of N values to test.
        l_values (np.array): List of l values to test.
        p_values (np.array): List of p values to test.
        k_values (np.array): List of k values to test (for k-mers).
        expected_coverage (list): List of expected coverage values corresponding to N values.
        experiment_name (str): Name of the experiment.
        paths (list): List of paths to save results and plots.
        num_iterations (int): Number of iterations to run for each parameter combination.
        log_scale (bool): Whether to use log scale for x-axis.
        separator (int): an integer to separate for different plots if they smaller or bigger than it.
        return_results (bool): Whether to return the results.

    Returns:
        None (Saves results to files and generates plots)
    """
    print(f"Running Experiment: {experiment_name}")

    print(f"reference_genome: {reference_genome}")
    print(f"n_values: {n_values}")
    print(f"l_values: {l_values}")
    print(f"p_values: {p_values}")
    print(f"k_values: {k_values}")
    to_print = [round(c, 2) for c in expected_coverage]
    print(f"expected_coverage: {to_print}\n")

    # Create parameter combinations
    params = []  # list of dictionaries

    for i, p in enumerate(p_values):
        for j, n in enumerate(n_values):
            for m, l in enumerate(l_values):
                for s, k in enumerate(k_values):
                    params.append({
                        'num_reads': n,
                        'read_length': l,
                        'error_prob': p,
                        'k': k,
                        'reference_genome': reference_genome,
                        'expected_coverage': expected_coverage[j] if len(n_values) > 1 else expected_coverage[m],
                        'experiment_name': experiment_name,
                        'num_iterations': num_iterations,
                        'contigs': None
                    })

    # Run simulations
    results = run_simulations_parallel(params, path=paths[1])

    # Save results
    os.makedirs(paths[0], exist_ok=True)
    save_results(results, experiment_name, path=paths[0])

    # Create folders for the experiment
    group_by_str = ['fixed_p', 'fixed_k']
    folders_name = create_paths([(paths[1], f"{group_name}") for group_name in group_by_str])

    # Plot results grouped by k
    x_key = "num_reads" if len(n_values) > 1 else "read_length"

    # Fixed p
    print(f"Plotting Measures for group by k with for {x_key}...")
    plot_experiment_results_by_other_values(results, x_key=x_key, coverage_key="expected_coverage",
                                            path=folders_name[0], log_scale=log_scale, num_iterations=num_iterations,
                                            separator=separator, other_value_key='k')

    # Fixed k
    print(f"Plotting Measures for group by error_prob with for {x_key}...")
    plot_experiment_results_by_other_values(results, x_key=x_key, coverage_key="expected_coverage",
                                            path=folders_name[1], log_scale=log_scale, num_iterations=num_iterations,
                                            separator=separator, other_value_key='error_prob')

    # Plot combined results
    for x_key in ["num_reads", "read_length"]:
        print(f"Plotting Measures for two values: p & k, for {x_key}...")
        plot_experiment_results_by_two_values(results, x_key=x_key, group_key_1="error_prob", group_key_2="k",
                                              coverage_key="expected_coverage", path=paths[1], log_scale=log_scale,
                                              num_iterations=num_iterations)

    if return_results:
        return results


def filter_results(results, key, values):
    """
    Filter results by a specific key-value pair.

    Parameters:
        results (list): List of result dictionaries.
        key (str): Key to filter by.
        values: Value or list of values to filter by.

    Returns:
        list: Filtered list of result dictionaries.
    """
    output = []
    if isinstance(values, (list, tuple, np.ndarray)):  # Check if values is iterable
        for value in values:
            filtered_results = [r for r in results if r[key] == value]
            output.extend(filtered_results)
    else:  # Handle single value case
        filtered_results = [r for r in results if r[key] == values]
        output.extend(filtered_results)

    return output


def run_simulations(params_list, num_iteration, path="plots"):
    """
    Run simulations with the given parameter combinations.

    Parameters:
        params_list (list): List of parameter dictionaries.
        num_iteration (int): The number of the specific iteration.
        path (str): Path to save the plots.

    Returns:
        list: List of result dictionaries
    """
    results_error_prone = []

    for params in params_list:
        contigs, measures, contigs_alignments_details, error_prone_reads = test_assembly(params['reference_genome'], params['read_length'],
                                                                                         params['num_reads'], params['error_prob'],
                                                                                         params['k'], params['experiment_name'],
                                                                                         num_iteration, path)

        # Add parameters to results - store later in csv
        params['contigs'] = contigs
        params['contigs_alignments_details'] = contigs_alignments_details
        params['error_prone_reads'] = error_prone_reads
        result_error_prone = {**params, **measures}
        results_error_prone.append(result_error_prone)

    return results_error_prone


def run_simulations_parallel(params_list, path="plots"):
    """
    Run simulations for each parameter combination in parallel using joblib.
    Each parameter set will be processed independently by a different core.

    Parameters:
        params_list (list): List of parameter dictionaries.
        path (str): Path to save the plots.

    Returns:
        list: List of result dictionaries with average and std for numeric keys.
    """
    def run_for_params(params):
        print(
            f"Running {params['experiment_name']} simulation with N={params['num_reads']}, "
            f"l={params['read_length']}, p={params['error_prob']}, k={params['k']}, "
            f"expected coverage={params['expected_coverage']:.2f}x"
        )
        # Create folders for the experiment
        experiment_name = (f"test_assembly/N={params['num_reads']}_l={params['read_length']}_"
                           f"p={params['error_prob']}_k={params['k']}")

        experiment_folder = create_paths([(path, experiment_name)])[0]

        all_iteration_results_error_prone = []

        # Run each simulation num_iterations times
        for i in range(params['num_iterations']):
            # run_simulations is assumed to return a list of result dictionaries;
            # we use the first one since we expect one result per simulation.
            results = run_simulations([params], num_iteration=i + 1, path=experiment_folder)
            #backup_folder = create_paths([(path, f"iteration_{i+1}")])[0]
            #save_results(backup_folder, experiment_name, path=path)

            all_iteration_results_error_prone.append(results[0])

        # Extract only numeric keys for averaging
        numeric_keys = [
            key for key in all_iteration_results_error_prone[0].keys()
            if isinstance(all_iteration_results_error_prone[0][key], (int, float, np.number))
        ]

        avg_results_error_prone = {
            key: np.mean([r[key] for r in all_iteration_results_error_prone]) for key in numeric_keys
        }
        std_results_error_prone = {
            key: np.std([r[key] for r in all_iteration_results_error_prone]) for key in numeric_keys
        }

        # Combine parameters with aggregated results
        formatted_results = {
            **params,
            **{f"{key} avg": avg_results_error_prone[key] for key in avg_results_error_prone},
            **{f"{key} std": std_results_error_prone[key] for key in std_results_error_prone},
            **{f"{key} raw": [r[key] for r in all_iteration_results_error_prone] for key in numeric_keys},
        }
        return formatted_results

    # Use all available cores (n_jobs=-1) to process the parameter combinations in parallel.
    results = Parallel(n_jobs=-1)(delayed(run_for_params)(params) for params in params_list)
    # TODO - decide n_jobs
    return results

if __name__ == "__main__":
    #todos_handled = False  # change to true when todos are handled.
    #assert todos_handled, "Handle TODOs Before"
    genome_fasta_file_name = "sequence.fasta"
    skip_1 , skip_2 , skip_3 = False, True, True #TODO: True-> skip over this exp, False-> make it
    data_replace_experiment = None #TODO: if necessary can be used
    run_experiments(genome_fasta_file_name, skip_1=skip_1, skip_2=skip_2, skip_3=skip_3,
                    data_replace_experiment=data_replace_experiment)
