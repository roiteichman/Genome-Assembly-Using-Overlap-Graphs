import numpy as np
import os
from generateErrorFreeReads import read_genome_from_fasta
from testAssembly import test_assembly
from plots import plot_experiment_results_by_p_values, plot_const_coverage_results, plot_coverage_comparison
from createAndSave import save_results, create_paths, load_coverage_results_from_csv
from collections import defaultdict
from consts import get_lower_bound_l, get_upper_bound_l, get_lower_bound_n, get_upper_bound_n, get_lower_bound_p, get_upper_bound_p
from joblib import Parallel, delayed


# Constants
lower_bound_l = get_lower_bound_l()
upper_bound_l = get_upper_bound_l()
lower_bound_n = get_lower_bound_n()
upper_bound_n = get_upper_bound_n()
lower_bound_p = get_lower_bound_p()
upper_bound_p = get_upper_bound_p()


def run_experiments(file_path="sequence.fasta", path_to_save_csvs="results", path_to_save_plots="plots",
                    path_to_logs="logs.txt"):
    """
    Main function to run all experimentation scenarios.

    Parameters:
        file_path (str): Path to the FASTA file containing the reference genome.
        path_to_save_csvs (str): Path to save the CSV result files.
        path_to_save_plots (str): Path to save the plots.
        path_to_logs (str): Path to save the log file.

    Returns:
        None (Saves results to files and generates plots)
    """

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""                           PREPARATIONS FOR THE EXPERIMENTS                           """""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Read the reference genome from the FASTA file
    genome = read_genome_from_fasta(file_path)
    genome_length = len(genome)

    c_smaller_than_1 = round(((lower_bound_n * lower_bound_l) / genome_length), 3)

    total_coverage_targets = [c_smaller_than_1, 2, 5, 10, 30]
    # TODO - decide wheather upper_bound_n or 10,000 as biggest num + k of k-mers + num_iterations
    n_values = np.unique(np.logspace(np.log10(lower_bound_n), np.log10(upper_bound_n), 5).astype(int))
    l_values = np.unique(np.linspace(lower_bound_l, upper_bound_l, 3).astype(int))
    error_probs = np.unique(np.logspace(np.log10(get_lower_bound_p()), np.log10(get_upper_bound_p()), 3))
    paths_comparison = []

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""                       FIRST EXPERIMENT - VARYING COVERAGE TARGETS (C)                """""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # For fixed C we will find N and l that keep the number of reads and read length constant
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    print("Experiment #1 started!")

    all_coverage_results = {}
    for C in total_coverage_targets:
        # Create paths for saving CSV files and plots
        paths_c = create_paths([(path_to_save_csvs, f"experiment_const_coverage/C_{C}"),
                                (path_to_save_plots, f"experiment_const_coverage/C_{C}")])
        paths_comparison += create_paths([(path_to_save_plots, f"experiment_const_coverage/comparison")])

        all_coverage_results[C] = experiment_const_coverage(genome, C, error_probs, l_values=l_values, x_axis_var="l",
                                                            experiment_name=f"experiment_const_coverage_{C}",
                                                            paths=paths_c, return_results=True)

    print("Experiment #1 completed!")

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""       SECOND EXPERIMENT - VARYING READS LENGTH FOR FIXED N AND FOR ALL C VALUES      """""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    print("Experiment #2 started!")

    # Fixed N
    result_vary_l = defaultdict(list)
    for n in n_values:

        paths_l = create_paths([(path_to_save_csvs, f"experiment_varying_l_fixed_n_{n}"),
                                (path_to_save_plots, f"experiment_varying_l_fixed_n_{n}")])
        paths_comparison.insert(1, create_paths([(path_to_save_plots, f"experiment_varying_l_fixed_n_{n}/comparison")])[0])

        median_l = int(l_values[len(l_values) // 2])

        result_vary_l[n].append(experiment_varying_value(genome, [n], l_values, error_probs,
                                                         expected_coverage=total_coverage_targets,
                                                         experiment_name=f"experiment_varying_l_fixed_n_{n}",
                                                         paths=paths_l, return_results=True, separator=median_l))

    print("Experiment #2 completed!")

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""      THIRD EXPERIMENT - VARYING NUMBER OF READS FOR FIXED L AND FOR ALL C VALUES     """""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    print("Experiment #3 started!") #TODO - log_scale=True

    # Fixed l
    result_vary_n = defaultdict(list)
    for l in l_values:

        paths_n = create_paths([(path_to_save_csvs, f"experiment_varying_n_fixed_l_{l}"),
                                (path_to_save_plots, f"experiment_varying_n_fixed_l_{l}")])
        paths_comparison.insert(2, create_paths([(path_to_save_plots, f"experiment_varying_n_fixed_l_{l}/comparison")])[0])

        median_n = int(n_values[len(n_values) // 2])

        result_vary_n[l].append(experiment_varying_value(genome, n_values, [l], error_probs,
                                                         expected_coverage=total_coverage_targets,
                                                         experiment_name=f"experiment_varying_n_fixed_l_{l}",
                                                         paths=paths_n, return_results=True, log_scale=True, 
                                                         separator=median_n))

    print("Experiment #3 completed!")

    # TODO - keep the option to do a combined graph from results_vary_l and results_vary_n
    plot_coverage_comparison(all_coverage_results, path=paths_comparison[0]) #TODO - Adjust
    #plot_coverage_comparison(result_vary_l, path=paths_comparison[1])
    #plot_coverage_comparison(result_vary_n, path=paths_comparison[2])

    print("All experiments completed!")


def experiment_const_coverage(reference_genome, coverage_target, error_probs, n_values=None, l_values=None,
                              x_axis_var="n", experiment_name="const_coverage", paths=None, num_iterations=10,
                              log_scale=False, return_results=False):
    """
    Experiment: Vary N and l while keeping coverage constant.
    Either n_values or l_values should be provided, and the other will be calculated.

    Parameters:
        reference_genome (str): Reference genome sequence.
        coverage_target (float): Target coverage to maintain (e.g., 10X).
        error_probs (np.array): List of p values to test.
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
    print(f"expected_coverage: {coverage_target}\n")

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
                'experiment_name': experiment_name,
                'num_iteration': num_iterations,
                'contigs': None
            })

    # Run simulations
    results = run_simulations_num_iteration_parallel(params, num_iterations, path=paths[1])

    # Save results
    os.makedirs(paths[0], exist_ok=True)
    save_results(results, experiment_name, path=paths[0])

    # Plot results
    plot_const_coverage_results(results, coverage_target=coverage_target, x_axis_var=x_axis_var, path=paths[1],
                                log_scale=log_scale)

    if return_results:
        return results


def experiment_varying_value(reference_genome, n_values, l_values, p_values, expected_coverage, experiment_name, paths,
                             num_iterations=10, log_scale=False, separator=None, return_results=False):
    """
    Experiment: Vary variable values to achieve different coverage depths.

    Parameters:
        reference_genome (str): Reference genome sequence.
        n_values (np.array): List of N values to test.
        l_values (np.array): List of l values to test.
        p_values (np.array): List of p values to test.
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
    to_print = [round(c, 2) for c in expected_coverage]
    print(f"expected_coverage: {to_print}\n")

    # Create parameter combinations
    params = []  # list of dictionaries

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
    results = run_simulations_num_iteration_parallel(params, num_iterations, path=path_to_save_plots)

    # Save results
    os.makedirs(path_to_save_csvs, exist_ok=True)
    save_results(results, experiment_name, path=path_to_save_csvs)

    # Plot results
    if len(n_values) > 1:
        plot_experiment_results_by_p_values(results, x_key="num_reads",
                                            coverage_key="expected_coverage", path=path_to_save_plots,
                                            num_iterations=num_iterations, log_scale=log_scale, separator=separator)
    elif len(l_values) > 1:
        plot_experiment_results_by_p_values(results, x_key="read_length",
                                            coverage_key="expected_coverage", path=path_to_save_plots,
                                            num_iterations=num_iterations, log_scale=log_scale, separator=separator)
    if return_results:
        return results


def run_simulations_num_iteration(params_list, num_iterations=10, path="plots"):
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

        all_iteration_results_error_prone = []

        for i in range(num_iterations):
            results = run_simulations([params], num_iteration=i + 1, path=experiment_folder)  # first iteration is 1
            all_iteration_results_error_prone.append(results[0])  # Get first dict from the list

        # Extract **only numeric keys** for averaging
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

        # Rename keys for clarity
        formatted_results = {
            **params,
            **{f"{key} avg": avg_results_error_prone[key] for key in avg_results_error_prone},
            **{f"{key} std": std_results_error_prone[key] for key in std_results_error_prone},
            **{f"{key} raw": [r[key] for r in all_iteration_results_error_prone] for key in numeric_keys},
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
        list: List of result dictionaries
    """
    results_error_prone = []

    for params in params_list:
        contigs, measures = test_assembly(params['reference_genome'], params['read_length'], params['num_reads'],
                                          params['error_prob'], params['experiment_name'], num_iteration, path)

        # Add parameters to results
        params['contigs'] = contigs
        result_error_prone = {**params, **measures}
        results_error_prone.append(result_error_prone)

    return results_error_prone


def run_simulations_num_iteration_parallel(params_list, num_iterations=10, path="plots"):
    """
    Run simulations for each parameter combination in parallel using joblib.
    Each parameter set will be processed independently by a different core.

    Parameters:
        params_list (list): List of parameter dictionaries.
        num_iterations (int): Number of iterations to run for each parameter combination.
        path (str): Path to save the plots.

    Returns:
        list: List of result dictionaries with average and std for numeric keys.
    """
    def run_for_params(params):
        print(
            f"Running {params['experiment_name']} simulation with N={params['num_reads']}, "
            f"l={params['read_length']}, p={params['error_prob']}, "
            f"expected coverage={params['expected_coverage']:.2f}x"
        )
        # Create folders for the experiment
        experiment_folder = f"{path}/test_assembly/N={params['num_reads']}_l={params['read_length']}_p={params['error_prob']}"
        os.makedirs(experiment_folder, exist_ok=True)

        all_iteration_results_error_prone = []
        for i in range(num_iterations):
            # run_simulations is assumed to return a list of result dictionaries;
            # we use the first one since we expect one result per simulation.
            results = run_simulations([params], num_iteration=i + 1, path=experiment_folder)
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
    todos_handled = False  # change to true when todos are handled.
    assert todos_handled, "Handle TODOs - plot for iterations - to plot them or not"
    run_experiments("sequence.fasta")