import numpy as np
import os
from generateErrorFreeReads import read_genome_from_fasta
from testAssembly import test_assembly
from plots import plot_experiment_results_by_p_values, plot_const_coverage_results, plot_coverage_comparison
from createAndSave import save_results, create_paths, load_coverage_results_from_csv
from collections import defaultdict
import datetime

# Constants
lower_bound_l = 50
upper_bound_l = 150
lower_bound_n = 100
upper_bound_n = 1000000


def current_time():
    """return the current time as a string"""

    # Get the current time
    now = datetime.datetime.now()

    # Format the time as a string (e.g., "2023-10-27 10:30:45")
    time_string = now.strftime("%Y-%m-%d %H:%M:%S")

    return time_string


def run_experiments(file_path="sequence.fasta", path_to_save_csvs = "results", path_to_save_plots = "plots",
                    path_to_logs = "logs.txt"):
    """
    Main function to run all experimentation scenarios.

    Parameters:
        file_path (str): Path to the FASTA file containing the reference genome.
        path_to_save_csvs (str): Path to save the CSV result files.
        path_to_save_plots (str): Path to save the plots.

    Returns:
        None (Saves results to files and generates plots)
    """

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""                           PREPARATIONS FOR THE EXPERIMENTS                           """""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    with open(path_to_logs, "w") as f:
        f.write(f"{current_time()} - Starting the experiments...\n")
        # Read the reference genome from the FASTA file
        genome = read_genome_from_fasta(file_path)
        genome_length = len(genome)
        error_probs = [0.001, 0.01, 0.05, 0.1]
        small_l = 50
        small_n = 100

        C_smaller_than_1 = ((small_l * small_n) / (genome_length))
        C_smaller_than_1 = round(C_smaller_than_1, 3)

        small_coverage_targets = [C_smaller_than_1, 3, 5, 7, 10]
        large_coverage_targets = [15, 20, 30, 50]
        total_coverage_targets = sorted(list(set(small_coverage_targets + large_coverage_targets)))
        f.write(f"{current_time()} - total_coverage_targets: {total_coverage_targets}\n")
        l_values = list(np.unique(np.linspace(lower_bound_l, upper_bound_l, 9).astype(int)))
        f.write(f"{current_time()} - l_values: {l_values}\n")
        n_value = np.unique(np.logspace(np.log10(lower_bound_n), np.log10(upper_bound_n), 9).astype(int))
        f.write(f"{current_time()} - n_value: {n_value}\n")
        paths_comparison = []

        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """""                       FIRST EXPERIMENT - VARYING COVERAGE TARGETS (C)                """""
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # For fixed C we will find N and l that keep the number of reads and read length constant
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        print("Experiment #1 started!")
        f.write(f"{current_time()} - Experiment #1 started!\n")

        all_coverage_results = {}
        for C in total_coverage_targets:
            # Create paths for saving CSV files and plots
            paths_c = create_paths([(path_to_save_csvs, f"experiment_const_coverage/C_{C}"),
                                    (path_to_save_plots, f"experiment_const_coverage/C_{C}")])
            paths_comparison += create_paths([(path_to_save_plots, f"experiment_const_coverage/comparison")])


            all_coverage_results[C] = experiment_const_coverage(genome, C, error_probs, l_values=l_values, x_axis_var="l",
                                                                experiment_name=f"experiment_const_coverage_{C}",
                                                                paths=paths_c, return_results=True)

        plot_coverage_comparison(all_coverage_results, path=paths_comparison[0])

        print("Experiment #1 completed!")
        f.write(f"{current_time()} - Experiment #1 completed!\n")

        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """""       SECOND EXPERIMENT - VARYING READS LENGTH FOR FIXED N AND FOR ALL C VALUES      """""
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        print(f"{current_time()} - Experiment #2 started!")

        # Fixed N
        result_vary_l = defaultdict(list)
        for n in n_value:

            paths_l = create_paths([(path_to_save_csvs, f"experiment_varying_l_fixed_n_{n}"),
                                    (path_to_save_plots, f"experiment_varying_l_fixed_n_{n}")])
            paths_comparison.insert(1, create_paths([(path_to_save_plots, f"experiment_varying_l_fixed_n_{n}/comparison")])[0])

            median_l = int(l_values[len(l_values) // 2])

            result_vary_l[n].append(experiment_varying_value(genome, [n], l_values, error_probs,
                                                             expected_coverage=total_coverage_targets,
                                                             experiment_name=f"experiment_varying_l_fixed_n_{n}",
                                                             paths=paths_l, return_results=True, separator=median_l))

        plot_coverage_comparison(result_vary_l, path=paths_comparison[1])
        
        print("Experiment #2 completed!")
        f.write(f"{current_time()} - Experiment #2 completed!\n")

        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """""      THIRD EXPERIMENT - VARYING NUMBER OF READS FOR FIXED L AND FOR ALL C VALUES     """""
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        print("Experiment #3 started!")
        f.write(f"{current_time()} - Experiment #3 started!\n")

        # Fixed l
        result_vary_n = defaultdict(list)
        for l in l_values:

            paths_n = create_paths([(path_to_save_csvs, f"experiment_varying_n_fixed_l_{l}"),
                                    (path_to_save_plots, f"experiment_varying_n_fixed_l_{l}")])
            paths_comparison.insert(2, create_paths([(path_to_save_plots, f"experiment_varying_n_fixed_l_{l}/comparison")])[0])

            median_n = int(n_value[len(n_value) // 2])

            result_vary_n[l].append(experiment_varying_value(genome, n_value, [l], error_probs,
                                                             expected_coverage=total_coverage_targets,
                                                             experiment_name=f"experiment_varying_n_fixed_l_{l}",
                                                             paths=paths_n, return_results=True, separator=median_n))

        plot_coverage_comparison(result_vary_n, path=paths_comparison[2])
        
        print("Experiment #3 completed!")
        f.write(f"{current_time()} - Experiment #3 completed!\n")

        # TODO - keep the option to do a combined graph from results_vary_l and results_vary_n

        print("All experiments completed!")
        f.write(f"{current_time()} - All experiments completed!\n")


def experiment_const_coverage(reference_genome, coverage_target, error_probs, n_values=None, l_values=None,
                              x_axis_var="n", experiment_name="const_coverage", paths=None, num_iterations=5,
                              log_scale=False, return_results=False):
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
                'experiment_name': experiment_name
            })

    # Run simulations
    results = run_simulations_num_iteration(params, num_iterations, path=paths[1])

    # Save results
    os.makedirs(paths[0], exist_ok=True)
    save_results(results, experiment_name, path=paths[0])

    # Plot results
    plot_const_coverage_results(results, coverage_target=coverage_target, x_axis_var=x_axis_var, path=paths[1],
                                log_scale=log_scale)

    if return_results:
        return results


def experiment_varying_value(reference_genome, n_values, l_values, p_values, expected_coverage, experiment_name, paths,
                             num_iterations=5, log_scale=False, separator=None, return_results=False):
    """
    Experiment: Vary variable values to achieve different coverage depths.

    Parameters:
        reference_genome (str): Reference genome sequence.
        n_values (list or ndarray): List of N values to test.
        l_values (list or ndarray): List of l values to test.
        p_values (list): List of p values to test.
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
    results = run_simulations_num_iteration(params, num_iterations, path=path_to_save_plots)

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

        all_iteration_results_error_prone = []

        for i in range(num_iterations):
            results_ep = run_simulations([params], num_iteration=i + 1, path=experiment_folder)  # first iteration is 1
            all_iteration_results_error_prone.append(results_ep[0])  # Get first dict from the list

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
        measures = test_assembly(params['reference_genome'], params['read_length'], params['num_reads'],
                                 params['error_prob'], params['experiment_name'], num_iteration, path)

        # Add parameters to results
        result_error_prone = {**params, **measures}
        results_error_prone.append(result_error_prone)

    return results_error_prone


def prepare_experiment_values(genome_length, coverage_target, fixed_value, varying_param='N', min_value=50,
                              max_value=150):
    """
    Prepare lists of N and l values for experiments based on coverage targets.

    Parameters:
    genome_length (int): Length of the genome.
    coverage_target (int): coverage values target.
    fixed_value (int): Fixed value for the parameter that is not varying.
    varying_param (str): Parameter to vary ('N' or 'l').
    min_value (int): Minimum value for the varying parameter.
    max_value (int): Maximum value for the varying parameter.

    Returns:
    list: List of values for the varying parameter.
    """
    values = []
    if varying_param == 'N':
        N = int(np.ceil(coverage_target * genome_length / fixed_value))
        if min_value <= N <= max_value:
            values.append(N)
        elif N > max_value:
            values.append("N>max")
        else:
            values.append("N<min")
    elif varying_param == 'l':
        l = int(np.ceil(coverage_target * genome_length / fixed_value))
        if min_value <= l <= max_value:
            values.append(l)
        elif l > max_value:
            values.append("l>max")
        else:
            values.append("l<min")

    return values


def find_values(genome_length, coverage_targets, fixed_value, lower_bound, upper_bound, varying_param='N', num_values=5):
    """
    The function finds the upper and lower bound for the varying parameter and its corresponding values.

    Parameters:
    genome_length (int): Length of the genome.
    coverage_targets (list): List of target coverage values.
    fixed_value (int): Fixed value for the parameter that is not varying.
    lower_bound (int): Lower bound for the varying parameter.
    upper_bound (int): Upper bound for the varying parameter.
    varying_param (str): Parameter to vary ('N' or 'l).
    num_values (int): Number of values to generate.

    Returns:
    dict: Dictionary with new_lower_bound and new_upper_bound as keys and list of values for the varying parameter as value.
    """
    best_lower_bound = 0
    best_upper_bound = genome_length
    best_values = None

    for new_lower_bound in range(lower_bound, -1, -1):

        # in case where the genome_length shorter than the upper_bound (varying_param=='l')
        maximum_value = upper_bound if upper_bound > genome_length else genome_length

        for new_upper_bound in range(upper_bound, maximum_value + 1):
            values = prepare_experiment_values(genome_length, coverage_targets, fixed_value,
                                               varying_param=varying_param, min_value=new_lower_bound,
                                               max_value=new_upper_bound, num_values=num_values)
            print(f"coverage_targets: {coverage_targets}")
            print(f"values: {values}")
            numeric_values = [v for v in values if isinstance(v, int)]
            print(f"numeric_values: {numeric_values}")

            if len(numeric_values)==num_values:
                # the best value it those that thier upper_bound and lower_bound vary as low as possible from upper_bound and lower_bound respectively
                return {(new_lower_bound, new_upper_bound): numeric_values}

    return {(best_lower_bound, best_upper_bound): best_values}


# Main function to run all experiments
if __name__ == "__main__":
    run_experiments("sequence.fasta")
