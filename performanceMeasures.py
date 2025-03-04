import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import defaultdict
from aligners import local_alignment, align_read_or_contig_to_reference
import numpy as np
import os

def calculate_genome_coverage(contigs, read_length, reference_genome):
    """
    Compute the percentage of genome bases covered by at least one contig.

    Parameters:
        contigs (list): List of assembled contigs.
        read_length (int): Length of each read.
        reference_genome (str): The reference genome.


    Returns:
        float: Genome coverage percentage.
    """
    print("++ calculate the reference_genome coverage ++")
    covered_bases = set()

    for contig in contigs:
        _, _, start, end = align_read_or_contig_to_reference(contig, reference_genome, read_length)

        if start != -1 and end != -1:
            # update teh covered_bases with the range of the contig
            covered_bases.update(list(range(start, end)))  # Mark covered bases

    return len(covered_bases) / len(reference_genome)


def calculate_mismatch_rate(contigs, read_length, reference_genome, match_score=10):
    """
    Compute mismatch rate between contigs and genome.
    The mismatch rate is the fraction of bases in contigs that mismatch the reference genome.
    if closer to 0 - the contigs are identical to the reference genome.
    if closer to 1 - the contigs are completely different from the reference genome.

    Parameters:
        contigs (list): List of assembled contigs.
        read_length (int): Length of each read.
        reference_genome (str): The reference genome sequence.
        match_score (int): Score for a matching base.

    Returns:
        float: Mismatch rate.
    """
    print("++ calculate the mismatch rate ++")
    total_score = 0
    optimal_score = 0

    for contig in contigs:
        print(f"align contig: {contig} with reference_genome: {reference_genome}")
        alignment, score, start, end = align_read_or_contig_to_reference(contig, reference_genome, read_length)
        print(f"alignment: {alignment}")
        print(f"score: {score}")
        print(f"start: {start}")
        print(f"end: {end}")

        if start != -1 and end != -1:
            total_score += score
            optimal_score += match_score * (end - start)

    return (optimal_score - total_score) / optimal_score if optimal_score > 0 else 0


def calculate_mismatch_rate2(contigs, read_length, reference_genome, match_score=10):
    """
    Compute mismatch rate between contigs and genome.

    The mismatch rate is the fraction of bases in contigs that mismatch the reference genome.

    - If close to 0: The contigs are nearly identical to the reference genome.
    - If close to 1: The contigs are highly different from the reference genome.
    - Also considers uncovered bases in the genome as mismatches.

    Parameters:
        contigs (list): List of assembled contigs.
        read_length (int): Length of each read.
        reference_genome (str): The reference genome sequence.
        match_score (int): Score for a matching base.

    Returns:
        float: Mismatch rate.
    """

    print("++ Calculating mismatch rate ++")

    total_score = 0
    # optimal score is the score if all contigs were perfectly aligned
    optimal_score = len(reference_genome) * match_score
    covered_bases = set()

    for contig in contigs:
        alignment, score, start, end = align_read_or_contig_to_reference(contig, reference_genome, read_length)

        if start != -1 and end != -1:
            total_score += score
            covered_bases.update(list(range(start, end)))  # Track covered genome positions

    return (optimal_score - total_score) / optimal_score if optimal_score > 0 else 0


def calculate_mismatch_rate3(contigs, read_length, reference_genome, match_score=10):
    """
    Compute mismatch rate between contigs and genome.

    The mismatch rate is the fraction of bases in contigs that mismatch the reference genome.

    - If close to 0: The contigs are nearly identical to the reference genome.
    - If close to 1: The contigs are highly different from the reference genome.
    - Also considers uncovered bases in the genome as mismatches.
    - Ignores redundant incorrect contigs if the genome is already fully reconstructed.

    Parameters:
        contigs (list): List of assembled contigs.
        read_length (int): Length of each read.
        reference_genome (str): The reference genome sequence.
        match_score (int): Score for a matching base.

    Returns:
        float: Mismatch rate.
    """

    print("++ Calculating mismatch rate ++")

    total_score = 0
    optimal_score = len(reference_genome) * match_score
    covered_bases = set()

    # Track the best score for each genome base
    best_coverage = {i: -float("inf") for i in range(len(reference_genome))}

    for contig in contigs:
        alignment, score, start, end = align_read_or_contig_to_reference(contig, reference_genome, read_length)

        if start != -1 and end != -1:
            total_score += score
            # Update the best coverage scores for each base
            for i in range(start, end):
                best_coverage[i] = max(best_coverage[i], score)

            covered_bases.update(list(range(start, end)))  # Track covered genome positions

    # Check if the genome is perfectly reconstructed
    if len(covered_bases) == len(reference_genome) and all(v >= match_score*len(reference_genome)
                                                           for v in best_coverage.values()):
        return 0.0  # The genome is fully reconstructed as one contig, no mismatches.

    # Ensure mismatch rate is never negative
    mismatch_rate = max((optimal_score - total_score) / optimal_score, 0.0)

    return mismatch_rate


def calculate_n50(contigs):
    """
    Compute the N50 value for a list of contigs.

    Parameters:
        contigs (list): List of assembled contigs.

    Returns:
        int: N50 value.
    """
    contig_lengths = [len(contig) for contig in contigs]
    contig_lengths.sort(reverse=True)
    cumulative_length = 0
    n50 = 0
    for length in contig_lengths:
        cumulative_length += length
        if cumulative_length >= sum(contig_lengths) / 2:
            n50 = length
            break
    return n50


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
    plt.savefig(f"{path}/{error_type_str}_genome_coverage_iteration_{str(num_iteration)}.png")
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

    plt.savefig(f"{path}/{error_type_str}_genome_depth_iteration_{str(num_iteration)}.png")
    plt.close()


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
        plt.savefig(f"{path}/{error_type_str}_contig_coverage_{contig_idx + 1}_iteration_{str(num_iteration)}.png")
        plt.close()


def calculate_essential_performance_measures(contigs, reads, num_reads, reads_length, error_prob, reference_genome,
                                             error_type_str, experiment_name, num_iteration, path="plots"):
    """
    Computes essential assembly quality metrics.

    Parameters:
        contigs (list): List of assembled contigs.
        reads (list): List of sequencing reads.
        num_reads (int): Number of reads used for assembly.
        reads_length (int): Length of each read.
        error_prob (float): Probability of mutation in error-prone reads.
        reference_genome (str): The reference genome sequence.
        error_type_str (str): Type of reads used for assembly (e.g., "error-free" or "error-prone").
        experiment_name (str): The name of the experiment.
        num_iteration (int): The number of the specific iteration.
    Returns:
        dict: Dictionary of performance metrics.

    Metrics:
    - Number of Contigs: Number of contigs assembled.
    - Genome Coverage: Percentage of genome covered by the contigs.
    - N50: Shortest contig length at 50% total contig length.
    - Mismatch Rate: Fraction of bases in contigs that mismatch the reference genome.
    - Reconstructed Genome Coverage: Plot of read coverage depth for each base in the assembled contigs.
    """
    #experiment_folder = f"{path}/test_assembly/"
    #os.makedirs(experiment_folder, exist_ok=True)
    # experiments_folder = f"{path}/test_assembly"
    # os.makedirs(experiments_folder, exist_ok=True)

    return {
        "Number of Contigs": len(contigs),
        "Genome Coverage": calculate_genome_coverage(contigs, reads_length, reference_genome),
        "N50": calculate_n50(contigs),
        "Mismatch Rate": calculate_mismatch_rate3(contigs, reads_length, reference_genome),
        # Wanted lower as posible, in average lower than p
        "Genome Coverage Plot": plot_genome_coverage(contigs, num_reads, reads_length, error_prob, reference_genome,
                                                     error_type_str, experiment_name, num_iteration, path),
        # Displays a plot
        "Genome Depth Plot": plot_genome_depth(reads, reference_genome, reads_length, error_prob, error_type_str,
                                               experiment_name, num_iteration, path),
        # Displays a plot
        "Reconstructed Genome Coverage": plot_reconstructed_coverage(contigs, reads, num_reads, reads_length, error_prob,
                                                                     reference_genome, error_type_str, experiment_name,
                                                                     num_iteration, path)
        # Displays a plot
        # Contig Coverage - I expect that it will be similar to the one we started with (e.g. 10X and not 20X)
    }


if __name__ == "__main__":
    toy_genome = "ATGCGTACGTTAGC"
    toy_contig = ['ATGCGTACGTTAGC','CCTTA']

    mismatch_rate = calculate_mismatch_rate2(toy_contig, 5, toy_genome)
    print(f"genome: {toy_genome}")
    print(f"contigs: {toy_contig}")
    print(f"Mismatch rate2: {mismatch_rate}")
    mismatch_rate = calculate_mismatch_rate3(toy_contig, 5, toy_genome)
    print(f"Mismatch rate3: {mismatch_rate}")
