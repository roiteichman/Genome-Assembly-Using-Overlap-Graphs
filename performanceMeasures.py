import numpy as np
from aligners import align_read_or_contig_to_reference
from plots import plot_genome_coverage, plot_genome_depth, plot_reconstructed_coverage


def calculate_genome_coverage(contigs_alignment_details, genome_length, expected_coverage, experiment_name, num_iteration, path="plots"):
    """
    Compute the percentage of genome bases covered by at least one contig.

    Parameters:
        contigs_alignment_details (dict): Dictionary of contig alignment details.
        genome_length (int): Length of the reference genome.
        expected_coverage (float): Expected coverage of the genome.
        experiment_name (str): The name of the experiment.
        num_iteration (int): The number of the specific iteration.
        path (str): Path to save the plot.

    Returns:
        float: Genome coverage percentage.
    """

    coverage = np.zeros(genome_length)

    for contig in contigs_alignment_details:
        start = contigs_alignment_details[contig]["Start Position"]
        end = contigs_alignment_details[contig]["End Position"]
        if start != -1 and end != -1:
            # update teh covered_bases with the range of the contig
            coverage[start:end] += 1

    plot_genome_coverage(coverage, genome_length, experiment_name, num_iteration, path)
    plot_genome_depth(coverage, expected_coverage, genome_length, experiment_name, num_iteration, path)

    return np.count_nonzero(coverage) / genome_length


def calculate_mismatch_rate(contigs_alignment_details, genome_length, match_score=10):
    """
    Compute mismatch rate between contigs and genome.

    The mismatch rate is the fraction of bases in contigs that mismatch the reference genome.

    - If close to 0: The contigs are nearly identical to the reference genome.
    - If close to 1: The contigs are highly different from the reference genome.
    - Also considers uncovered bases in the genome as mismatches.
    - Ignores redundant incorrect contigs if the genome is already fully reconstructed.

    Parameters:
        contigs_alignment_details (dict): Dictionary of contig alignment details.
        genome_length (int): Length of the reference genome.
        match_score (int): Score for a matching base.

    Returns:
        float: Mismatch rate.
    """
    total_score = 0
    optimal_score = genome_length * match_score
    covered_bases = set()

    # Track the best score for each genome base
    best_coverage = {i: -float("inf") for i in range(genome_length)}

    for contig in contigs_alignment_details:
        start = contigs_alignment_details[contig]["Start Position"]
        end = contigs_alignment_details[contig]["End Position"]
        if start != -1 and end != -1:
            score = contigs_alignment_details[contig]["Alignment Score"]
            total_score += score
            # Update the best coverage scores for each base
            for i in range(start, min(end, genome_length)):
                best_coverage[i] = max(best_coverage[i], score)

            covered_bases.update(list(range(start, end)))  # Track covered genome positions

    # Check if the genome is perfectly reconstructed
    if len(covered_bases) == genome_length and all(v >= match_score*genome_length
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

def calculate_measures(contigs, reads, num_reads, reads_length, error_prob, reference_genome,
                                             experiment_name, num_iteration, path="plots"):
    """
    Computes essential assembly quality metrics.

    Parameters:
        contigs (list): List of assembled contigs.
        reads (list): List of sequencing reads.
        num_reads (int): Number of reads used for assembly.
        reads_length (int): Length of each read.
        error_prob (float): Probability of mutation in error-prone reads.
        reference_genome (str): The reference genome sequence.
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
    print(f"Calculating performance measures for {experiment_name} (Iteration {num_iteration})")
    contigs_alignment_details = {}

    for contig in contigs:
        _, score, start, end = align_read_or_contig_to_reference(contig, reference_genome, reads_length)

        contigs_alignment_details[contig] = {
            "Alignment Score": score,
            "Start Position": start,
            "End Position": end,
        }

        expected_coverage = num_reads * reads_length / len(reference_genome)

        plot_reconstructed_coverage(contigs, reads, num_reads, reads_length, reference_genome,
                                    experiment_name, num_iteration, path)

    return {
        "Number of Contigs": len(contigs),
        "Genome Coverage": calculate_genome_coverage(contigs_alignment_details, len(reference_genome), expected_coverage, experiment_name, num_iteration, path),
        "N50": calculate_n50(contigs),
        "Mismatch Rate": calculate_mismatch_rate(contigs_alignment_details, reads_length),
    }