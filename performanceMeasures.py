from aligners import align_read_or_contig_to_reference
from plots import plot_genome_coverage, plot_genome_depth, plot_reconstructed_coverage


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