from generateErrorFreeReads import read_genome_from_fasta, generate_error_free_reads
from generateErrorProneReads import generate_error_prone_reads
from overlapGraphs import assemble_contigs_using_overlap_graphs
from performanceMeasures import calculate_measures


def test_assembly(genome, l, N, error_prob, experiment_name, num_iteration, path):
    """
    Test the genome assembly process using error-free and error-prone reads.

    Parameters:
        genome (str): The original genome sequence.
        l (int): Read length.
        N (int): Number of reads.
        error_prob (float): Probability of mutation in error-prone reads.
        experiment_name (str): The name of the experiment.
        num_iteration (int): The number of the specific iteration.
        path (str): The path to save the plots.

    Returns:
        tuple: A tuple containing the assembled contigs and the performance measures.
    """
    # Generate error-free reads
    error_free_reads = generate_error_free_reads(genome, l, N)
    # Generate error-prone reads
    error_prone_reads = generate_error_prone_reads(error_free_reads, error_prob)

    contigs_error_prone = assemble_contigs_using_overlap_graphs(error_prone_reads)

    # Compute performance measures for error-prone assembly
    performance_error_prone = calculate_measures(contigs_error_prone, error_prone_reads, len(error_prone_reads), l,
                                                 error_prob, genome, experiment_name, num_iteration, path)

    return contigs_error_prone, performance_error_prone

