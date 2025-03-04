from generateErrorFreeReads import read_genome_from_fasta, generate_error_free_reads
from generateErrorProneReads import generate_error_prone_reads
from overlapGraphs import assemble_contigs_using_overlap_graphs
from performanceMeasures import calculate_essential_performance_measures


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
        dict: A dictionary containing the performance measures for error-free and error-prone assemblies.
    """
    # Generate error-free reads
    error_free_reads = generate_error_free_reads(genome, l, N)
    # Generate error-prone reads
    error_prone_reads = generate_error_prone_reads(error_free_reads, error_prob)

    contigs_error_prone = assemble_contigs_using_overlap_graphs(error_prone_reads, genome, l)

    # Compute performance measures for error-prone assembly
    performance_error_prone = calculate_essential_performance_measures(contigs_error_prone, error_prone_reads,
                                                                       len(error_prone_reads), l, error_prob, genome,
                                                                       "error_prone", experiment_name,
                                                                       num_iteration, path)

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print(f"--- Error-Prone Reads Assembly ({experiment_name} - iteration: {num_iteration}) ---")
    print(f"Error-Prone Reads: {error_prone_reads}")
    print(f"is_changed: {set(error_free_reads) != set(error_prone_reads)}")
    print(f"Contigs: {contigs_error_prone}")
    print(f"Original Genome: {genome}")
    print(f"performance_error_free:")
    print(f"Number of Contigs: {performance_error_prone['Number of Contigs']}")
    print(f"Genome Coverage: {performance_error_prone['Genome Coverage']}")
    print(f"Genome Coverage Plot: {performance_error_prone['Genome Coverage Plot']}")
    print(f"Genome Depth Plot: {performance_error_prone['Genome Depth Plot']}")
    print(f"N50: {performance_error_prone['N50']}")
    print(f"Mismatch Rate: {performance_error_prone['Mismatch Rate']}")
    print(f"Reconstructed Genome Coverage: {performance_error_prone['Reconstructed Genome Coverage']}")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    return performance_error_prone

