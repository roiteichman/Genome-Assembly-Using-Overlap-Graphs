import random


def read_genome_from_fasta(file_path):
    """
    Read a genome sequence from a FASTA file and extract the complete genome.

    Parameters:
        file_path (str): Path to the FASTA file.

    Returns:
        str: The complete genome sequence as a single string.
    """
    genome = ""
    with open(file_path, 'r') as file:
          for line in file:
            if not line.startswith('>'):  # first line in fasta file starts with > skip
                genome += line.strip()
    return genome


def generate_error_free_reads(genome_sequence, read_length, num_reads):
    """
    Generates error-free reads that cover the genome as uniformly as possible.

    Parameters:
        genome_sequence (str): The complete genome sequence.
        read_length (int): Length of each read.
        num_reads (int): Total number of reads to generate.

    Returns:
        list: List of error-free reads.
    """
    genome_length = len(genome_sequence)

    reads = []

    for _ in range(num_reads):

        # take a read in a length of read_length uniformly from the genome
        # consider the genome linear and not cyclic
        # means if the start position+read_length>len(genome)
        # give a shorter read just between [start_position:len(genome)]
        start_position = random.randint(0, genome_length-1) # -1 because I don't want to get an empty read
        if start_position + read_length > genome_length:
            read = genome_sequence[start_position:genome_length]
        else:
            read = genome_sequence[start_position:start_position+read_length]

        reads.append(read)

    return reads


def calculate_coverage(genome, N, l):
    return N * l / len(genome)
