import random
from generateErrorFreeReads import read_genome_from_fasta, generate_error_free_reads


def introduce_errors(read, error_prob):
    """Introduces sequencing errors (random mutations) into a read."""
    bases = ["A", "T", "C", "G"]
    mutated_read = []

    for base in read:
        if random.random() < error_prob:  # Introduce an error with probability p
            new_base = random.choice([b for b in bases if b != base])  # Ensure different base
            mutated_read.append(new_base)
        else:
            mutated_read.append(base)

    return "".join(mutated_read)


def generate_error_prone_reads(error_free_reads, error_prob):
    """
    Generates error-prone reads using uniform coverage.
    """
    error_reads = []

    for read in error_free_reads:
        error_reads.append(introduce_errors(read, error_prob))  # Add sequencing errors

    return error_reads



