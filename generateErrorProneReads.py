import numpy as np
from numba import njit

@njit
def introduce_errors(read: str, error_prob: float, alphabet: dict) -> str:
    """
        Introduces sequencing errors (random mutations) into a read.

        Parameters:
            read (str): The read to introduce errors into.
            error_prob (float): The probability of introducing an error.
            alphabet (dict): The alphabet of nucleotides.

        Returns:
            str: The read with errors introduced.
        """
    errs = np.nonzero(np.random.random(len(read)) <= error_prob)[0]
    indices = np.array(list(range(3)), dtype=np.uint8)
    nucleotides = np.random.choice(indices, len(errs))
    seq = []
    prev_position = 0
    for i, position in enumerate(errs):
        seq.append(read[prev_position:position])
        nuc = alphabet[read[position]][nucleotides[i]]
        seq.append(nuc)
        prev_position = position + 1
    seq.append(read[prev_position:])
    return "".join(seq)

@njit
def generate_error_prone_reads(error_free_reads, error_prob):
    """
    Generates error-prone reads using uniform coverage.
    Use the `introduce_errors` function to introduce errors into the reads.

    Parameters:
        error_free_reads (list): List of error-free reads.
        error_prob (float): Probability of introducing an error.

    Returns:
        list: List of error-prone reads
    """
    print("Generating error-prone reads...")
    alphabet = {"A": "CGT", "C": "AGT", "G": "ACT", "T": "ACG"}
    return [introduce_errors(read, error_prob, alphabet) for read in error_free_reads]

