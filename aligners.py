import numpy as np
from numba import njit


@njit
def overlap_alignment(s, t, match_score=10, mismatch=-1, indel=-2**31):
    """
    Compute the best overlap alignment score between two sequences.
    Overhanging ends are not penalized.

    Parameters:
        s (str): Source sequence.
        t (str): Target sequence.
        match_score (int): Score for a matching base.
        mismatch (int): Penalty for a mismatch.
        indel (int): Penalty for an insertion/deletion (gap).

    Returns:
        tuple: (alignment_to_print, align_s, align_t, best_score, alignment_end_position)
        alignment_to_print (str): The best overlap alignment, nice to print and see the alignment.
        align_s (str): The part of the source sequence that aligned.
        align_t (str): The part of the target sequence that aligned.
        best_score (int): The alignment score.
        alignment_end_position (int): End position of the alignment in the target sequence, for concatenate them correctly.
    """
    n, m = len(s), len(t)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32) # No penalty for overhanging ends

    traceback = np.zeros((n + 1, m + 1), dtype=np.int8)  # Table for traceback

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag_score = dp[i - 1][j - 1] + (match_score if s[i - 1] == t[j - 1] else mismatch)
            up_score = dp[i - 1][j] + indel
            left_score = dp[i][j - 1] + indel

            # Find the maximum score
            if diag_score >= up_score and diag_score >= left_score:
                dp[i, j] = diag_score
                traceback[i, j] = 0
            elif up_score >= left_score:
                dp[i, j] = up_score
                traceback[i, j] = 1
            else:
                dp[i, j] = left_score
                traceback[i, j] = 2

    # Find best overlap score (max value in last row or last column)
    max_score = -float('inf')
    overlap_len = 0

    for j in range(m + 1):  # running on the last row - s is the 'origin' and t the 'destination'
        if dp[n][j] > max_score:
            max_score = dp[n][j]
            overlap_len = j

    # Backtrack to construct alignment
    align_s, align_t = "", ""
    i, j = n, overlap_len  # Changed to i, j

    while i > 0 and j > 0:
        if traceback[i][j] == 0:  # Diagonal (match/mismatch)
            align_s = s[i - 1] + align_s
            align_t = t[j - 1] + align_t
            i -= 1
            j -= 1
        elif traceback[i][j] == 1:  # Up (deletion in t)
            align_s = s[i - 1] + align_s
            align_t = "-" + align_t
            i -= 1
        else:  # Left (insertion in t)
            align_s = "-" + align_s
            align_t = t[j - 1] + align_t
            j -= 1

    alignment_to_print = f"\nTarget:   {align_t}\n          {'|' * len(align_t)}\nQuery:    {align_s}"
    best_score = int(max_score)
    alignment_end_position = overlap_len

    return alignment_to_print, align_s, align_t, best_score, alignment_end_position


@njit
def local_alignment(query, reference, match_score=10, mismatch=-1, indel=-1):
    """
    Compute the best local alignment score between two sequences using dynamic programming.

    Parameters:
        query (str): The sequence to align (read or contig).
        reference (str): The reference sequence (genome).
        match_score (int): Score for a matching base.
        mismatch (int): Penalty for a mismatch.
        indel (int): Penalty for an insertion/deletion (gap).

    Returns:
        tuple: (alignment_to_print, aligned_reference, aligned_query, best score, start position in reference, end position in reference)
        alignment_to_print (str): The best overlap alignment, nice to print and see the alignment.
        aligned_reference (str): The part of the reference (genome or contig) that aligned.
        aligned_query (str): The part of the query (read or contig) sequence that aligned.
        best_score (int): The alignment score.
        start_pos (int): Start position in the reference genome.
        end_pos (int): End position in the reference genome.
    """
    n, m = len(query), len(reference)
    # DP matrix and traceback table
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    traceback = np.zeros((n + 1, m + 1), dtype=np.int8)

    # Track best score and position
    best_score = 0
    best_i, best_j = 0, 0

    # Fill the DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag_score = dp[i - 1][j - 1] + (match_score if query[i - 1] == reference[j - 1] else mismatch)
            up_score = dp[i - 1][j] + indel
            left_score = dp[i][j - 1] + indel

            if diag_score >= up_score and diag_score >= left_score and diag_score >= 0:
                dp[i, j] = diag_score
                traceback[i, j] = 1  # Diagonal (match/mismatch)
            elif up_score >= left_score and up_score >= 0:
                dp[i, j] = up_score
                traceback[i, j] = 2
            elif left_score >= 0:
                dp[i, j] = left_score
                traceback[i, j] = 3
            else:
                dp[i, j] = 0 #traceback[i, j] = 0 already initialized

            # Update best scoring position
            if dp[i][j] > best_score:
                best_score = dp[i][j]
                best_i, best_j = i, j

    # Backtrack using the traceback matrix
    aligned_query, aligned_reference = "", ""
    i, j = best_i, best_j

    while i > 0 and j > 0 and dp[i][j] > 0:
        if traceback[i][j] == 1:  # Diagonal (match/mismatch)
            aligned_query = query[i - 1] + aligned_query
            aligned_reference = reference[j - 1] + aligned_reference
            i -= 1
            j -= 1
        elif traceback[i][j] == 2:  # Up (gap in reference)
            aligned_query = query[i - 1] + aligned_query
            aligned_reference = "-" + aligned_reference
            i -= 1
        elif traceback[i][j] == 3:  # Left (gap in query)
            aligned_query = "-" + aligned_query
            aligned_reference = reference[j - 1] + aligned_reference
            j -= 1
        else:
            break  # Stop when we hit a score of 0

    # Convert to reference genome indices
    start_pos = j
    end_pos = best_j

    alignment_to_print = (f"\nTarget:   {aligned_reference}\n          {'|' * len(aligned_reference)}\nQuery:    "
                      f"{aligned_query}")

    return alignment_to_print, aligned_reference, aligned_query, best_score, start_pos, end_pos


def align_read_or_contig_to_reference(read_or_contig, reference_genome, read_length, match_score=10, mismatch=-1, indel=-1):
    """
    Aligns a contig to a reference genome using local alignment.

    Args:
        read_or_contig (str): The read / contig sequence.
        reference_genome (str): The reference genome sequence.
        read_length (int): The length of the reads used for assembly.
        match_score (int): Score for a matching base.
        mismatch (int): Penalty for a mismatch.
        indel (int): Penalty for an insertion/deletion (gap).

    Returns:
        tuple: A tuple containing the alignment, the score, start, and end positions of the alignment
        in the reference genome.
    """
    length_read_or_contig = len(read_or_contig)
    if length_read_or_contig < read_length:
        to_print, aligned_ref, aligned_read_or_contig, score, start, end = local_alignment(read_or_contig,
                                                                                           reference_genome
                                                                                           [-length_read_or_contig:],
                                                                                           match_score, mismatch,
                                                                                           indel)

        start = len(reference_genome) - length_read_or_contig + start
        end = len(reference_genome) - length_read_or_contig + end

    else:
        to_print, aligned_ref, aligned_read_or_contig, score, start, end = local_alignment(read_or_contig,
                                                                                           reference_genome,
                                                                                           match_score, mismatch, indel)

    return to_print, aligned_ref, aligned_read_or_contig, score, start, end
