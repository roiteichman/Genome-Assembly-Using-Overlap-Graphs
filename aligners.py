import numpy as np
from numba import njit


@njit
def overlap_alignment(s, t, match_score=10, mismatch=-1, indel=-5):
    """
    Compute the best overlap alignment score between two sequences.
    Overhanging ends are not penalized.

    Parameters:
        s (str): First sequence.
        t (str): Second sequence.
        match_score (int): Score for a matching base.
        mismatch (int): Penalty for a mismatch.
        indel (int): Penalty for an insertion/deletion (gap).

    Returns:
        tuple: (best_alignment, best_score, best_overlap_length)
        best_alignment (str): The best overlap alignment.
        best_score (int): The alignment score.
        alignment_end_position (int): End position of the alignment in the target sequence.
    """
    n, m = len(s), len(t)
    dp = np.zeros((n + 1, m + 1), dtype=np.int64)

    traceback = np.zeros((n + 1, m + 1), dtype=np.int64)  # Table for traceback

    # Initialize DP table - Just for code readability
    for i in range(1, n + 1):
        dp[i][0] = 0  # No penalty for overhanging ends
    for j in range(1, m + 1):
        dp[0][j] = 0

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            scores = [
                dp[i - 1][j - 1] + (match_score if s[i - 1] == t[j - 1] else mismatch),
                dp[i - 1][j] + indel,
                dp[i][j - 1] + indel,
            ]
            dp[i][j] = max(scores)
            traceback[i][j] = scores.index(dp[i][j])  # Store traceback direction

    # Find best overlap score (max value in last row or last column)
    max_score = -float('inf')
    overlap_len = 0

    # for i in range(n + 1): # running on the last column
    #    if dp[i][m] > max_score:
    #        max_score = dp[i][m]
    #        overlap_len = i
    for j in range(m + 1):  # running on the last row - s is the 'origin' and t the 'destination'
        if dp[n][j] > max_score:
            max_score = dp[n][j]
            overlap_len = j

    # Backtrack to construct alignment
    align_s = ""
    align_t = ""
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

    # Print the alignment
    # print(f"\nTarget:   {align_t}\n          {'|' * len(align_t)}\nQuery:    {align_s}")

    # return best_alignment, best_score, best_overlap_len
    best_alignment = f"\nTarget:   {align_t}\n          {'|' * len(align_t)}\nQuery:    {align_s}"
    best_score = max_score
    alignment_end_position = overlap_len

    return best_alignment, best_score, alignment_end_position


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
        tuple: (best_alignment, best score, start position in reference, end position in reference)
        best_alignment (str): The best local alignment.
        best_score (int): The alignment score.
        start_pos (int): Start position in the reference genome.
        end_pos (int): End position in the reference genome.
    """
    n, m = len(query), len(reference)

    # DP matrix and traceback table
    dp = np.zeros((n + 1, m + 1), dtype=np.int64)
    traceback = np.zeros((n + 1, m + 1), dtype=np.int64)

    # Track best score and position
    best_score = 0
    best_i, best_j = 0, 0

    # Fill the DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_or_mismatch = dp[i - 1][j - 1] + (match_score if query[i - 1] == reference[j - 1] else mismatch)
            delete = dp[i - 1][j] + indel
            insert = dp[i][j - 1] + indel

            # Find best score for this cell
            dp[i][j] = max(0, match_or_mismatch, delete, insert)

            # Assign traceback direction
            if dp[i][j] == match_or_mismatch:
                traceback[i][j] = 1  # Diagonal (match/mismatch)
            elif dp[i][j] == delete:
                traceback[i][j] = 2  # Up (gap in reference)
            elif dp[i][j] == insert:
                traceback[i][j] = 3  # Left (gap in query)

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

    best_alignment = (f"\nTarget:   {aligned_reference}\n          {'|' * len(aligned_reference)}\nQuery:    "
                      f"{aligned_query}")

    return best_alignment, best_score, start_pos, end_pos


def align_read_or_contig_to_reference(read_or_contig, reference_genome, read_length):
    """
    Aligns a contig to a reference genome using local alignment.

    Args:
        read_or_contig (str): The read / contig sequence.
        reference_genome (str): The reference genome sequence.
        read_length (int): The length of the reads used for assembly.

    Returns:
        tuple: A tuple containing the alignment, the score, start, and end positions of the alignment
        in the reference genome.
    """
    length_read_or_contig = len(read_or_contig)
    if length_read_or_contig < read_length:
        print(f"read_length: {read_length}")
        print(f"read_or_contig: {read_or_contig}")
        print(f"reference_genome[-{length_read_or_contig}:]: {reference_genome[-length_read_or_contig:]}")
        alignment, score, start, end = local_alignment(read_or_contig, reference_genome[-length_read_or_contig:])
        print(f"current_start: {start}")
        print(f"current_end: {end}")
        start = len(reference_genome) - length_read_or_contig + start
        end = len(reference_genome) - length_read_or_contig + end
        print(f"updated_start: {start}")
        print(f"updated_end: {end}")
    else:
        alignment, score, start, end = local_alignment(read_or_contig, reference_genome)
    return alignment, score, start, end


if __name__ == "__main__":
    a, s, st, en = align_read_or_contig_to_reference("ATGCG", "ATGCGTACGATGCGATGCGTACGATGCG", 41)
