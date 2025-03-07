import numpy as np
from numba import njit
from Bio.Align import PairwiseAligner


@njit
def overlap_alignment(s, t, match_score=10, mismatch=-1, indel=float('-inf')):
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
    dp = np.zeros((n + 1, m + 1), dtype=np.int64) # No penalty for overhanging ends

    traceback = np.zeros((n + 1, m + 1), dtype=np.int64)  # Table for traceback

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

    best_alignment = f"\nTarget:   {align_t}\n          {'|' * len(align_t)}\nQuery:    {align_s}"
    best_score = int(max_score)
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
        alignment, score, start, end = local_alignment(read_or_contig, reference_genome[-length_read_or_contig:],
                                                       match_score, mismatch, indel)

        start = len(reference_genome) - length_read_or_contig + start
        end = len(reference_genome) - length_read_or_contig + end

    else:
        alignment, score, start, end = local_alignment(read_or_contig, reference_genome, match_score, mismatch, indel)
    return alignment, score, start, end


def local_alignment_biopython(seq1, seq2, match=10, mismatch=-1, gap_open=-1, gap_extend=-1):
    """
    Perform local alignment and extract the best alignment info.

    Args:
        seq1 (str): First sequence.
        seq2 (str): Second sequence.
        match (int): Score for a match between characters in the sequences.
        mismatch (int): Penalty for a mismatch.
        gap_open (float): Penalty for opening a gap.
        gap_extend (float): Penalty for extending a gap.

    Returns:
        tuple: (best_alignment, best_score, start_pos, end_pos)
            - best_alignment: Tuple of aligned sequences (aligned_seq1, aligned_seq2).
            - best_score: Highest local alignment score.
            - start_pos: Start position of the alignment in the second sequence (seq2).
            - end_pos: End position of the alignment in the second sequence (seq2).
    """
    # Initialize the aligner
    aligner = PairwiseAligner()
    aligner.mode = "local"  # Use local alignment mode
    aligner.match_score = match
    aligner.mismatch_score = mismatch
    aligner.open_gap_score = gap_open
    aligner.extend_gap_score = gap_extend

    # Perform the alignment
    alignments = aligner.align(seq1, seq2)
    best_alignment = alignments[0]  # Best alignment object

    # Extract aligned sequences
    aligned_seq1 = best_alignment.target  # Aligned part of seq1 (target)
    aligned_seq2 = best_alignment.query  # Aligned part of seq2 (query)

    # Extract alignment score
    best_score = best_alignment.score

    # Extract alignment start and end positions
    start_pos, end_pos = None, None
    if best_alignment.aligned.any():  # Validate if alignment exists
        start_pos = best_alignment.aligned[1][0][0]  # Start position in seq2
        end_pos = best_alignment.aligned[1][-1][-1]  # End position in seq2

    # Return aligned sequences, score, and positions
    return (
        (aligned_seq1, aligned_seq2),  # Aligned sequences
        best_score,  # Best alignment score
        start_pos,  # Start position in seq2
        end_pos  # End position in seq2
    )


if __name__ == "__main__":
    sequence1 = "GG"
    sequence2 = "TACCGT"
    a, b, c, d = local_alignment(sequence1, sequence2)

    e, f, g, h = local_alignment_biopython(sequence1, sequence2)

    print(f"a: {a}\n, b: {b}, c: {c}, d: {d}")

    print("=========================")

    print(f"e: {e}\n, f: {f}, g: {g}, h: {h}")