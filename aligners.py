import numpy as np
from numba import njit
from Bio.Align import PairwiseAligner


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

    error_case = ("\nTarget:   \n          \nQuery:    ", 0, 0, 0)

    if not seq1 or not seq2:  # Handle empty sequences
        return error_case

    # Perform the alignment
    alignments = aligner.align(seq1, seq2)

    if not alignments:  # Handle empty alignments
        return error_case

    best_alignment = alignments[0]  # Best alignment object

    if not best_alignment.aligned.any():  # Handle empty alignment
        return error_case

    align_t = best_alignment.target  # Aligned part of seq1 (target)
    align_s = best_alignment.query  # Aligned part of seq2 (query)
    best_score = best_alignment.score    # Extract alignment score

    start_pos, end_pos = 0, 0
    if best_alignment.aligned.any():  # Validate if alignment exists
        start_pos = best_alignment.aligned[1][0][0]  # Start position in seq2
        end_pos = best_alignment.aligned[1][-1][-1]  # End position in seq2

    target_shorter = len(align_t) < len(align_s)

    if target_shorter:
        align_s = align_s[start_pos:end_pos]

    best_alignment_formatted = (
        f"\nTarget:   {align_t }\n"
        f"          {'|' * len(min(align_s, align_t))}\n"
        f"Query:    {align_s}\n"
    )

    # Return aligned sequences, score, and positions
    return (
        best_alignment_formatted,  # Aligned sequences
        best_score,  # Best alignment score
        start_pos,  # Start position in seq2
        end_pos  # End position in seq2
    )


if __name__ == "__main__":

    import random

    def generate_random_sequence(length):
        """Generates a random DNA sequence."""
        bases = "ATGC"
        return "".join(random.choice(bases) for _ in range(length))

    test_cases = []

    # 1-10: Varying Lengths, Perfect Matches
    for i in range(10):
        length = random.randint(1, 20)
        seq = generate_random_sequence(length)
        test_cases.append((seq, seq, f"Perfect match, length {length}"))  # perfect match, varying length

    # 11-20: Varying Lengths, No Matches
    for i in range(10):
        length1 = random.randint(1, 15)
        length2 = random.randint(1, 15)
        seq1 = generate_random_sequence(length1)
        seq2 = generate_random_sequence(length2)
        while seq1 == seq2:
            seq2 = generate_random_sequence(length2)
        test_cases.append((seq1, seq2, f"No match, lengths {length1}, {length2}"))  # no match, varying lengths

    # 21-30: Short Overlaps
    for i in range(10):
        overlap = random.randint(1, 5)
        seq = generate_random_sequence(10)
        offset = random.randint(0, 5)
        if random.random() < 0.5:
            target = seq[offset:offset + overlap]
            query = seq[offset:offset + overlap] + generate_random_sequence(random.randint(1, 5))
        else:
            query = seq[offset:offset + overlap]
            target = seq[offset:offset + overlap] + generate_random_sequence(random.randint(1, 5))

        test_cases.append((target, query, f"Short overlap {overlap}, offset {offset}"))  # short overlaps

    # 31-40: Mismatches
    for i in range(10):
        length = random.randint(5, 15)
        seq = generate_random_sequence(length)
        query = list(seq)
        mismatches = random.randint(1, 3)
        for _ in range(mismatches):
            index = random.randint(0, length - 1)
            bases = "ATGC".replace(query[index], "")
            query[index] = random.choice(bases)
        test_cases.append((seq, "".join(query), f"{mismatches} mismatches"))  # mismatches

    # 41-50: Gaps
    for i in range(10):
        length = random.randint(5, 15)
        seq = generate_random_sequence(length)
        query = list(seq)
        gaps = random.randint(1, 3)
        for _ in range(gaps):
            index = random.randint(0, length)
            if random.random() < 0.5:
                query.insert(index, "-")
            else:
                if 0 < index < len(query):
                    query.pop(index)
        test_cases.append((seq, "".join(query), f"{gaps} gaps"))  # gaps

    # 51-60: Mismatches and Gaps Combined
    for i in range(10):
        length = random.randint(5, 15)
        seq = generate_random_sequence(length)
        query = list(seq)
        mismatches = random.randint(1, 2)
        gaps = random.randint(1, 2)
        for _ in range(mismatches):
            index = random.randint(0, length - 1)
            bases = "ATGC".replace(query[index], "")
            query[index] = random.choice(bases)
        for _ in range(gaps):
            index = random.randint(0, len(query))
            if random.random() < 0.5:
                query.insert(index, "-")
            else:
                if 0 < index < len(query):
                    query.pop(index)
        test_cases.append((seq, "".join(query), f"{mismatches} mismatches, {gaps} gaps"))  # mismatches and gaps

    # 61-70: Long Sequences with Short Matches
    for i in range(10):
        match_length = random.randint(3, 8)
        match_seq = generate_random_sequence(match_length)
        target_prefix = generate_random_sequence(random.randint(20, 50))
        target_suffix = generate_random_sequence(random.randint(20, 50))
        query_prefix = generate_random_sequence(random.randint(20, 50))
        query_suffix = generate_random_sequence(random.randint(20, 50))
        test_cases.append((target_prefix + match_seq + target_suffix, query_prefix + match_seq + query_suffix,
                           f"Long sequences, short match {match_length}"))  # long seqs short match

    # 71-80: Long Sequences, No Local Alignment
    for i in range(10):
        length1 = random.randint(50, 100)
        length2 = random.randint(50, 100)
        seq1 = generate_random_sequence(length1)
        seq2 = generate_random_sequence(length2)
        while seq1 == seq2:
            seq2 = generate_random_sequence(length2)
        test_cases.append((seq1, seq2, f"Long sequences, no match, lengths {length1}, {length2}"))  # long seqs no match

    # 81-90: Repeating Patterns
    for i in range(10):
        pattern_length = random.randint(3, 5)
        pattern = generate_random_sequence(pattern_length)
        target_repeats = random.randint(3, 5)
        query_repeats = random.randint(2, 4)
        target = pattern * target_repeats
        query = pattern * query_repeats
        test_cases.append((target, query, f"Repeating pattern, length {pattern_length}"))  # repeating patterns

    # 91-100: Edge Cases (Very Short, Very Long, Extreme Gaps/Mismatches)
    test_cases.append(("A", "", "Empty query"))  # empty query
    test_cases.append(("", "T", "Empty target"))  # empty target
    test_cases.append(("A" * 1000, "T" * 1000, "Very long, no match"))  # very long no match
    test_cases.append(("ATC" * 100, "ATC" * 100, "Very long perfect match"))  # very long perfect match
    test_cases.append(("A-T-C-G" * 10, "ATCG" * 10, "Extreme gaps"))  # extreme gaps
    test_cases.append(("ATCG" * 10, "A-T-C-G" * 10, "Extreme gaps 2"))  # extreme gaps 2
    test_cases.append(("ATCG" * 10, "TTTT" * 10, "Extreme mismatches"))  # extreme mismatches
    test_cases.append(("ATCG", "A---TCG", "Gaps in query"))  # gaps in query
    test_cases.append(("A---TCG", "ATCG", "Gaps in target"))  # gaps in target
    test_cases.append(("ATCG", "ATCGATCG", "query longer than target"))  # query longer than target

    errors = []
    for i, (target, query, description) in enumerate(test_cases):
        print(f"Target: {target}\nQuery: {query}")
        a, b, c, d = local_alignment(target, query)
        print("==========")
        print(f"alignment: {a}\n, score: {b}, start: {c}, end: {d}")

        e, f, g, h = local_alignment_biopython(target, query)
        print(f"alignment: {e}\n, score: {f}, start: {g}, end: {h}")
        if b==f and c==g and d==h:
            print("Both functions are working correctly")
            print("==========")
        elif b+1==f and c==g and d==h:
            print("==========")
            no_way = "start pos in dp 1 ^^ before ^^ start pos in Bio.Align"
            errors.append(f"test: {i}:\ntarget: {target}\nquery: {query}\ndescription: {no_way+description}\nscore_dp={b}, "
                          f"score_biopython={f}\nstart_dp={c}, start_biopython={g}\nend_dp={d}, end_biopython={h}\n")
        elif b-1==f and c==g and d==h:
            no_way = "start pos in dp 1 ** after ** start pos in Bio.Align"
            errors.append(f"test: {i}:\ntarget: {target}\nquery: {query}\ndescription: {no_way+description}\nscore_dp={b}, "
                          f"score_biopython={f}\nstart_dp={c}, start_biopython={g}\nend_dp={d}, end_biopython={h}\n")
    if len(errors) == 0:
        print("All tests passed successfully")
    else:
        print("Errors in the following tests:")
        for error in errors:
            print(error)

    import time

    def run_performance_test(local_alignment_func1, local_alignment_func2, num_tests=100, min_length=10,
                             max_length=100):
        """
        Runs performance tests on two local alignment functions.

        Args:
            local_alignment_func1: The first local alignment function.
            local_alignment_func2: The second local alignment function.
            num_tests: The number of test cases to run.
            min_length: The minimum sequence length.
            max_length: The maximum sequence length.
        """

        time_local_alignment = 0
        time_bio_align = 0

        for i in range(num_tests):
            length1 = random.randint(min_length, max_length)
            length2 = random.randint(min_length, max_length)
            target = generate_random_sequence(length1)
            query = generate_random_sequence(length2)

            # Time function 1
            start_time = time.time()
            local_alignment(target, query)
            end_time = time.time()
            time_local_alignment += end_time - start_time

            # Time function 2
            start_time = time.time()
            local_alignment_biopython(target, query)
            end_time = time.time()
            time_bio_align += end_time - start_time

        print(f"Total time for function 1: {time_local_alignment:.4f} seconds")
        print(f"Total time for function 2: {time_bio_align:.4f} seconds")

        if time_local_alignment < time_bio_align:
            print("Function 1 is faster.")
        elif time_bio_align < time_local_alignment:
            print("Function 2 is faster.")
        else:
            print("Both functions took the same time.")


    run_performance_test(local_alignment, local_alignment_biopython)