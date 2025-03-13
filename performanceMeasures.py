import numpy as np
from aligners import align_read_or_contig_to_reference
from plots import plot_genome_coverage, plot_genome_depth, plot_reconstructed_coverage


def calculate_genome_coverage_and_mismatch_rate(contigs_alignment_details, reference_genome, expected_coverage, experiment_name, num_iteration, path="plots"):
    """
    Compute the percentage of genome bases covered by at least one contig.

    Parameters:
        contigs_alignment_details (dict): Dictionary of contig alignment details.
        reference_genome (str): The reference genome.
        expected_coverage (float): Expected coverage of the genome.
        experiment_name (str): The name of the experiment.
        num_iteration (int): The number of the specific iteration.
        path (str): Path to save the plot.

    Returns:
        tuple (float, float): Genome coverage rate and mismatch rate relative to the genome.
    """
    genome_length = len(reference_genome)
    coverage = np.zeros(genome_length)
    mismatches_aligned_regions = np.zeros(genome_length)

    for contig in contigs_alignment_details:
        alignment_to_print = contigs_alignment_details[contig]["Print"]
        aligned_ref = contigs_alignment_details[contig]["Alignment_reference"]
        aligned_query = contigs_alignment_details[contig]["Alignment_query"]
        score = contigs_alignment_details[contig]["Alignment Score"]
        start = contigs_alignment_details[contig]["Start Position"]
        end = contigs_alignment_details[contig]["End Position"]
        if start != -1 and end != -1:
            # update teh covered_bases with the range of the contig
            coverage[start:end] += 1
            #print(f"Updating coverage for contig {contig} from {start} to {end-1}")
            # update the mismatch rate for the aligned regions
            for i in range(end - start):  # iterate from 0 to the length of the covered section.
                ref_char = aligned_ref[i]
                query_char = aligned_query[i]
                if query_char == '-' or query_char != ref_char:
                    """
                    print("====================")
                    print(f"genom: {reference_genome}")
                    print(f"print: {alignment_to_print}")
                    print(f"start: {start}, i: {i}, end: {end}, len(aligned_ref): {len(aligned_ref)}")
                    print(f"query_char: {query_char}, ref_char: {ref_char}")
                    print(f"aligned_ref[{i}]= {aligned_ref[i]}")
                    print("====================")
                    """
                    mismatches_aligned_regions[start + i] += 1

    # reduce amount of plots
    if num_iteration != 1 and np.all(coverage == coverage[0]):
        pass
    else:
        # plot only interesting plot from the first iteration
        plot_genome_coverage(coverage, genome_length, experiment_name, num_iteration, path)
        plot_genome_depth(coverage, expected_coverage, genome_length, experiment_name, num_iteration, path)

    #print(f"genome length: {genome_length}")
    covered_bases = np.count_nonzero(coverage)
    #print(f"Covered bases: {covered_bases}")
    uncovered_bases = genome_length-covered_bases
    #print(f"Uncovered bases: {uncovered_bases}")
    coverage_rate = covered_bases / genome_length
    bases_covered_with_mismatch_or_indle = np.count_nonzero(mismatches_aligned_regions)
    #print(f"Bases covered with mismatch or indel: {bases_covered_with_mismatch_or_indle}")
    mismatch_rate_aligned_regions = bases_covered_with_mismatch_or_indle / covered_bases if covered_bases > 0 else 0.0
    mismatch_rate_full_genome = (bases_covered_with_mismatch_or_indle+uncovered_bases) / genome_length

    """mismatch_rate_full_genome = calculate_mismatch_rate_full_genome(contigs_alignment_details, reference_genome,
                                                                    coverage)"""
    return coverage_rate, mismatch_rate_aligned_regions, mismatch_rate_full_genome


def calculate_mismatch_rate_aligned_regions(contigs_alignment_details, reference_genome):
    """
    Compute mismatch rate between contigs and genome at nucleotide level using vectorized comparisons.

    The mismatch rate is defined as:
        min(1.0, max(0.0, (total mismatches / total aligned bases) * (total aligned bases / genome_length))
    This gives a value between 0 and 1 representing the fraction of errors relative to the genome.

    Parameters:
        contigs_alignment_details (dict): Dictionary of contig alignment details.
            Each entry should have "Start Position", "End Position", and "Alignment Score".
        reference_genome (str): The original genome sequence.

    Returns:
        float: Mismatch rate.
    """
    genome_length = len(reference_genome)
    total_mismatches = 0
    total_aligned_bases = 0

    for contig, details in contigs_alignment_details.items():
        start = details["Start Position"]
        end = details["End Position"]
        if start != -1 and end != -1:
            aligned_length = end - start
            total_aligned_bases += aligned_length

            # Determine the aligned segments in contig and reference.
            contig_aligned_seq = contig[max(0, -start): min(len(contig), len(contig) + (genome_length - end))]
            reference_aligned_seq = reference_genome[max(0, start): min(genome_length, end)]
            min_len = min(len(contig_aligned_seq), len(reference_aligned_seq))
            if min_len > 0:
                # Convert to numpy arrays for vectorized comparison.
                contig_arr = np.array(list(contig_aligned_seq[:min_len]))
                ref_arr = np.array(list(reference_aligned_seq[:min_len]))
                mismatches = np.sum(contig_arr != ref_arr)
            else:
                mismatches = 0
            total_mismatches += mismatches

    if total_aligned_bases == 0:
        return 0.0  # Avoid division by zero if no bases are aligned

    # Compute the per-base error and scale by the coverage fraction.
    mismatch_rate = (total_mismatches / total_aligned_bases) * (total_aligned_bases / genome_length)
    return min(1.0, max(0.0, mismatch_rate))


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


def calculate_mismatch_rate_full_genome(contigs_alignment_details, reference_genome, coverage):
    """
    Compute overall mismatch rate relative to the entire genome.

    This function adds the number of uncovered bases (from the coverage vector)
    as full errors to the mismatches in aligned regions, then divides by the genome length.

    Parameters:
      contigs_alignment_details (dict): Dictionary of contig alignment details.
      reference_genome (str): The reference genome sequence.
      coverage (np.ndarray): Coverage vector for each base in the genome.

    Returns:
      float: Mismatch rate relative to the full genome.
    """
    genome_length = len(reference_genome)
    total_mismatches = 0
    total_aligned_bases = 0

    for contig, details in contigs_alignment_details.items():
        start = details["Start Position"]
        end = details["End Position"]
        if start != -1 and end != -1:
            aligned_length = end - start
            total_aligned_bases += aligned_length
            contig_aligned_seq = contig[max(0, -start): min(len(contig), len(contig) + (genome_length - end))]
            reference_aligned_seq = reference_genome[max(0, start): min(genome_length, end)]
            min_len = min(len(contig_aligned_seq), len(reference_aligned_seq))
            if min_len > 0:
                contig_arr = np.array(list(contig_aligned_seq[:min_len]))
                ref_arr = np.array(list(reference_aligned_seq[:min_len]))
                mismatches = np.sum(contig_arr != ref_arr)
            else:
                mismatches = 0
            total_mismatches += mismatches

    # Instead of subtracting, we count uncovered bases (where coverage == 0)
    uncovered_bases = np.count_nonzero(coverage == 0)
    # Add uncovered bases as full errors.
    total_mismatches += uncovered_bases
    mismatch_rate = total_mismatches / genome_length
    return min(1.0, mismatch_rate)


def calculate_measures(contigs, reads, num_reads, reads_length, error_prob, ref_genome,
                       experiment_name, num_iteration, path="plots"):
        """
        Computes essential assembly quality metrics.

        Parameters:
            contigs (list): List of assembled contigs.
            reads (list): List of sequencing reads.
            num_reads (int): Number of reads used for assembly.
            reads_length (int): Length of each read.
            error_prob (float): Probability of mutation in error-prone reads.
            ref_genome (str): The reference genome sequence.
            experiment_name (str): The name of the experiment.
            num_iteration (int): The number of the specific iteration.
        Returns:
            dict: tuple(dict, dict) Dictionaries of performance metrics and inforamtion about the alignemtns for storing in the results.csv file.

        Metrics:
        - Number of Contigs: Number of contigs assembled.
        - Genome Coverage: Percentage of genome covered by the contigs.
        - N50: Shortest contig length at 50% total contig length.
        - Mismatch Rate: Fraction of bases in contigs that mismatch the reference genome.
        - Reconstructed Genome Coverage: Plot of read coverage depth for each base in the assembled contigs.
        """
        print(f"Calculating performance measures for {experiment_name} (Iteration {num_iteration})")
        contigs_alignment_details = {}
        expected_coverage = num_reads * reads_length / len(ref_genome)

        for contig in contigs:
            to_print, aligned_ref, aligned_read_or_contig, score, start, end = (
                align_read_or_contig_to_reference(contig, ref_genome, reads_length))

            contigs_alignment_details[contig] = {
                "Print": to_print,
                "Alignment_reference": aligned_ref,
                "Alignment_query": aligned_read_or_contig,
                "Alignment Score": score,
                "Start Position": start,
                "End Position": end,
            }

            #print(f"Alignment: {to_print}\nScore: {score}\nStart: {start}\nEnd: {end}")

            # TODO - run just for small amount of experiments because very computational heavy
            """plot_reconstructed_coverage(contigs, reads, num_reads, reads_length, reference_genome,
                                        experiment_name, num_iteration, path)"""

        genome_coverage, mismatch_rate_aligned_regions, mismatch_rate_full_genome = (
            calculate_genome_coverage_and_mismatch_rate(contigs_alignment_details, ref_genome, expected_coverage,
                                                        experiment_name, num_iteration, path))

        measures = {
            "Number of Contigs": len(contigs),
            "Genome Coverage": genome_coverage,
            "N50": calculate_n50(contigs),
            "Mismatch Rate Aligned Regions": mismatch_rate_aligned_regions,
            "Mismatch Rate Genome Level": mismatch_rate_full_genome,
        }

        return measures, contigs_alignment_details


if __name__ == "__main__":
    """
    from generateErrorFreeReads import generate_error_free_reads
    from generateErrorProneReads import generate_error_prone_reads
    from overlapGraphs import assemble_contigs_using_overlap_graphs
    genome = "GAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTTGATAAAGCAGGAATTACTACTGCTTGTTTACGAATTAAATCGAAGTGGACTGCTGGCGGAAAATGAGAAAATTCGACCTATCCTTGCGCAGCTCGAGAAGCTCTTACTTTGCGACCTTTCGCCATCAACTAACGATTCTGTCAAAAACTGACGCGTTGGATGAGGAGAAGTGGCTTAATATGCTTGGCACGTTCGTCAAGGACTGGTTTAGATATGAGTCACATTTTGTTCATGGTAGAGATTCTCTTGTTGACATTTTAAAAGAGCGTGGATTACTATCTGAGTCCGATGCTGTTCAACCACTAATAGGTAAGAAATCATGAGTCAAGTTACTGAACAATCCGTACGTTTCCAGACCGCTTTGGCCTCTATTAAGCTCATTCAGGCTTCTGCCGTTTTGGATTTAACCGAAGATGATTTCGATTTTCTGACGAGTAACAAAGTTTGGATTGCTACTGACCGCTCTCGTGCTCGTCGCTGCGTTGAGGCTTGCGTTTATGGTACGCTGGACTTTGTGGGATACCCTCGCTTTCCTGCTCCTGTTGAGTTTATTGCTGCCGTCATTGCTTATTATGTTCATCCCGTCAACATTCAAACGGCCTGTCTCATCATGGAAGGCGCTGAATTTACGGAAAACATTATTAATGGCGTCGAGCGTCCGGTTAAAGCCGCTGAATTGTTCGCGTTTACCTTGCGTGTACGCGCAGGAAACACTGACGTTCTTACTGACGCAGAAGAAAACGTGCGTCAAAAATTACGTGCGGAAGGAGTGATGTAATGTCTAAAGGTAAAAAACGTTCTGGCGCTCGCCCTGGTCGTCCGCAGCCGTTGCGAGGTACTAAAGGCAAGCGTAAAGGCGCTCGTCTTTGGTATGTAGGTGGTCAACAATTTTAATTGCAGGGGCTTCGGCCCCTTACTTGAGGATAAATTATGTCTAATATTCAAACTGGCGCCGAGCGTATGCCGCATGACCTTTCCCATCTTGGCTTCCTTGCTGGTCAGATTGGTCGTCTTATTACCATTTCAACTACTCCGGTTATCGCTGGCGACTCCTTCGAGATGGACGCCGTTGGCGCTCTCCGTCTTTCTCCATTGCGTCGTGGCCTTGCTATTGACTCTACTGTAGACATTTTTACTTTTTATGTCCCTCATCGTCACGTTTATGGTGAACAGTGGATTAAGTTCATGAAGGATGGTGTTAATGCCACTCCTCTCCCGACTGTTAACACTACTGGTTATATTGACCATGCCGCTTTTCTTGGCACGATTAACCCTGATACCAATAAAATCCCTAAGCATTTGTTTCAGGGTTATTTGAATATCTATAACAACTATTTTAAAGCGCCGTGGATGCCTGACCGTACCGAGGCTAACCCTAATGAGCTTAATCAAGATGATGCTCGTTATGGTTTCCGTTGCTGCCATCTCAAAAACATTTGGACTGCTCCGCTTCCTCCTGAGACTGAGCTTTCTCGCCAAATGACGACTTCTACCACATCTATTGACATTATGGGTCTGCAAGCTGCTTATGCTAATTTGCATACTGACCAAGAACGTGATTACTTCATGCAGCGTTACCATGATGTTATTTCTTCATTTGGAGGTAAAACCTCTTATGACGCTGACAACCGTCCTTTACTTGTCATGCGCTCTAATCTCTGGGCATCTGGCTATGATGTTGATGGAACTGACCAAACGTCGTTAGGCCAGTTTTCTGGTCGTGTTCAACAGACCTATAAACATTCTGTGCCGCGTTTCTTTGTTCCTGAGCATGGCACTATGTTTACTCTTGCGCTTGTTCGTTTTCCGCCTACTGCGACTAAAGAGATTCAGTACCTTAACGCTAAAGGTGCTTTGACTTATACCGATATTGCTGGCGACCCTGTTTTGTATGGCAACTTGCCGCCGCGTGAAATTTCTATGAAGGATGTTTTCCGTTCTGGTGATTCGTCTAAGAAGTTTAAGATTGCTGAGGGTCAGTGGTATCGTTATGCGCCTTCGTATGTTTCTCCTGCTTATCACCTTCTTGAAGGCTTCCCATTCATTCAGGAACCGCCTTCTGGTGATTTGCAAGAACGCGTACTTATTCGCCACCATGATTATGACCAGTGTTTCCAGTCCGTTCAGTTGTTGCAGTGGAATAGTCAGGTTAAATTTAATGTGACCGTTTATCGCAATCTGCCGACCACTCGCGATTCAATCATGACTTCGTGATAAAAGATTGAGTGTGAGGTTATAACGCCGAAGCGGTAAAAATTTTAATTTTTGCCGCTGAGGGGTTGACCAAGCGAAGCGCGGTAGGTTTTCTGCTTAGGAGTTTAATCATGTTTCAGACTTTTATTTCTCGCCATAATTCAAACTTTTTTTCTGATAAGCTGGTTCTCACTTCTGTTACTCCAGCTTCTTCGGCACCTGTTTTACAGACACCTAAAGCTACATCGTCAACGTTATATTTTGATAGTTTGACGGTTAATGCTGGTAATGGTGGTTTTCTTCATTGCATTCAGATGGATACATCTGTCAACGCCGCTAATCAGGTTGTTTCTGTTGGTGCTGATATTGCTTTTGATGCCGACCCTAAATTTTTTGCCTGTTTGGTTCGCTTTGAGTCTTCTTCGGTTCCGACTACCCTCCCGACTGCCTATGATGTTTATCCTTTGAATGGTCGCCATGATGGTGGTTATTATACCGTCAAGGACTGTGTGACTATTGACGTCCTTCCCCGTACGCCGGGCAATAACGTTTATGTTGGTTTCATGGTTTGGTCTAACTTTACCGCTACTAAATGCCGCGGATTGGTTTCGCTGAATCAGGTTATTAAAGAGATTATTTGTCTCCAGCCACTTAAGTGAGGTGATTTATGTTTGGTGCTATTGCTGGCGGTATTGCTTCTGCTCTTGCTGGTGGCGCCATGTCTAAATTGTTTGGAGGCGGTCAAAAAGCCGCCTCCGGTGGCATTCAAGGTGATGTGCTTGCTACCGATAACAATACTGTAGGCATGGGTGATGCTGGTATTAAATCTGCCATTCAAGGCTCTAATGTTCCTAACCCTGATGAGGCCGCCCCTAGTTTTGTTTCTGGTGCTATGGCTAAAGCTGGTAAAGGACTTCTTGAAGGTACGTTGCAGGCTGGCACTTCTGCCGTTTCTGATAAGTTGCTTGATTTGGTTGGACTTGGTGGCAAGTCTGCCGCTGATAAAGGAAAGGATACTCGTGATTATCTTGCTGCTGCATTTCCTGAGCTTAATGCTTGGGAGCGTGCTGGTGCTGATGCTTCCTCTGCTGGTATGGTTGACGCCGGATTTGAGAATCAAAAAGAGCTTACTAAAATGCAACTGGACAATCAGAAAGAGATTGCCGAGATGCAAAATGAGACTCAAAAAGAGATTGCTGGCATTCAGTCGGCGACTTCACGCCAGAATACGAAAGACCAGGTATATGCACAAAATGAGATGCTTGCTTATCAACAGAAGGAGTCTACTGCTCGCGTTGCGTCTATTATGGAAAACACCAATCTTTCCAAGCAACAGCAGGTTTCCGAGATTATGCGCCAAATGCTTACTCAAGCTCAAACGGCTGGTCAGTATTTTACCAATGACCAAATCAAAGAAATGACTCGCAAGGTTAGTGCTGAGGTTGACTTAGTTCATCAGCAAACGCAGAATCAGCGGTATGGCTCTTCTCATATTGGCGCTACTGCAAAGGATATTTCTAATGTCGTCACTGATGCTGCTTCTGGTGTGGTTGATATTTTTCATGGTATTGATAAAGCTGTTGCCGATACTTGGAACAATTTCTGGAAAGACGGTAAAGCTGATGGTATTGGCTCTAATTTGTCTAGGAAATAACCGTCAGGATTGACACCCTCCCAATTGTATGTTTTCATGCCTCCAAATCTTGGAGGCTTTTTTATGGTTCGTTCTTATTACCCTTCTGAATGTCACGCTGATTATTTTGACTTTGAGCGTATCGAGGCTCTTAAACCTGCTATTGAGGCTTGTGGCATTTCTACTCTTTCTCAATCCCCAATGCTTGGCTTCCATAAGCAGATGGATAACCGCATCAAGCTCTTGGAAGAGATTCTGTCTTTTCGTATGCAGGGCGTTGAGTTCGATAATGGTGATATGTATGTTGACGGCCATAAGGCTGCTTCTGACGTTCGTGATGAGTTTGTATCTGTTACTGAGAAGTTAATGGATGAATTGGCACAATGCTACAATGTGCTCCCCCAACTTGATATTAATAACACTATAGACCACCGCCCCGAAGGGGACGAAAAATGGTTTTTAGAGAACGAGAAGACGGTTACGCAGTTTTGCCGCAAGCTGGCTGCTGAACGCCCTCTTAAGGATATTCGCGATGAGTATAATTACCCCAAAAAGAAAGGTATTAAGGATGAGTGTTCAAGATTGCTGGAGGCCTCCACTATGAAATCGCGTAGAGGCTTTGCTATTCAGCGTTTGATGAATGCAATGCGACAGGCTCATGCTGATGGTTGGTTTATCGTTTTTGACACTCTCACGTTGGCTGACGACCGATTAGAGGCGTTTTATGATAATCCCAATGCTTTGCGTGACTATTTTCGTGATATTGGTCGTATGGTTCTTGCTGCCGAGGGTCGCAAGGCTAATGATTCACACGCCGACTGCTATCAGTATTTTTGTGTGCCTGAGTATGGTACAGCTAATGGCCGTCTTCATTTCCATGCGGTGCACTTTATGCGGACACTTCCTACAGGTAGCGTTGACCCTAATTTTGGTCGTCGGGTACGCAATCGCCGCCAGTTAAATAGCTTGCAAAATACGTGGCCTTATGGTTACAGTATGCCCATCGCAGTTCGCTACACGCAGGACGCTTTTTCACGTTCTGGTTGGTTGTGGCCTGTTGATGCTAAAGGTGAGCCGCTTAAAGCTACCAGTTATATGGCTGTTGGTTTCTATGTGGCTAAATACGTTAACAAAAAGTCAGATATGGACCTTGCTGCTAAAGGTCTAGGAGCTAAAGAATGGAACAACTCACTAAAAACCAAGCTGTCGCTACTTCCCAAGAAGCTGTTCAGAATCAGAATGAGCCGCAACTTCGGGATGAAAATGCTCACAATGACAAATCTGTCCACGGAGTGCTTAATCCAACTTACCAAGCTGGGTTACGACGCGACGCCGTTCAACCAGATATTGAAGCAGAACGCAAAAAGAGAGATGAGATTGAGGCTGGGAAAAGTTACTGTAGCCGACGTTTTGGCGGCGCAACCTGTGACGACAAATCTGCTCAAATTTATGCGCGCTTCGATAAAAATGATTGGCGTATCCAACCTGCA"
    free = generate_error_free_reads(genome, 50, 1000)
    prone = generate_error_prone_reads(free, 0.1)
    contigs = assemble_contigs_using_overlap_graphs(prone)
    calculate_measures(contigs, prone, 1000, 50, 0.1, genome, "t1", 1)
    """

    """a = 'A' * 60
    c = 'C' * 40

    mistake = 'G' * 4

    genome = a + c

    match_score = 10
    mismatch = -1

    contigs_detail = {}  # initialize outside all loops

    contig_a = ('A' * 56) + mistake
    contig_b = mistake + ('C' * 36)


    from aligners import align_read_or_contig_to_reference

    align_a, score_a, start_a, end_a = align_read_or_contig_to_reference(contig_a+'H', genome, 50)
    print(f"align_a: {align_a}\nscore_a: {score_a}\nstart_a: {start_a}\nend_a: {end_a}")
    for letter in ['a', 'B', 'c', 'D', 'E', 'F', 'g', 'H', 'I', 'J', 'K', 'L', 'M']:
        contigs_detail[contig_a+letter] = {
            "Alignment Score": score_a,
            "Start Position": start_a, "End Position": end_a}

    align_b, score_b, start_b, end_b = align_read_or_contig_to_reference('N'+contig_b, genome, 50)
    print(f"align_b: {align_b}\nscore_b: {score_b}\nstart_b: {start_b}\nend_b: {end_b}")
    for letter in ['N', 'O', 'P', 'Q', 'R', 'S', 't', 'U', 'V', 'W', 'X', 'Y', 'Z']:
        contigs_detail[letter+contig_b] = {
            "Alignment Score": score_b,
            "Start Position": start_b, "End Position": end_b}

    print(f"genome: {genome}")
    print(f"contigs: {contigs_detail.keys()}")

    error_rate = calculate_mismatch_rate_nucleotide_level(contigs_detail, genome)

    error_rate_v2 = calculate_mismatch_rate_nucleotide_level_v2(contigs_detail, genome)

    error_rate3 = calculate_mismatch_rate3(contigs_detail, genome, 10)

    error_rate3_v2 = calculate_mismatch_rate3_v2(contigs_detail, genome, 10)

    print(error_rate)
    print("=====")
    print(error_rate3)

    print("=====")

    print(error_rate_v2)
    print("=====")
    print(error_rate3_v2)"""

    #genome = "ATGCGTACGTTAGC"#"ATGCGTACGTTAGCATGCGTACGTTAGC"

    """
    ACGTTAGC
          ||||||||
    TACG-T-GC
    """

    # generate 10 reads of length 5 of the genome
    #reads = [genome[i:i+5] for i in range(0, len(genome)-5)]
    #import random
    # chose random 10 reads
    #reads = random.sample(reads, 10)
    from overlapGraphs import assemble_contigs_using_overlap_graphs, assemble_contigs_string
    #reads = contigs = "ATGCG" #assemble_contigs_using_overlap_graphs(reads, 2)
    #['ACGTTGCGT', 'TGCGT', 'TGCGT']#assemble_contigs_using_overlap_graphs(reads)

    #results = calculate_measures(["TACGTGC"], ["TACGTGC"], 1, 5, 0, genome, "t1", 1)

    #print(f"genome: {genome}")
    #print(f"contigs: TACGTGC")
    #print(f"reads: TACGTGC")
    #print(results)
    #print("++++++++++++++++++++")
    """
    contigs_string = assemble_contigs_string(reads)
    results_string = calculate_measures(contigs_string, reads, 10, 5, 0, genome, "t1", 1)
    print(f"genome: {genome}")
    print(f"contigs: {contigs_string}")
    print(f"reads: {reads}")
    print(results_string)
    """