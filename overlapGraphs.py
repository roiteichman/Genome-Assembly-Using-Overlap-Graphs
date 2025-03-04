import networkx as nx
from aligners import overlap_alignment, local_alignment, align_read_or_contig_to_reference


def construct_overlap_graph(reads):
    """
    Construct an overlap graph based on overlap alignment scores.

    Parameters:
        reads (list): A list of DNA reads.

    Returns:
        dict: A dictionary where keys are reads and values are lists of tuples
              (neighboring_read, score, overlap_length).
    """
    overlap_graph = {}
    # Compare all pairs of reads to find overlaps
    for i, read_a in enumerate(reads):
        overlap_graph[read_a] = []
        for j, read_b in enumerate(reads):
            if i != j:
                # Use dynamic programming function to calculate the score and the length of the overlap alignment
                print(f"aligning {read_a} and {read_b}")
                alignment, score, alignment_end_position = overlap_alignment(read_a, read_b)
                print(f"alignment is: \n{alignment}\n")
                print(f"score is: {score}")
                print(f"alignment_end_position is: {alignment_end_position }")
                overlap_graph[read_a].append((read_b, score, alignment_end_position ))
    return overlap_graph


def create_contig(start_read, overlap_graph, in_visited_reads, out_visited_reads, topo_order):
    """
    Create a contig using overlap alignment-based traversal while preserving the topological order.

    Parameters:
        start_read (str): The read to start from.
        overlap_graph (dict): The overlap graph.
        in_visited_reads (set): Reads that start a contig.
        out_visited_reads (set): Reads already merged into a contig.
        topo_order (dict): A mapping of reads to their position in the topological order.

    Returns:
        str: The assembled contig.
    """
    print(f"start_read is: {start_read}")
    print(f"overlap_graph is: {overlap_graph}")
    print(f"in_visited_reads is: {in_visited_reads}")
    print(f"out_visited_reads is: {out_visited_reads}")
    print(f"topo_order is: {topo_order}")
    contig = start_read
    out_visited_reads.add(start_read)

    if start_read in overlap_graph:
        while overlap_graph.get(start_read):
            # Filter out reads that have already been visited
            print(f"overlap_graph.get({start_read}): {overlap_graph.get(start_read)}")
            print(f"overlap_graph[{start_read}]: {overlap_graph[start_read]}")
            print(f"in_visited_reads is: {in_visited_reads}")
            # get the reads that are not in in_visited_reads
            valid_overlaps = [entry for entry in overlap_graph[start_read] if entry[0] not in in_visited_reads]
            print(f"valid_overlaps is: {valid_overlaps}")

            if not valid_overlaps:
                break

            # Select the next read based on topological order
            next_read = min(valid_overlaps, key=lambda x: topo_order.get(x[0], float('inf')))[0]
            print(f"next_read: {next_read}")

            # Merge the read based on overlap length
            alignment_end_position = next((entry[2] for entry in valid_overlaps if entry[0] == next_read), 0)
            print("**********")
            print(f"contig: {contig}")
            print(f"next_read: {next_read}, ")
            print(f"alignment_end_position : {alignment_end_position }")
            print(f"{next_read}[{alignment_end_position }:]: {next_read[alignment_end_position :]}")

            contig += next_read[alignment_end_position:]
            print(f"concated_contig: {contig}")
            print("**********")
            # Mark as visited and continue
            in_visited_reads.add(next_read)
            start_read = next_read

    return contig


def assemble_contigs_using_overlap_graphs(reads, genome, l):
    """
    Assemble contigs using an overlap alignment graph with cycle removal and topological sorting.

    Parameters:
        reads (list): List of DNA reads.
        genome (str): The reference genome sequence.
        l (int): Read length.

    Returns:
        list: List of assembled contigs.
    """

    # Step 1: Construct the initial overlap graph
    overlap_graph = construct_overlap_graph(reads)
    print(f"overlap_graph is: {overlap_graph}")

    # Step 2: remove and sort the graph
    sorted_reads, sorted_graph = remove_and_sort(overlap_graph, reads, genome, l)
    print(f"sorted_reads is: {sorted_reads}")
    print(f"sorted_graph is: {sorted_graph}")

    # Create a mapping of reads to their topological order index
    topo_order = {read: idx for idx, read in enumerate(sorted_reads)}
    print(f"topo_order is: {topo_order}")

    # Step 4: Assemble contigs following the sorted order
    in_visited_reads = set()
    out_visited_reads = set()
    contigs = []

    for read in sorted_reads:
        print(f"read is: {read}")
        print(f"in_visited_reads is: {in_visited_reads}")
        print(f"out_visited_reads is: {out_visited_reads}")
        if read not in in_visited_reads:
            contig = create_contig(read, overlap_graph, in_visited_reads, out_visited_reads, topo_order)
            contigs.append(contig)

    return contigs


def remove_and_sort(overlap_graph, reads, reference_genome, read_length):
    """
    Sorts the overlap graph topologically by determining read order and removing conflicting edges.

    Parameters:
        overlap_graph (dict): A dictionary where keys are reads and values are lists of tuples
                              (neighboring_read, score, overlap_length).
        reads (list): List of read sequences.
        reference_genome (str): The reference genome sequence.
        read_length (int): Length of each read.

    Returns:
        tuple: (sorted_reads, overlap_graph)
        sorted_reads (list): A topologically sorted list of read sequences.
        overlap_graph (dict): The updated overlap graph with conflicting edges removed.
    """
    orders = {}  # Dictionary to store read order

    # Determine read order using local alignment
    for read in reads:
        print(f"\nalign read {read} to {reference_genome}")
        alignment, score, start, end = align_read_or_contig_to_reference(read, reference_genome, read_length)
        print(f"alignment is: {alignment}")
        print(f"score is: {score}")
        print(f"start is: {start}")
        print(f"end is: {end}\n")
        if start != -1:
            orders[read] = start

    # Sort reads based on their order in the reference genome
    print(f"orders is: {orders}")
    sorted_reads = sorted(orders.keys(), key=lambda r: orders[r])
    print(f"sorted_reads is: {sorted_reads}")

    # Remove edges that violate topological order
    for read_a in sorted_reads:
        if read_a in overlap_graph:
            print("^^^^^^^^^^^^")
            print(f"read_a is: {read_a}")
            cpy = overlap_graph[read_a].copy()
            print(f"overlap_graph[read_a] is: {overlap_graph[read_a]}")
            overlap_graph[read_a] = [
                (read_b, score, overlap_len)
                for read_b, score, overlap_len in overlap_graph[read_a]
                if read_b in orders and orders[read_a] < orders[read_b]
            ]
            removed = [entry for entry in cpy if entry not in overlap_graph[read_a]]
            print(f"overlap_graph[read_a] is: {overlap_graph[read_a]}")
            print(f"is removed: {cpy != overlap_graph[read_a]}")
            print(f"removed is: {removed}")
            print("^^^^^^^^^^^^")

    # check if the graph is DAG
    if not nx.is_directed_acyclic_graph(nx.DiGraph(overlap_graph)):
        raise ValueError("Graph is not a DAG! Cycles still exist.")

    return sorted_reads, overlap_graph
