import networkx as nx
from aligners import overlap_alignment


def construct_overlap_graph_nx_k(reads, k=5):
    """
    Construct an overlap graph based on overlap alignment scores using NetworkX.

    Parameters:
        reads (list): A list of DNA reads.
        k (int): The length of the k-mer prefix to use for filtering reads.

    Returns:
        nx.DiGraph: A NetworkX directed graph where nodes are reads and edges
                    represent overlaps with scores and end positions.
    """
    print("Constructing overlap graph...")
    read_copies = {}
    for read in reads:
        read_copies[read] = read_copies.get(read, 0) + 1

    overlap_graph = nx.DiGraph()

    # Add nodes to G and handle copies, e.g., read = "AAA" so nodes will be "AAA_0", "AAA_1", ... will be added
    for read, count in read_copies.items():
        for copy_index in range(count):
            node_name = f"{read}_{copy_index}"
            overlap_graph.add_node(node_name)

    # Build a prefix index: map from k-mer (prefix) to a list of (read, count)
    prefix_index = {}
    for read, count in read_copies.items():
        if len(read) >= k:
            prefix = read[:k]
        else:
            prefix = read # if read is shorter than k, use the whole read as prefix
        if prefix not in prefix_index:
            prefix_index[prefix] = []
        prefix_index[prefix].append((read, count))

    # For each read, check if its suffix k-mer is present in the prefix index
    for read_a, count_a in read_copies.items():
        if len(read_a) >= k:
            suffix = read_a[-k:]
        else:
            suffix = read_a
        # Only consider reads that share the same prefix
        candidate_reads = prefix_index.get(suffix, [])
        for read_b, count_b in candidate_reads:
            # Avoid self-overlaps and overlaps between identical reads
            if read_a != read_b:
                alignment, score, alignment_end_position = overlap_alignment(read_a, read_b)
                # For each copy of the reads, add an edge with the alignment score
                for copy_index_a in range(count_a):
                    for copy_index_b in range(count_b):
                        node_a_name = f"{read_a}_{copy_index_a}"
                        node_b_name = f"{read_b}_{copy_index_b}"
                        overlap_graph.add_edge(node_a_name, node_b_name, weight=score,
                                               end_position=alignment_end_position)
    return overlap_graph, read_copies


def create_contig(start_read, dag, visited, topo_order):
    """
    Create a contig using overlap alignment-based traversal while preserving the topological order.

    Parameters:
        start_read (str): The read to start from.
        dag (nx.DiGraph): The overlap graph.
        visited (set): Reads that start a contig.
        topo_order (dict): A mapping of reads to their position in the topological order.

    Returns:
        str: The assembled contig.
    """
    print("Creating contig...")
    # Initialize the contig with the start read
    contig = start_read.split("_")[0]
    # Mark the start read as visited
    visited.add(start_read.split("_")[0])
    neighbors = list(dag.neighbors(start_read))
    # Traverse the graph while preserving the topological order
    while neighbors:
        valid_neighbors = [neighbor for neighbor in neighbors if neighbor.split("_")[0] not in visited]

        # If there are no valid neighbors, stop
        if not valid_neighbors:
            break

        # Select the next read based on topological order
        next_read = min(valid_neighbors, key=lambda neighbor: topo_order.get(neighbor.split("_")[0], float('inf')))
        alignment_end_position = dag.edges[start_read, next_read]['end_position']
        # Merge the read based on overlap length
        contig += next_read.split("_")[0][alignment_end_position:]

        # Continue traversal
        start_read = next_read
        neighbors = list(dag.neighbors(start_read))

    return contig


def remove_cycles_from_graph(overlap_graph):
    """
    Remove cycles from a directed overlap graph by removing the weakest edge (lowest overlap score)
    until the graph becomes a DAG.

    Parameters:
        overlap_graph (nx.DiGraph): The directed overlap graph.

    Returns:
        nx.DiGraph: A DAG (Directed Acyclic Graph) with cycles removed.
    """
    print("Removing cycles from graph...")
    G = overlap_graph

    while True:
        try:
            # try to find a cycle in the graph
            cycle = nx.find_cycle(G, orientation='original')
        except nx.NetworkXNoCycle:
            break
        # Remove the weakest edge in the cycle
        weakest_edge = min(((u, v, G[u][v]["weight"]) for u, v, _ in cycle), key=lambda x: x[2])
        u, v, weight = weakest_edge
        G.remove_edge(u, v)

    return G


def topological_sort(dag):
    """
    Perform topological sorting on a DAG.

    Parameters:
        dag (nx.DiGraph): A directed acyclic graph (DAG).

    Returns:
        list: A topologically sorted list of reads.
    """
    print("Sorting graph topologically...")
    try:
        sorted_reads = list(nx.topological_sort(dag))
        return sorted_reads
    except nx.NetworkXUnfeasible:
        raise ValueError("Graph is not a DAG! Cycles still exist.")


def assemble_contigs_using_overlap_graphs(reads):
    """
    Assemble contigs using an overlap alignment graph with cycle removal and topological sorting.

    Parameters:
        reads (list): List of DNA reads.

    Returns:
        list: List of assembled contigs.
    """

    # Step 1: Construct the initial overlap graph
    overlap_graph, read_copies = construct_overlap_graph_nx(reads)
    # Step 2: remove and sort the graph
    dag = remove_cycles_from_graph(overlap_graph)
    # Step 3: Sort the graph topologically
    topo_order_with_copies = {node: i for i, node in enumerate(nx.topological_sort(dag))}
    # Remove the copy index from the read and create a mapping of reads to their topological order index
    topo_order = {}
    for read_with_copy in topo_order_with_copies.keys():
        read = read_with_copy.split("_")[0]
        topo_order[read] = topo_order_with_copies[read_with_copy]

    # Step 4: Assemble contigs following the sorted order
    visited = set()
    contigs = []
    for read in topo_order.keys():
        if read not in visited:
            for copy_index in range(read_copies[read]):
                node_name = f"{read}_{copy_index}"
                # Create a contig starting from the read
                contig = create_contig(node_name, dag, visited, topo_order)
                contigs.append(contig)
    return contigs
