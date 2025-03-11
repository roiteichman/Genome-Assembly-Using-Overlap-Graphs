import networkx as nx
from aligners import overlap_alignment
from itertools import combinations

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
    assert k >= 0, "k-mer length must be non-negative"
    read_copies = {}
    for read in reads:
        read_copies[read] = read_copies.get(read, 0) + 1

    overlap_graph = nx.DiGraph()

    # Add nodes to G and handle copies, e.g., read = "AAA" so nodes will be "AAA_0", "AAA_1", ... will be added
    for read, count in read_copies.items():
        for copy_index in range(count):
            node_name = f"{read}_{copy_index}"
            overlap_graph.add_node(node_name)

    prefix_index = {}
    if k > 0:
        # Build a prefix index: map from k-mer (prefix) to a list of (read, count)
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
        if len(read_a) >= k > 0:
            suffix = read_a[-k:]
        else:
            suffix = read_a
        # Only consider reads that share the same prefix
        candidate_reads = prefix_index.get(suffix, []) if k > 0 else read_copies.items()
        for read_b, count_b in candidate_reads:
            # Avoid self-overlaps and overlaps between identical reads
            if read_a != read_b:
                to_print, aligned_source, aligned_target, score, alignment_end_position = overlap_alignment(read_a, read_b)#, mismatch=-1000000) #TODO - remove, its just for small example
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
        visited.add(start_read.split("_")[0])  # mark every read in the path as visited.

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


def assemble_contigs_using_overlap_graphs(reads, k=5):
    """
    Assemble contigs using an overlap alignment graph with cycle removal and topological sorting.

    Parameters:
        reads (list): List of DNA reads.
        k (int): The length of the k-mer prefix to use for filtering reads.

    Returns:
        list: List of assembled contigs.
    """

    # Step 1: Construct the initial overlap graph
    print("Constructing overlap graph...")
    overlap_graph, read_copies = construct_overlap_graph_nx_k(reads, k=k)
    # Step 2: remove and sort the graph
    print("Removing cycles from graph...")
    dag = remove_cycles_from_graph(overlap_graph)
    # Step 3: Sort the graph topologically
    topo_order_with_copies = {node: i for i, node in enumerate(nx.topological_sort(dag))}
    # Remove the copy index from the read and create a mapping of reads to their topological order index
    topo_order = {}
    for read_with_copy in topo_order_with_copies.keys():
        read = read_with_copy.split("_")[0]
        topo_order[read] = topo_order_with_copies[read_with_copy]

    # Step 4: Assemble contigs following the sorted order
    print("Creating contig...")
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


def construct_overlap_graph_string(reads):
    """
    Constructs an overlap graph following the string graph approach.

    Parameters:
        reads (list): List of DNA reads.

    Returns:
        nx.DiGraph: A NetworkX directed graph representing the string graph.
        dict: A dictionary containing the number of copies for each read.
    """
    read_copies = {}
    for read in reads:
        read_copies[read] = read_copies.get(read, 0) + 1

    overlap_graph = nx.DiGraph()

    # Add nodes to G and handle copies
    for read, count in read_copies.items():
        for copy_index in range(count):
            node_name = f"{read}_{copy_index}"
            overlap_graph.add_node(node_name)

    # Find overlaps and add edges
    for read_a, count_a in read_copies.items():
        for read_b, count_b in read_copies.items():
            if read_a != read_b:
                to_print, aligned_source, aligned_target, score, alignment_end_position = overlap_alignment(read_a, read_b, mismatch=-1000000) #TODO - remove, its just for small example

                if score > 0:  # Only add edges for significant overlaps
                    for copy_index_a in range(count_a):
                        for copy_index_b in range(count_b):
                            node_a_name = f"{read_a}_{copy_index_a}"
                            node_b_name = f"{read_b}_{copy_index_b}"
                            overlap_graph.add_edge(node_a_name, node_b_name, weight=score, end_position=alignment_end_position)

    return overlap_graph, read_copies


def transitive_reduction(overlap_graph, fuzz=10):
    """
    Performs transitive reduction on the overlap graph.

    Parameters:
        overlap_graph (nx.DiGraph): The overlap graph.
        fuzz (int): The fuzz factor for length comparisons.

    Returns:
        nx.DiGraph: The reduced overlap graph.
    """
    G = overlap_graph.copy()
    mark = {node: "vacant" for node in G.nodes()}
    reduce_edges = {(u, v): False for u, v in G.edges()}

    for v in G.nodes():
        print(f"Processing node: {v}")

        for u, w in G.out_edges(v):
            print(f"u: {u}, w: {w}")
            mark[w] = "inplay"

        for v_to_w in G.out_edges(v):
            print(f"v_to_w: {v_to_w}")
            w = v_to_w[1]
            print(f"v_to_w[1]: {v_to_w[1]}")

            for w_to_x in G.out_edges(w):
                print(f"w_to_x: {w_to_x}")
                x = w_to_x[1]
                print(f"w_to_x[1]: {w_to_x[1]}")
                if G.has_edge(v, x) and mark[x] == "inplay":
                    if G[w][x]['weight'] + G[v][w]['weight'] >= G[v][x]['weight']:
                        mark[x] = "eliminated"
                elif mark[x] == "inplay":
                    mark[x] = "eliminated"

        for v_to_w in G.out_edges(v):
            print(f"v_to_w: {v_to_w}")
            w = v_to_w[1]
            print(f"v_to_w[1]: {v_to_w[1]}")
            for w_to_x in G.out_edges(w):
                print(f"w_to_x: {w_to_x}")
                x = w_to_x[1]
                print(f"w_to_x[1]: {w_to_x[1]}")
                if G.has_edge(v, x) and mark[x] == "inplay":
                    if G[w][x]['weight'] + G[v][w]['weight'] >= G[v][x]['weight']:
                        mark[x] = "eliminated"
                        print(f"marking {x} as eliminated")
                elif mark[x] == "inplay":
                    mark[x] = "eliminated"
                    print(f"marking {x} as eliminated")

        for v_to_w in G.out_edges(v):
            print(f"v_to_w: {v_to_w}")
            w = v_to_w[1]
            print(f"v_to_w[1]: {v_to_w[1]}")
            if mark[w] == "eliminated":
                print(f"marking edge {v_to_w} as reduced")
                reduce_edges[(v_to_w[0],v_to_w[1])] = True
            mark[w] = "vacant"
            print(f"marking {w} as vacant")

    for u, v in reduce_edges:
        if reduce_edges[(u,v)]:
            print(f"Removing edge: {u} -> {v}")
            G.remove_edge(u, v)

    return G


def assemble_contigs_string(reads, fuzz=5):
    """
    Assembles contigs using the string graph approach.

    Parameters:
        reads (list): List of DNA reads.

    Returns:
        list: List of assembled contigs.
    """
    overlap_graph, read_copies = construct_overlap_graph_string(reads)
    print(f"overlap_graph: {overlap_graph.edges}")
    reduced_graph = transitive_reduction(overlap_graph)
    print(f"reduced_graph: {reduced_graph.edges}")
    visited = set()
    contigs = []

    for node in reduced_graph.nodes():
        read_base = node.split("_")[0]
        if read_base not in visited:
            contig = create_contig(node, reduced_graph, visited, {}) #topo_order is not used in the paper's algorithm.
            contigs.append(contig)

    return contigs


def construct_string_graph(reads):
    """
    Constructs a string graph where nodes are reads and edges represent overlaps
    scored by overlap alignment.
    """
    graph = nx.DiGraph()

    # Add nodes for each read
    for read in reads:
        graph.add_node(read)

    # Compute overlaps using overlap alignment
    for read_a, read_b in combinations(reads, 2):
        to_print, aligned_source, aligned_target, score, end_position = overlap_alignment(read_a, read_b, mismatch=-1000000) #TODO - remove, its just for small example

        if score > 0:  # Ensure there is a significant overlap
            graph.add_edge(read_a, read_b, weight=score, end_position=end_position)

    print(f"graph: {graph.edges}")
    return graph


def transitive_reduction2(graph):
    """
    Perform transitive reduction to remove redundant edges.
    """
    reduced_graph = graph.copy()

    for v in graph.nodes():
        for u, w in combinations(graph.successors(v), 2):
            if nx.has_path(graph, u, w):
                if reduced_graph.has_edge(v, w):
                    reduced_graph.remove_edge(v, w)
                    print(f"Removing edge: {v} -> {w}")

    return reduced_graph


def find_unitigs(graph):
    """
    Collapse non-branching paths into unitigs.
    """
    unitigs = []
    visited = set()

    for node in graph.nodes():
        if node in visited:
            continue

        # Extend forward
        path = [node]
        while len(list(graph.successors(path[-1]))) == 1 and len(list(graph.predecessors(path[-1]))) == 1:
            next_node = list(graph.successors(path[-1]))[0]
            print(f"next_node: {next_node}")
            if next_node in visited:
                break
            path.append(next_node)

        # Mark nodes as visited
        for n in path:
            visited.add(n)

        # Merge unitig sequence
        unitig_seq = path[0]
        for i in range(1, len(path)):
            overlap_len = graph.edges[path[i - 1], path[i]]['end_position']
            unitig_seq += path[i][overlap_len:]

        unitigs.append(unitig_seq)

    return unitigs


def assemble_contigs(reads):
    """
    Full pipeline for genome assembly using the Fragment Assembly String Graph approach.
    """
    graph = construct_string_graph(reads)
    reduced_graph = transitive_reduction2(graph)
    contigs = find_unitigs(reduced_graph)
    return contigs



if __name__ == "__main__":
    genome = "ATGCGTACGTTAGCACGTGTTCGATAGC"

    # generate 10 reads of length 5 of the genome
    reads = [genome[i:i + 5] for i in range(0, len(genome) - 5)]

    import random

    # chose random 10 reads
    reads = ['TGTTC', 'TGCGT', 'ACGTG', 'CACGT', 'AGCAC', 'GATAG', 'CGATA', 'GTACG', 'CGTAC', 'ATGCG']#random.sample(reads, 10)
    """reads = [x6'TGTTC', x2'TGCGT', x5'ACGTG', 'CACGT', 'AGCAC', 'GATAG', %x7%'CGATA', x4'GTACG', x3'CGTAC', x1'ATGCG']#random.sample(reads, 10)"""


    from performanceMeasures import calculate_measures_old, calculate_measures

    contigs = assemble_contigs_using_overlap_graphs(reads, k=0)
    res = calculate_measures_old(contigs, reads, len(reads), len(reads[0]), 0, genome, "experiment_name", 1, "path")
    print(f"genome: {genome}")
    print(f"reads: {reads}")
    print(f"{len(contigs)} Assembled contigs: {contigs}")
    print(f"Results: {res}")
    print("============")

    contigs_paper = assemble_contigs_string(reads)
    res_paper = calculate_measures(contigs_paper, reads, len(reads), len(reads[0]), 0, genome, "experiment_name_paper", 1, "path")
    print(f"genome: {genome}")
    print(f"reads: {reads}")
    print(f"{len(contigs_paper)} Assembled contigs: {contigs_paper}")
    print(f"Results: {res_paper}")
    print("============")

    contigs_chat = assemble_contigs(reads)
    res_chat = calculate_measures(contigs_chat, reads, len(reads), len(reads[0]), 0, genome, "experiment_name_chat", 1, "path")
    print(f"genome: {genome}")
    print(f"reads: {reads}")
    print(f"{len(contigs_chat)} Assembled contigs: {contigs_chat}")
    print(f"Results: {res_chat}")
    print("============")
