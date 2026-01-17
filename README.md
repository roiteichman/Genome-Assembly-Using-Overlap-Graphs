# Genome Assembly using Overlap Graphs

A Python-based simulation framework for **De Novo Genome Assembly** using the Overlap-Layout-Consensus (OLC) approach. This project implements a full assembly pipeline-from synthetic read generation (with stochastic errors) to graph construction and contig assembly—designed to evaluate algorithmic robustness under varying sequencing conditions.

## 🧬 Project Overview

This repository contains an implementation of a genome assembly engine that reconstructs a target genome (e.g., PhiX) from short reads. The system is built to analyze the trade-offs between coverage, read length, and sequencing error rates.

Key capabilities include:
* **Synthetic Data Generation:** Simulates Next-Generation Sequencing (NGS) reads with configurable error probabilities.
* **Graph-Based Assembly:** Constructs an overlap graph where nodes represent reads and edges represent overlaps.
* **Optimization:** Utilizes **K-mer indexing** to prune the search space and accelerate graph construction.
* **High-Performance Computing:** Leverages **Numba** for JIT-compiled alignment algorithms and **Joblib** for parallelized experiment execution.
* **Comprehensive Evaluation:** Automatically calculates N50, Genome Coverage, and Mismatch Rates.

## 🛠️ Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/genome-assembly-overlap-graphs.git](https://github.com/your-username/genome-assembly-overlap-graphs.git)
    cd genome-assembly-overlap-graphs
    ```

2.  Install the required dependencies:
    ```bash
    pip install numpy pandas networkx matplotlib biopython numba joblib
    ```

## 🚀 Usage

1.  **Prepare Reference Genome:**
    Place your reference genome file (FASTA format) in the root directory and name it `sequence.fasta`.

2.  **Run Experiments:**
    Execute the main script to run the full simulation pipeline. This will generate reads, run the assembly, and save the metrics.
    ```bash
    python experiments.py
    ```

3.  **View Results:**
    * **Metrics:** CSV files containing N50, coverage, and error rates will be saved in the `results/` directory.
    * **Visualizations:** Plots analyzing the impact of parameters ($N$, $l$, $p$) on assembly quality will be saved in `temp_plots/`.

## 📂 Code Structure

* `experiments.py`: Main entry point. Manages parallel execution of experiments across different parameter sweeps.
* `overlapGraphs.py`: Core logic for building the Overlap Graph using NetworkX and K-mer filtering.
* `aligners.py`: **Numba-optimized** pairwise alignment functions (Needleman-Wunsch variation) to detect overlaps.
* `generateErrorProneReads.py`: Simulates sequencing noise by introducing mutations/errors into the reads.
* `testAssembly.py`: Runs a single assembly iteration and computes performance statistics.
* `performanceMeasures.py`: Calculates evaluation metrics (N50, Genome Coverage, Mismatch Rate).

## 📊 Methodology

The assembly pipeline follows these steps:
1.  **Read Generation:** The system samples uniform reads from the reference genome and introduces noise based on a defined error probability ($p$).
2.  **Overlap Detection:** Calculates overlap scores between reads. To maintain efficiency, a K-mer index pre-filters potential candidates.
3.  **Graph Construction:** Builds a directed graph where edges are weighted by overlap scores.
4.  **Layout & Consensus:** Traverses the graph (removing cycles and using a greedy approach) to merge reads into contigs.
5.  **Validation:** Maps the assembled contigs back to the reference genome to verify accuracy and compute robustness metrics.

## 📄 Research Paper

For a detailed analysis of the algorithmic robustness and reconstruction accuracy under stochastic errors, please refer to the full report:
[**View Research Report (PDF)**](./Genome%20Assembly%20Using%20Overlap%20Graphs%20-%20Final%20Report%20-%20Roi%20Teichman.pdf)

---
*Created by Roi Teichman as part of the "Algorithms in Computational Biology" course at the Technion.*
