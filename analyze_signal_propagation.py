"""Analyze how far signals propagate per settling iteration."""

import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from collections import deque
from messy_perceptron_network.core.fast_network import FastMessyPerceptronNetwork

def analyze_propagation_depth(network):
    """Analyze information flow depth in the network."""
    print("="*70)
    print("SIGNAL PROPAGATION ANALYSIS")
    print("="*70)

    # Build adjacency list from signal connections
    adj = {i: [] for i in range(network.n_perceptrons)}
    signal_edges = list(zip(network.signal_indices[0].tolist(),
                           network.signal_indices[1].tolist()))

    for src, dst in signal_edges:
        adj[src].append(dst)

    input_indices = set(network.input_perceptron_indices.tolist())
    output_indices = set(network.output_perceptron_indices.tolist())

    print(f"\nNetwork structure:")
    print(f"  Total neurons: {network.n_perceptrons}")
    print(f"  Input neurons: {len(input_indices)} (indices {min(input_indices)}-{max(input_indices)})")
    print(f"  Output neurons: {len(output_indices)} (indices {min(output_indices)}-{max(output_indices)})")
    print(f"  Signal connections: {len(signal_edges)}")
    print(f"  Settling iterations: {network.settling_iterations}")

    # BFS to measure shortest paths from inputs to outputs
    def bfs_distances(start_nodes, target_nodes):
        """BFS from multiple start nodes to find distances to targets."""
        distances = []

        for start in start_nodes:
            visited = {start: 0}
            queue = deque([start])

            while queue:
                node = queue.popleft()
                depth = visited[node]

                if depth >= 50:  # Max search depth
                    break

                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited[neighbor] = depth + 1
                        queue.append(neighbor)

                        if neighbor in target_nodes:
                            distances.append(depth + 1)

        return distances

    # Compute shortest paths from ALL inputs to ALL outputs
    print(f"\nComputing shortest paths from inputs to outputs...")
    sample_inputs = list(input_indices)[:50]  # Sample for performance
    sample_outputs = list(output_indices)

    distances = bfs_distances(sample_inputs, sample_outputs)

    if distances:
        print(f"\nShortest path statistics (input → output):")
        print(f"  Paths found: {len(distances)}")
        print(f"  Min hops: {min(distances)}")
        print(f"  Max hops: {max(distances)}")
        print(f"  Mean hops: {np.mean(distances):.1f}")
        print(f"  Median hops: {np.median(distances):.1f}")
        print(f"  25th percentile: {np.percentile(distances, 25):.1f}")
        print(f"  75th percentile: {np.percentile(distances, 75):.1f}")

        # Distribution
        hop_counts = {}
        for d in distances:
            hop_counts[d] = hop_counts.get(d, 0) + 1

        print(f"\n  Path length distribution:")
        for hops in sorted(hop_counts.keys())[:15]:  # Show first 15
            count = hop_counts[hops]
            pct = 100 * count / len(distances)
            bar = "█" * int(pct / 2)
            print(f"    {hops:2d} hops: {bar} {pct:5.1f}%")

        # Compare to settling iterations
        coverage = sum(1 for d in distances if d <= network.settling_iterations)
        coverage_pct = 100 * coverage / len(distances)

        print(f"\n{'='*70}")
        print(f"SETTLING ITERATION COVERAGE:")
        print(f"{'='*70}")
        print(f"Paths reachable in {network.settling_iterations} iterations: {coverage}/{len(distances)} ({coverage_pct:.1f}%)")

        if coverage_pct < 50:
            print(f"⚠ CRITICAL: Less than 50% of paths covered!")
            print(f"  Many outputs can't receive input signals in time")
            needed = int(np.percentile(distances, 90))
            print(f"  Recommend: {needed} settling iterations for 90% coverage")
        elif coverage_pct < 80:
            print(f"⚠ WARNING: Less than 80% of paths covered")
            needed = int(np.percentile(distances, 90))
            print(f"  Recommend: {needed} settling iterations for 90% coverage")
        elif coverage_pct < 95:
            print(f"✓ ACCEPTABLE: {coverage_pct:.1f}% coverage")
            needed = int(np.percentile(distances, 95))
            print(f"  Consider: {needed} iterations for 95% coverage")
        else:
            print(f"✓✓ EXCELLENT: {coverage_pct:.1f}% coverage")

        # Analyze per-iteration reachability
        print(f"\n{'='*70}")
        print(f"CUMULATIVE REACHABILITY BY ITERATION:")
        print(f"{'='*70}")

        for iters in range(1, min(network.settling_iterations + 10, 30)):
            reachable = sum(1 for d in distances if d <= iters)
            pct = 100 * reachable / len(distances)
            print(f"  After {iters:2d} iterations: {pct:5.1f}% of output neurons reachable")

    else:
        print("⚠ NO PATHS FOUND from inputs to outputs!")
        print("  Network may be disconnected or graph structure issue")

    # Analyze what happens in ONE iteration (parallel propagation)
    print(f"\n{'='*70}")
    print(f"WHAT HAPPENS IN ONE ITERATION:")
    print(f"{'='*70}")
    print(f"In each iteration, ALL neurons update IN PARALLEL:")
    print(f"  1. Each neuron receives signals from ALL its {network.signal_indices.shape[1] // network.n_perceptrons:.1f} inputs (avg)")
    print(f"  2. Sparse matrix multiply: output = A_signal @ activations")
    print(f"  3. All neurons compute new activations simultaneously")
    print(f"  4. Information propagates ONE HOP through the graph")
    print(f"\nThis is NOT one neuron at a time - it's one graph hop at a time.")
    print(f"With {network.settling_iterations} iterations, signals travel {network.settling_iterations} hops maximum.")

def main():
    print("\nCreating network with FIXED I/O (non-overlapping)...")
    network = FastMessyPerceptronNetwork(
        n_perceptrons=500,
        avg_degree=15,
        n_input_perceptrons=250,
        n_output_perceptrons=125,
        settling_iterations=7,
        seed=42
    )

    analyze_propagation_depth(network)

if __name__ == '__main__':
    main()
