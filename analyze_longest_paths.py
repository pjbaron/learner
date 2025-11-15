"""
Analyze longest paths in the recurrent network.

For recurrent networks, we need settling iterations >= longest path length
where longest path includes following backward loops once.
"""

import torch
import numpy as np
from collections import deque
from messy_perceptron_network.core.fast_network import FastMessyPerceptronNetwork

def find_longest_paths_with_loops(network, max_depth=100):
    """
    Find longest simple paths from inputs to outputs.

    Uses DFS to explore paths, allowing backward jumps through recurrent connections.
    """
    # Build adjacency list from signal connections
    adj = {i: [] for i in range(network.n_perceptrons)}
    signal_indices = network.signal_indices.numpy()

    for src, dst in zip(signal_indices[0], signal_indices[1]):
        adj[src].append(dst)

    input_indices = network.input_perceptron_indices.numpy()
    output_indices = set(network.output_perceptron_indices.numpy().tolist())
    neuron_distances = network.neuron_distances

    longest_paths = []

    def dfs(node, path_length, visited):
        """DFS to find longest paths to outputs."""
        if node in output_indices:
            return path_length

        if path_length >= max_depth:
            return 0  # Avoid infinite loops

        max_length = 0
        for neighbor in adj[node]:
            if neighbor not in visited:  # Simple path (no revisiting)
                visited.add(neighbor)
                length = dfs(neighbor, path_length + 1, visited)
                max_length = max(max_length, length)
                visited.remove(neighbor)

        return max_length

    # Sample inputs to avoid expensive full search
    sample_inputs = input_indices[:min(50, len(input_indices))]

    print("Analyzing longest paths from sample of inputs...")
    for input_idx in sample_inputs:
        visited = {input_idx}
        longest = dfs(input_idx, 0, visited)
        if longest > 0:
            longest_paths.append(longest)

    return longest_paths

print("Creating deep messy perceptron network...")
network = FastMessyPerceptronNetwork(
    n_perceptrons=2000,
    avg_degree=30,
    n_input_perceptrons=250,
    n_output_perceptrons=125,
    settling_iterations=30,
    seed=42
)

print("\n" + "="*60)
print("LONGEST PATH ANALYSIS")
print("="*60)

longest_paths = find_longest_paths_with_loops(network, max_depth=100)

if longest_paths:
    print(f"\nAnalyzed {len(longest_paths)} paths from input samples:")
    print(f"  Shortest of sampled paths: {min(longest_paths)} hops")
    print(f"  Longest of sampled paths: {max(longest_paths)} hops")
    print(f"  Average: {np.mean(longest_paths):.1f} hops")
    print(f"  Median: {np.median(longest_paths):.1f} hops")

    print(f"\n{'='*60}")
    print(f"SETTLING ITERATIONS CHECK")
    print(f"{'='*60}")
    print(f"Current settling iterations: {network.settling_iterations}")
    print(f"Longest path found: {max(longest_paths)}")

    if network.settling_iterations >= max(longest_paths):
        print(f"✓ SUFFICIENT: Settling iterations cover longest paths")
    else:
        print(f"✗ INSUFFICIENT: Need at least {max(longest_paths)} iterations")
        print(f"  Recommend: {max(longest_paths) + 5} iterations (with buffer)")
else:
    print("No paths found from inputs to outputs!")

# Also show distance distribution
print(f"\n{'='*60}")
print(f"DISTANCE-TO-OUTPUT STATISTICS")
print(f"{'='*60}")

input_indices = network.input_perceptron_indices.numpy()
neuron_distances = network.neuron_distances

input_dists = neuron_distances[input_indices]
print(f"Input neuron distances: {input_dists.min():.0f} - {input_dists.max():.0f}")
print(f"Average input distance: {input_dists.mean():.1f}")
print(f"Max distance in network: {np.max(neuron_distances[~np.isinf(neuron_distances)]):.0f}")
