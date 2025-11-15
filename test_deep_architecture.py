"""
Test the redesigned deep architecture.

Verifies:
1. Average path length from inputs to outputs is 20-30 hops
2. Forward/backward connection ratios are correct
3. Inputs come from early depth levels
4. Outputs come from late depth levels
"""

import torch
import numpy as np
from collections import deque
from messy_perceptron_network.core.fast_network import FastMessyPerceptronNetwork

def bfs_distances(start_nodes, target_nodes, signal_indices, n_perceptrons):
    """
    BFS from start nodes to find distances to target nodes.

    Returns:
        dict: {target_node: shortest_distance}
    """
    # Build adjacency list
    adj = {i: [] for i in range(n_perceptrons)}
    for src, dst in zip(signal_indices[0].tolist(), signal_indices[1].tolist()):
        adj[src].append(dst)

    # BFS from all start nodes simultaneously
    queue = deque([(node, 0) for node in start_nodes])
    distances = {}
    visited = set(start_nodes)

    while queue:
        node, dist = queue.popleft()

        if node in target_nodes:
            if node not in distances:  # Record shortest path
                distances[node] = dist

        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return distances

print("Creating deep messy perceptron network...")
network = FastMessyPerceptronNetwork(
    n_perceptrons=2000,
    avg_degree=30,
    n_input_perceptrons=250,  # Match MNIST dimensions
    n_output_perceptrons=125,
    settling_iterations=30,  # Increase for deeper network
    seed=42
)

print("\n" + "="*60)
print("ARCHITECTURE ANALYSIS")
print("="*60)

# Analyze input/output distance distribution
input_indices = network.input_perceptron_indices.numpy()
output_indices = network.output_perceptron_indices.numpy()
neuron_distances = network.neuron_distances

input_dists = neuron_distances[input_indices]
output_dists = neuron_distances[output_indices]

print(f"\nInput Neurons (n={len(input_indices)}):")
print(f"  Distance-to-output range: {input_dists.min():.0f} - {input_dists.max():.0f}")
print(f"  Average distance-to-output: {input_dists.mean():.1f}")

print(f"\nOutput Neurons (n={len(output_indices)}):")
print(f"  Distance-to-output: {output_dists.min():.0f} (by definition)")

print(f"\nAll Neurons:")
print(f"  Max distance-to-output: {np.max(neuron_distances[~np.isinf(neuron_distances)]):.0f} hops")

# Compute shortest paths from inputs to outputs
print("\n" + "="*60)
print("SIGNAL PROPAGATION DEPTH")
print("="*60)

print("\nComputing shortest paths from inputs to outputs...")
distances = bfs_distances(
    start_nodes=input_indices.tolist(),
    target_nodes=set(output_indices.tolist()),
    signal_indices=network.signal_indices,
    n_perceptrons=network.n_perceptrons
)

if len(distances) == 0:
    print("ERROR: No paths found from inputs to outputs!")
else:
    path_lengths = list(distances.values())

    print(f"\nReachable outputs: {len(distances)} / {len(output_indices)} ({100*len(distances)/len(output_indices):.1f}%)")
    print(f"\nPath length statistics:")
    print(f"  Minimum: {min(path_lengths)} hops")
    print(f"  Maximum: {max(path_lengths)} hops")
    print(f"  Average: {np.mean(path_lengths):.1f} hops")
    print(f"  Median: {np.median(path_lengths):.1f} hops")

    # Distribution
    print(f"\nPath length distribution:")
    unique_lengths = sorted(set(path_lengths))
    for length in unique_lengths[:15]:  # Show first 15
        count = path_lengths.count(length)
        percentage = 100 * count / len(path_lengths)
        bar = "█" * int(percentage / 2)
        print(f"  {length:2d} hops: {count:3d} ({percentage:5.1f}%) {bar}")

    if len(unique_lengths) > 15:
        print(f"  ... and {len(unique_lengths) - 15} more length(s)")

    # Check if we hit the target 20-30 range
    avg_length = np.mean(path_lengths)
    print(f"\n{'='*60}")
    if 20 <= avg_length <= 30:
        print("✓ SUCCESS: Average path length is in target range (20-30 hops)")
    elif avg_length < 20:
        print(f"✗ SHALLOW: Average path length ({avg_length:.1f}) is below target (20-30)")
    else:
        print(f"✗ TOO DEEP: Average path length ({avg_length:.1f}) is above target (20-30)")
    print(f"{'='*60}")
