"""Analyze the actual structure and behavior of the messy network."""

import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from messy_perceptron_network.core.fast_network import FastMessyPerceptronNetwork
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def analyze_path_depth(network):
    """Analyze how far information propagates through the network."""
    print("\n" + "="*70)
    print("PATH DEPTH ANALYSIS")
    print("="*70)

    # Build adjacency list for signal connections
    adj = {i: [] for i in range(network.n_perceptrons)}
    signal_edges = list(zip(network.signal_indices[0].tolist(),
                           network.signal_indices[1].tolist()))

    for src, dst in signal_edges:
        adj[src].append(dst)

    # Compute average shortest path from input to output neurons
    input_indices = network.input_perceptron_indices.tolist()
    output_indices = network.output_perceptron_indices.tolist()

    path_lengths = []

    def bfs_distance(start, targets):
        """BFS to find shortest distance to any target."""
        visited = {start: 0}
        queue = [start]
        head = 0

        while head < len(queue):
            node = queue[head]
            head += 1
            depth = visited[node]

            if node in targets:
                return depth

            if depth >= 30:  # Max search depth
                continue

            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited[neighbor] = depth + 1
                    queue.append(neighbor)

        return None  # No path found

    # Sample subset for performance
    sample_inputs = min(50, len(input_indices))
    sample_outputs = min(50, len(output_indices))

    for inp in input_indices[:sample_inputs]:
        dist = bfs_distance(inp, set(output_indices[:sample_outputs]))
        if dist is not None:
            path_lengths.append(dist)

    if path_lengths:
        print(f"Average shortest path (input→output): {np.mean(path_lengths):.2f} steps")
        print(f"Min: {min(path_lengths)}, Max: {max(path_lengths)}")
        print(f"With {network.settling_iterations} settling iterations, effective depth = "
              f"{network.settling_iterations} iterations × multiple paths")
    else:
        print("No direct paths found from inputs to outputs!")

    # Analyze reachability in 7 steps
    def count_reachable(start, max_depth=7):
        """Count how many neurons reachable in max_depth steps."""
        visited = {start}
        current_level = [start]

        for depth in range(max_depth):
            next_level = []
            for node in current_level:
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.append(neighbor)
            current_level = next_level

        return len(visited)

    # Sample input neurons
    reachable_counts = []
    for inp in input_indices[:20]:
        count = count_reachable(inp, network.settling_iterations)
        reachable_counts.append(count)

    print(f"\nNeurons reachable from inputs in {network.settling_iterations} iterations:")
    print(f"  Average: {np.mean(reachable_counts):.1f} / {network.n_perceptrons} ({100*np.mean(reachable_counts)/network.n_perceptrons:.1f}%)")
    print(f"  Min: {min(reachable_counts)}, Max: {max(reachable_counts)}")


def analyze_activation_diversity(network, data_loader, n_samples=500):
    """Analyze activation diversity across neurons."""
    print("\n" + "="*70)
    print("ACTIVATION DIVERSITY ANALYSIS")
    print("="*70)

    network.eval()
    device = next(network.parameters()).device

    # Collect activations across multiple samples
    all_activations = []

    with torch.no_grad():
        count = 0
        for batch_x, _ in data_loader:
            if count >= n_samples:
                break

            batch_x = batch_x.view(batch_x.size(0), -1).to(device)

            # Get activations during forward pass
            outputs, history = network(batch_x, return_history=True)

            # Use final settling iteration activations
            final_activations = history[-1]  # Shape: (batch, n_perceptrons)
            all_activations.append(final_activations.cpu())

            count += batch_x.size(0)

    # Concatenate all activations
    all_activations = torch.cat(all_activations, dim=0)  # Shape: (n_samples, n_perceptrons)

    print(f"Collected activations from {all_activations.size(0)} samples")

    # For each neuron, compute activation diversity metrics
    diversities = []
    sparsities = []

    for neuron_idx in range(network.n_perceptrons):
        activations = all_activations[:, neuron_idx]

        # Diversity metric 1: Standard deviation (high = responds to many patterns)
        diversity = activations.std().item()
        diversities.append(diversity)

        # Sparsity: % of times neuron is active (|activation| > 0.1)
        active_count = (activations.abs() > 0.1).sum().item()
        sparsity = active_count / len(activations)
        sparsities.append(sparsity)

    diversities = np.array(diversities)
    sparsities = np.array(sparsities)

    print(f"\nActivation Diversity (std dev):")
    print(f"  Mean: {diversities.mean():.4f}")
    print(f"  Std: {diversities.std():.4f}")
    print(f"  Min: {diversities.min():.4f}, Max: {diversities.max():.4f}")
    print(f"  Coefficient of Variation: {diversities.std()/diversities.mean():.3f}")

    print(f"\nActivation Sparsity (% samples active):")
    print(f"  Mean: {sparsities.mean():.2%}")
    print(f"  Std: {sparsities.std():.2%}")

    # Check if there's hierarchical structure
    # High CV in diversity = some neurons are "foundational" (high diversity), others specialized
    cv = diversities.std() / diversities.mean()

    print(f"\n{'='*70}")
    print("INTERPRETATION:")
    print("="*70)

    if cv < 0.3:
        print("⚠ LOW diversity variation - neurons are uniformly diverse")
        print("  → Network is a 'flat recurrent soup', no clear hierarchy")
        print("  → Activation-based consolidation may NOT help")
    elif cv < 0.7:
        print("✓ MODERATE diversity variation - some hierarchy emerging")
        print("  → Some neurons more foundational than others")
        print("  → Activation-based consolidation MIGHT help")
    else:
        print("✓✓ HIGH diversity variation - clear hierarchical structure")
        print("  → Strong separation of foundational vs specialized neurons")
        print("  → Activation-based consolidation SHOULD help significantly")

    # Identify top foundational neurons (high diversity)
    top_indices = np.argsort(diversities)[-10:][::-1]
    print(f"\nTop 10 most diverse (foundational) neurons:")
    for i, idx in enumerate(top_indices, 1):
        is_input = idx in network.input_perceptron_indices
        is_output = idx in network.output_perceptron_indices
        role = "INPUT" if is_input else ("OUTPUT" if is_output else "HIDDEN")
        print(f"  {i}. Neuron {idx} ({role}): diversity={diversities[idx]:.4f}, active={sparsities[idx]:.1%}")


def main():
    print("Creating network...")
    network = FastMessyPerceptronNetwork(
        n_perceptrons=500,
        avg_degree=15,
        n_input_perceptrons=250,
        n_output_perceptrons=125,
        settling_iterations=7,
        seed=42
    )

    # Analyze structure
    analyze_path_depth(network)

    # Load MNIST for activation analysis
    print("\nLoading MNIST data...")
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Analyze activations
    analyze_activation_diversity(network, train_loader, n_samples=500)


if __name__ == '__main__':
    main()
