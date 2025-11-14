"""
Analyze multi-timescale learning from loop topology.

Check if perceptrons in different length loops actually exhibit different
learning dynamics (gradient magnitudes, update rates).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from messy_perceptron_network.core.fast_network_modulated import FastMessyPerceptronNetwork


def analyze_perceptron_loops(network):
    """Identify which perceptrons are in short vs long loops."""
    print("\nAnalyzing loop structure...")

    # Build adjacency list
    adj_list = defaultdict(list)
    for src, dst in zip(network.signal_indices[0].numpy(), network.signal_indices[1].numpy()):
        adj_list[src].append(dst)

    # Find loops for each perceptron
    perceptron_loop_lengths = defaultdict(list)

    for start in range(network.n_perceptrons):
        # DFS to find paths back to start
        visited_in_path = set()

        def dfs(node, path_length, max_depth=35):
            if path_length > max_depth:
                return
            if node == start and path_length > 0:
                # Found a loop!
                perceptron_loop_lengths[start].append(path_length)
                return
            if node in visited_in_path:
                return

            visited_in_path.add(node)
            for neighbor in adj_list[node]:
                dfs(neighbor, path_length + 1, max_depth)
            visited_in_path.remove(node)

        dfs(start, 0)

    # Categorize perceptrons
    short_loop_perceptrons = []  # In loops 2-5
    medium_loop_perceptrons = []  # In loops 6-15
    long_loop_perceptrons = []  # In loops 16-30+
    no_loop_perceptrons = []  # Not in any loops

    for p in range(network.n_perceptrons):
        loops = perceptron_loop_lengths[p]
        if not loops:
            no_loop_perceptrons.append(p)
        else:
            min_loop = min(loops)
            if min_loop <= 5:
                short_loop_perceptrons.append(p)
            elif min_loop <= 15:
                medium_loop_perceptrons.append(p)
            else:
                long_loop_perceptrons.append(p)

    print(f"Perceptron loop categorization:")
    print(f"  Short loops (2-5): {len(short_loop_perceptrons)} perceptrons")
    print(f"  Medium loops (6-15): {len(medium_loop_perceptrons)} perceptrons")
    print(f"  Long loops (16-30+): {len(long_loop_perceptrons)} perceptrons")
    print(f"  No loops: {len(no_loop_perceptrons)} perceptrons")

    return {
        'short': short_loop_perceptrons,
        'medium': medium_loop_perceptrons,
        'long': long_loop_perceptrons,
        'none': no_loop_perceptrons
    }


def measure_gradient_by_loop_type(network, dataloader, device='cpu'):
    """Measure gradient magnitudes for perceptrons in different loop types."""

    # Get loop categorization
    loop_categories = analyze_perceptron_loops(network)

    # Create simple classifier
    class SimpleClassifier(nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network
            self.input_projection = nn.Linear(784, network.n_input_perceptrons)
            self.output_layer = nn.Linear(network.n_output_perceptrons, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            projected = self.input_projection(x)
            network_output = self.network(projected)
            return self.output_layer(network_output)

    classifier = SimpleClassifier(network).to(device)
    criterion = nn.CrossEntropyLoss()

    # Collect gradients over a few batches
    gradient_stats = {
        'short': [],
        'medium': [],
        'long': [],
        'none': []
    }

    print("\nMeasuring gradients over 10 batches...")
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        if batch_idx >= 10:
            break

        inputs = inputs.to(device)
        labels = labels.to(device)

        classifier.zero_grad()
        outputs = classifier(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Measure threshold gradients (per-perceptron parameter)
        if network.thresholds.grad is not None:
            thresh_grads = network.thresholds.grad.abs().cpu().numpy()

            for category, perceptron_list in loop_categories.items():
                if perceptron_list:
                    category_grads = thresh_grads[perceptron_list]
                    gradient_stats[category].append(category_grads.mean())

    # Print results
    print("\nGradient magnitude by loop type:")
    for category in ['short', 'medium', 'long', 'none']:
        if gradient_stats[category]:
            mean_grad = np.mean(gradient_stats[category])
            std_grad = np.std(gradient_stats[category])
            print(f"  {category:6s} loops: {mean_grad:.6f} ± {std_grad:.6f}")

    # Check if there's differentiation
    short_mean = np.mean(gradient_stats['short']) if gradient_stats['short'] else 0
    long_mean = np.mean(gradient_stats['long']) if gradient_stats['long'] else 0

    if long_mean > 0:
        ratio = short_mean / long_mean
        print(f"\nShort/Long gradient ratio: {ratio:.2f}x")
        if ratio > 1.5:
            print("✓ Multi-timescale learning IS happening (short loops update faster)")
        elif ratio < 0.67:
            print("✓ Inverse pattern (long loops update faster - unexpected)")
        else:
            print("✗ NO differentiation (gradients similar regardless of loop length)")
    else:
        print("✗ Cannot compute ratio (no long loops or zero gradients)")

    return gradient_stats


if __name__ == "__main__":
    # Create network
    print("Creating network...")
    network = FastMessyPerceptronNetwork(
        n_perceptrons=500,
        avg_degree=15,
        n_input_perceptrons=250,
        n_output_perceptrons=125,
        settling_iterations=7,
        seed=42
    )

    # Load MNIST
    print("\nLoading MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Analyze
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    gradient_stats = measure_gradient_by_loop_type(network, dataloader, device)
