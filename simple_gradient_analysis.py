"""
Simple analysis: Do gradients differ across perceptrons in a way that would
support multi-timescale learning?
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from messy_perceptron_network.core.fast_network_modulated import FastMessyPerceptronNetwork


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


def analyze_gradient_distribution(network, dataloader, device='cpu'):
    """Check if gradients vary across perceptrons (multi-timescale signal)."""

    classifier = SimpleClassifier(network).to(device)
    criterion = nn.CrossEntropyLoss()

    print("\nAnalyzing gradient distribution across perceptrons...\n")

    # Collect gradients over several batches
    all_threshold_grads = []
    all_signal_grads = []

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        if batch_idx >= 20:
            break

        inputs = inputs.to(device)
        labels = labels.to(device)

        classifier.zero_grad()
        outputs = classifier(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Per-perceptron gradients (thresholds)
        if network.thresholds.grad is not None:
            all_threshold_grads.append(network.thresholds.grad.abs().cpu().numpy())

        # Per-connection gradients (signal weights)
        if network.signal_weights.grad is not None:
            all_signal_grads.append(network.signal_weights.grad.abs().cpu().numpy())

    # Average across batches
    avg_threshold_grads = np.mean(all_threshold_grads, axis=0)  # (n_perceptrons,)
    avg_signal_grads = np.mean(all_signal_grads, axis=0)  # (n_connections,)

    print("Perceptron gradient statistics (thresholds):")
    print(f"  Mean:   {avg_threshold_grads.mean():.6f}")
    print(f"  Std:    {avg_threshold_grads.std():.6f}")
    print(f"  Min:    {avg_threshold_grads.min():.6f}")
    print(f"  Max:    {avg_threshold_grads.max():.6f}")
    print(f"  Coefficient of Variation: {avg_threshold_grads.std() / avg_threshold_grads.mean():.3f}")

    # Check if there's meaningful variation
    cv = avg_threshold_grads.std() / avg_threshold_grads.mean()
    if cv > 1.0:
        print("\n✓ HIGH variation in gradients across perceptrons (CV > 1.0)")
        print("  This suggests different perceptrons are learning at different rates")
    elif cv > 0.5:
        print("\n~ MODERATE variation in gradients (0.5 < CV < 1.0)")
        print("  Some differentiation, but may not be enough")
    else:
        print("\n✗ LOW variation in gradients (CV < 0.5)")
        print("  Perceptrons are updating similarly - no multi-timescale learning")

    # Identify fast vs slow learners
    threshold = np.percentile(avg_threshold_grads, 75)
    fast_learners = np.where(avg_threshold_grads > threshold)[0]
    slow_learners = np.where(avg_threshold_grads < np.percentile(avg_threshold_grads, 25))[0]

    print(f"\nFast learners (top 25%): {len(fast_learners)} perceptrons")
    print(f"Slow learners (bottom 25%): {len(slow_learners)} perceptrons")
    print(f"Gradient ratio (fast/slow): {avg_threshold_grads[fast_learners].mean() / avg_threshold_grads[slow_learners].mean():.2f}x")

    # Check connection gradients
    print("\n\nConnection gradient statistics (signal weights):")
    print(f"  Mean:   {avg_signal_grads.mean():.6f}")
    print(f"  Std:    {avg_signal_grads.std():.6f}")
    print(f"  CV:     {avg_signal_grads.std() / avg_signal_grads.mean():.3f}")

    return {
        'threshold_grads': avg_threshold_grads,
        'signal_grads': avg_signal_grads,
        'fast_learners': fast_learners,
        'slow_learners': slow_learners
    }


if __name__ == "__main__":
    print("="*70)
    print("Multi-Timescale Learning Analysis")
    print("="*70)

    # Create network
    print("\nCreating network...")
    network = FastMessyPerceptronNetwork(
        n_perceptrons=500,
        avg_degree=15,
        n_input_perceptrons=250,
        n_output_perceptrons=125,
        settling_iterations=7,
        seed=42
    )

    # Load MNIST
    print("Loading MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Analyze
    results = analyze_gradient_distribution(network, dataloader, device)

    print("\n" + "="*70)
    print("Analysis Complete")
    print("="*70)
