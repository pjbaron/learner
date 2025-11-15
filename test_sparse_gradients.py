"""
Test continual learning with sparse gradients.

During backprop, we zero out low-value gradients, keeping only the top k%
by absolute magnitude. Hypothesis: the network will learn using fewer nodes
with larger influence, which may reduce catastrophic forgetting.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

from messy_perceptron_network.core.fast_network import FastMessyPerceptronNetwork

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameter: what percentage of gradients to keep (by absolute magnitude)
GRADIENT_SPARSITY = 0.20  # Keep top 20%, zero out bottom 80%

print(f"\nSparse Gradient Configuration:")
print(f"  Keeping top {GRADIENT_SPARSITY*100:.0f}% of gradients by magnitude")
print(f"  Zeroing out bottom {(1-GRADIENT_SPARSITY)*100:.0f}% of gradients")

# Load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)

# Split into two tasks
def get_task_indices(dataset, digits):
    """Get indices for specific digits."""
    indices = [i for i, (_, label) in enumerate(dataset) if label in digits]
    return indices

# Task 1: digits 0-4
task1_train_indices = get_task_indices(mnist_train, [0, 1, 2, 3, 4])
task1_test_indices = get_task_indices(mnist_test, [0, 1, 2, 3, 4])

# Task 2: digits 5-9
task2_train_indices = get_task_indices(mnist_train, [5, 6, 7, 8, 9])
task2_test_indices = get_task_indices(mnist_test, [5, 6, 7, 8, 9])

# Limit to reasonable sizes
task1_train_subset = Subset(mnist_train, task1_train_indices[:2000])
task1_test_subset = Subset(mnist_test, task1_test_indices[:500])
task2_train_subset = Subset(mnist_train, task2_train_indices[:2000])
task2_test_subset = Subset(mnist_test, task2_test_indices[:500])

# Create dataloaders
batch_size = 32
task1_train_loader = DataLoader(task1_train_subset, batch_size=batch_size, shuffle=True)
task1_test_loader = DataLoader(task1_test_subset, batch_size=batch_size)
task2_train_loader = DataLoader(task2_train_subset, batch_size=batch_size, shuffle=True)
task2_test_loader = DataLoader(task2_test_subset, batch_size=batch_size)

print(f"\nDataset sizes:")
print(f"  Task 1 train: {len(task1_train_subset)}, test: {len(task1_test_subset)}")
print(f"  Task 2 train: {len(task2_train_subset)}, test: {len(task2_test_subset)}")

# Create network
print("\nCreating deep messy perceptron network...")
network = FastMessyPerceptronNetwork(
    n_perceptrons=2000,
    avg_degree=30,
    n_input_perceptrons=250,
    n_output_perceptrons=125,
    settling_iterations=60,
    seed=42
).to(device)

# Map 10 MNIST classes to output perceptrons
output_layer = nn.Linear(network.n_output_perceptrons, 10, bias=True).to(device)

def sparsify_gradients(parameters, sparsity=0.20):
    """
    Zero out low-magnitude gradients, keeping only top k% by absolute value.

    Args:
        parameters: Model parameters with gradients
        sparsity: Fraction of gradients to keep (0.0 to 1.0)

    Returns:
        Tuple of (n_total_params, n_kept_params, threshold_value)
    """
    # Collect all gradient magnitudes
    grad_magnitudes = []
    grad_params = []

    for param in parameters:
        if param.grad is not None:
            grad_magnitudes.append(param.grad.abs().flatten())
            grad_params.append(param)

    if not grad_magnitudes:
        return 0, 0, 0.0

    # Concatenate all gradient magnitudes
    all_grads = torch.cat(grad_magnitudes)
    n_total = all_grads.numel()

    # Find threshold: keep top k% by magnitude
    k = max(1, int(n_total * sparsity))
    threshold = torch.topk(all_grads, k).values[-1].item()

    # Zero out gradients below threshold
    n_kept = 0
    for param in grad_params:
        if param.grad is not None:
            mask = param.grad.abs() >= threshold
            param.grad = param.grad * mask.float()
            n_kept += mask.sum().item()

    return n_total, n_kept, threshold


def evaluate(network, output_layer, data_loader):
    """Evaluate accuracy on a dataset."""
    network.eval()
    output_layer.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.view(-1, 28*28).to(device)
            labels = labels.to(device)

            # Downsample to 250 inputs
            images_downsampled = images[:, ::3][:, :250]

            # Forward pass
            activations = network(images_downsampled)
            logits = output_layer(activations)
            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


def train_epoch(network, output_layer, optimizer, data_loader, use_sparse_gradients=True):
    """Train for one epoch with optional gradient sparsification."""
    network.train()
    output_layer.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    total_params_count = 0
    total_kept_count = 0

    all_params = list(network.parameters()) + list(output_layer.parameters())

    for images, labels in data_loader:
        images = images.view(-1, 28*28).to(device)
        labels = labels.to(device)

        # Downsample to 250 inputs
        images_downsampled = images[:, ::3][:, :250]

        # Forward
        activations = network(images_downsampled)
        logits = output_layer(activations)
        loss = criterion(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Sparsify gradients BEFORE optimizer step
        if use_sparse_gradients:
            n_total, n_kept, threshold = sparsify_gradients(all_params, GRADIENT_SPARSITY)
            total_params_count += n_total
            total_kept_count += n_kept

        optimizer.step()

        total_loss += loss.item()

    avg_sparsity = 100.0 * total_kept_count / total_params_count if total_params_count > 0 else 0
    return total_loss / len(data_loader), avg_sparsity


# Optimizer
all_params = list(network.parameters()) + list(output_layer.parameters())
optimizer = optim.Adam(all_params, lr=0.001)

print("\n" + "="*60)
print("TASK 1 TRAINING (Digits 0-4) - WITH SPARSE GRADIENTS")
print("="*60)

n_epochs = 10
for epoch in range(n_epochs):
    train_loss, sparsity = train_epoch(network, output_layer, optimizer, task1_train_loader)
    test_acc = evaluate(network, output_layer, task1_test_loader)
    print(f"Epoch {epoch+1:2d}: Loss={train_loss:.4f}, Test Acc={test_acc:.1f}%, Grad Sparsity={sparsity:.1f}%")

task1_accuracy_after_task1 = evaluate(network, output_layer, task1_test_loader)
print(f"\nTask 1 final test accuracy: {task1_accuracy_after_task1:.1f}%")

print("\n" + "="*60)
print("TASK 2 TRAINING (Digits 5-9) - WITH SPARSE GRADIENTS")
print("="*60)

for epoch in range(n_epochs):
    train_loss, sparsity = train_epoch(network, output_layer, optimizer, task2_train_loader)
    test_acc = evaluate(network, output_layer, task2_test_loader)
    print(f"Epoch {epoch+1:2d}: Loss={train_loss:.4f}, Test Acc={test_acc:.1f}%, Grad Sparsity={sparsity:.1f}%")

task2_accuracy_after_task2 = evaluate(network, output_layer, task2_test_loader)
print(f"\nTask 2 final test accuracy: {task2_accuracy_after_task2:.1f}%")

# Measure catastrophic forgetting
task1_accuracy_after_task2 = evaluate(network, output_layer, task1_test_loader)
forgetting = 100.0 * (task1_accuracy_after_task1 - task1_accuracy_after_task2) / task1_accuracy_after_task1

print("\n" + "="*60)
print("CONTINUAL LEARNING RESULTS (SPARSE GRADIENTS)")
print("="*60)
print(f"Gradient sparsity: keeping top {GRADIENT_SPARSITY*100:.0f}% by magnitude")
print(f"Task 1 accuracy after Task 1: {task1_accuracy_after_task1:.1f}%")
print(f"Task 1 accuracy after Task 2: {task1_accuracy_after_task2:.1f}%")
print(f"Task 2 accuracy after Task 2: {task2_accuracy_after_task2:.1f}%")
print(f"\nCatastrophic forgetting: {forgetting:.1f}%")

if forgetting < 20:
    print("✓ EXCELLENT: Minimal catastrophic forgetting!")
elif forgetting < 40:
    print("✓ GOOD: Moderate catastrophic forgetting")
elif forgetting < 60:
    print("~ FAIR: Significant forgetting, but some retention")
else:
    print("✗ POOR: High catastrophic forgetting")

print("\n" + "="*60)
print("COMPARISON WITH BASELINE (Standard Adam)")
print("="*60)
print("Baseline results (from test_continual_learning_fast.py):")
print("  Task 1 after Task 1: 98.0%")
print("  Task 1 after Task 2: 2.0%")
print("  Forgetting: 98.0% (POOR)")
print()
print("Sparse gradient results (current test):")
print(f"  Task 1 after Task 1: {task1_accuracy_after_task1:.1f}%")
print(f"  Task 1 after Task 2: {task1_accuracy_after_task2:.1f}%")
print(f"  Forgetting: {forgetting:.1f}%")

improvement = 98.0 - forgetting
print(f"\nForgetting reduction: {improvement:.1f} percentage points")

if improvement > 20:
    print("✓ SIGNIFICANT IMPROVEMENT over baseline!")
elif improvement > 5:
    print("✓ MODERATE IMPROVEMENT over baseline")
elif improvement > -5:
    print("~ SIMILAR to baseline")
else:
    print("✗ WORSE than baseline")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
