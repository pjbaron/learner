"""
Test continual learning with adaptive plasticity + sparse gradients.

Approach: Gradient-based importance tracking (similar to Synaptic Intelligence)
- During Task 1: Track cumulative |gradient| for each parameter
- After Task 1: Compute importance scores (high = critical for Task 1)
- During Task 2: Apply sparse gradients AND scale by plasticity (1 - importance)

This protects important Task 1 weights while allowing unimportant weights
to freely learn Task 2.
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

# Hyperparameters
GRADIENT_SPARSITY = 0.20  # Keep top 20% of gradients by magnitude
PLASTICITY_DAMPING = 0.1  # Small constant to prevent full freezing of important weights

print(f"\nAdaptive Plasticity + Sparse Gradient Configuration:")
print(f"  Gradient sparsity: keeping top {GRADIENT_SPARSITY*100:.0f}% by magnitude")
print(f"  Plasticity formula: gradient *= (plasticity_damping + (1 - importance))")
print(f"  Plasticity damping: {PLASTICITY_DAMPING} (prevents complete freezing)")

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


class ImportanceTracker:
    """Tracks parameter importance via cumulative gradient magnitude."""

    def __init__(self, parameters):
        self.importance = {}
        self.cumulative_gradients = {}

        # Initialize tracking for each parameter
        for i, param in enumerate(parameters):
            if param.requires_grad:
                self.cumulative_gradients[i] = torch.zeros_like(param.data)
                self.importance[i] = torch.zeros_like(param.data)

    def accumulate_gradients(self, parameters):
        """Accumulate gradient magnitudes during Task 1 training."""
        for i, param in enumerate(parameters):
            if param.grad is not None and i in self.cumulative_gradients:
                self.cumulative_gradients[i] += param.grad.abs()

    def compute_importance(self, parameters):
        """
        Compute normalized importance scores after Task 1.
        Returns statistics about importance distribution.
        """
        # Collect all cumulative gradient values
        all_values = []
        for i in self.cumulative_gradients:
            all_values.append(self.cumulative_gradients[i].flatten())

        all_values = torch.cat(all_values)

        # Normalize to [0, 1] based on global max
        max_importance = all_values.max()
        if max_importance > 0:
            for i in self.cumulative_gradients:
                self.importance[i] = self.cumulative_gradients[i] / max_importance

        # Compute statistics
        mean_importance = all_values.mean().item()
        median_importance = all_values.median().item()
        max_importance_val = max_importance.item()

        # Count how many parameters have high importance (>0.5)
        high_importance_count = (all_values > 0.5).sum().item()
        total_count = all_values.numel()
        high_importance_pct = 100.0 * high_importance_count / total_count

        return {
            'mean': mean_importance,
            'median': median_importance,
            'max': max_importance_val,
            'high_importance_pct': high_importance_pct,
            'total_params': total_count
        }

    def apply_plasticity(self, parameters, damping=0.1):
        """
        Apply plasticity scaling to gradients: gradient *= (damping + (1 - importance))

        High importance (close to 1) -> low plasticity (close to damping)
        Low importance (close to 0) -> high plasticity (close to 1 + damping)
        """
        for i, param in enumerate(parameters):
            if param.grad is not None and i in self.importance:
                plasticity = damping + (1.0 - self.importance[i])
                param.grad *= plasticity


def sparsify_gradients(parameters, sparsity=0.20):
    """Zero out low-magnitude gradients, keeping only top k%."""
    grad_magnitudes = []
    grad_params = []

    for param in parameters:
        if param.grad is not None:
            grad_magnitudes.append(param.grad.abs().flatten())
            grad_params.append(param)

    if not grad_magnitudes:
        return 0, 0, 0.0

    all_grads = torch.cat(grad_magnitudes)
    n_total = all_grads.numel()

    k = max(1, int(n_total * sparsity))
    threshold = torch.topk(all_grads, k).values[-1].item()

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

            images_downsampled = images[:, ::3][:, :250]

            activations = network(images_downsampled)
            logits = output_layer(activations)
            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


def train_epoch(network, output_layer, optimizer, data_loader,
                importance_tracker=None, track_importance=False, apply_plasticity=False):
    """
    Train for one epoch with optional importance tracking and plasticity.

    Args:
        track_importance: If True, accumulate gradients for importance calculation
        apply_plasticity: If True, scale gradients by plasticity before optimizer step
    """
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

        images_downsampled = images[:, ::3][:, :250]

        # Forward
        activations = network(images_downsampled)
        logits = output_layer(activations)
        loss = criterion(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Track importance (Task 1 only)
        if track_importance and importance_tracker is not None:
            importance_tracker.accumulate_gradients(all_params)

        # Apply sparse gradients (both tasks)
        n_total, n_kept, threshold = sparsify_gradients(all_params, GRADIENT_SPARSITY)
        total_params_count += n_total
        total_kept_count += n_kept

        # Apply plasticity (Task 2 only)
        if apply_plasticity and importance_tracker is not None:
            importance_tracker.apply_plasticity(all_params, damping=PLASTICITY_DAMPING)

        optimizer.step()
        total_loss += loss.item()

    avg_sparsity = 100.0 * total_kept_count / total_params_count if total_params_count > 0 else 0
    return total_loss / len(data_loader), avg_sparsity


# Optimizer
all_params = list(network.parameters()) + list(output_layer.parameters())
optimizer = optim.Adam(all_params, lr=0.001)

# Initialize importance tracker
importance_tracker = ImportanceTracker(all_params)

print("\n" + "="*60)
print("TASK 1 TRAINING (Digits 0-4)")
print("WITH: Sparse Gradients + Importance Tracking")
print("="*60)

n_epochs = 10
for epoch in range(n_epochs):
    train_loss, sparsity = train_epoch(
        network, output_layer, optimizer, task1_train_loader,
        importance_tracker=importance_tracker,
        track_importance=True,  # Track importance during Task 1
        apply_plasticity=False  # No plasticity yet
    )
    test_acc = evaluate(network, output_layer, task1_test_loader)
    print(f"Epoch {epoch+1:2d}: Loss={train_loss:.4f}, Test Acc={test_acc:.1f}%, Grad Sparsity={sparsity:.1f}%")

task1_accuracy_after_task1 = evaluate(network, output_layer, task1_test_loader)
print(f"\nTask 1 final test accuracy: {task1_accuracy_after_task1:.1f}%")

# Compute importance scores
print("\n" + "="*60)
print("COMPUTING IMPORTANCE SCORES")
print("="*60)
stats = importance_tracker.compute_importance(all_params)
print(f"Importance statistics:")
print(f"  Mean importance: {stats['mean']:.4f}")
print(f"  Median importance: {stats['median']:.4f}")
print(f"  Max importance: {stats['max']:.4f}")
print(f"  High importance (>0.5): {stats['high_importance_pct']:.1f}% of {stats['total_params']} params")

print("\n" + "="*60)
print("TASK 2 TRAINING (Digits 5-9)")
print("WITH: Sparse Gradients + Adaptive Plasticity")
print("="*60)

for epoch in range(n_epochs):
    train_loss, sparsity = train_epoch(
        network, output_layer, optimizer, task2_train_loader,
        importance_tracker=importance_tracker,
        track_importance=False,  # Don't track during Task 2
        apply_plasticity=True    # Apply plasticity to protect Task 1
    )
    test_acc = evaluate(network, output_layer, task2_test_loader)
    print(f"Epoch {epoch+1:2d}: Loss={train_loss:.4f}, Test Acc={test_acc:.1f}%, Grad Sparsity={sparsity:.1f}%")

task2_accuracy_after_task2 = evaluate(network, output_layer, task2_test_loader)
print(f"\nTask 2 final test accuracy: {task2_accuracy_after_task2:.1f}%")

# Measure catastrophic forgetting
task1_accuracy_after_task2 = evaluate(network, output_layer, task1_test_loader)
forgetting = 100.0 * (task1_accuracy_after_task1 - task1_accuracy_after_task2) / task1_accuracy_after_task1

print("\n" + "="*60)
print("CONTINUAL LEARNING RESULTS")
print("="*60)
print(f"Method: Adaptive Plasticity + Sparse Gradients")
print(f"  Gradient sparsity: {GRADIENT_SPARSITY*100:.0f}%")
print(f"  Plasticity damping: {PLASTICITY_DAMPING}")
print()
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
print("COMPARISON WITH PREVIOUS METHODS")
print("="*60)
print("Baseline (Standard Adam):")
print("  Task 1 after Task 1: 98.0%")
print("  Task 1 after Task 2: 2.0%")
print("  Forgetting: 98.0%")
print()
print("Sparse Gradients Only:")
print("  Task 1 after Task 1: 97.2%")
print("  Task 1 after Task 2: 1.0%")
print("  Forgetting: 99.0%")
print()
print("Adaptive Plasticity + Sparse Gradients (current):")
print(f"  Task 1 after Task 1: {task1_accuracy_after_task1:.1f}%")
print(f"  Task 1 after Task 2: {task1_accuracy_after_task2:.1f}%")
print(f"  Forgetting: {forgetting:.1f}%")

baseline_forgetting = 98.0
improvement = baseline_forgetting - forgetting
print(f"\nForgetting reduction vs baseline: {improvement:.1f} percentage points")

if improvement > 30:
    print("✓✓ MAJOR IMPROVEMENT over baseline!")
elif improvement > 15:
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
