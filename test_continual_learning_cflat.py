"""
Continual Learning Test with C-Flat Optimizer (State-of-the-Art 2025)

Tests the deep messy perceptron network on sequential MNIST tasks
using C-Flat (NeurIPS 2024) for mitigating catastrophic forgetting.

C-Flat promotes flatter loss landscapes specifically optimized for
continual learning, seeking parameters that generalize well across tasks.

Paper: "Make Continual Learning Stronger via C-Flat"
GitHub: https://github.com/WanNaa/C-Flat
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

from messy_perceptron_network.core.fast_network import FastMessyPerceptronNetwork
from messy_perceptron_network.optimizers.c_flat import CFlatOptimizer

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Load MNIST
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)

def get_task_indices(dataset, digits):
    """Get indices for specific digits."""
    indices = [i for i, (_, label) in enumerate(dataset) if label in digits]
    return indices

# Task 1: digits 0-4
# Task 2: digits 5-9
task1_train_indices = get_task_indices(mnist_train, [0, 1, 2, 3, 4])
task1_test_indices = get_task_indices(mnist_test, [0, 1, 2, 3, 4])
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

print(f"Dataset sizes:")
print(f"  Task 1 train: {len(task1_train_subset)}, test: {len(task1_test_subset)}")
print(f"  Task 2 train: {len(task2_train_subset)}, test: {len(task2_test_subset)}")

# Create network with proper depth
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

def train_epoch_cflat(network, output_layer, optimizer, data_loader):
    """Train for one epoch using C-Flat optimizer."""
    network.train()
    output_layer.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    for images, labels in data_loader:
        images = images.view(-1, 28*28).to(device)
        labels = labels.to(device)

        # Downsample to 250 inputs
        images_downsampled = images[:, ::3][:, :250]

        # Define closure for C-Flat
        def loss_fn():
            activations = network(images_downsampled)
            logits = output_layer(activations)
            loss = criterion(logits, labels)
            return logits, loss

        # C-Flat optimization step
        optimizer.set_closure(loss_fn)
        logits, loss = optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)

# Create base optimizer
base_optimizer = optim.Adam(
    list(network.parameters()) + list(output_layer.parameters()),
    lr=0.001
)

# Wrap with C-Flat for continual learning
print("\nInitializing C-Flat optimizer (NeurIPS 2024)...")
print("  rho = 0.2 (perturbation radius)")
print("  lambda = 0.2 (first-order flatness weight)")

cflat_optimizer = CFlatOptimizer(
    params=list(network.parameters()) + list(output_layer.parameters()),
    base_optimizer=base_optimizer,
    model=nn.Sequential(network, output_layer),
    rho=0.2,      # From paper: fixed at 0.2
    lambda_=0.2,  # From paper: fixed at 0.2
    adaptive=False
)

print("\n" + "="*60)
print("TASK 1 TRAINING (Digits 0-4) - WITH C-FLAT")
print("="*60)

n_epochs = 10
for epoch in range(n_epochs):
    train_loss = train_epoch_cflat(network, output_layer, cflat_optimizer, task1_train_loader)
    test_acc = evaluate(network, output_layer, task1_test_loader)
    print(f"Epoch {epoch+1:2d}: Loss={train_loss:.4f}, Test Acc={test_acc:.1f}%")

task1_accuracy_after_task1 = evaluate(network, output_layer, task1_test_loader)
print(f"\nTask 1 final test accuracy: {task1_accuracy_after_task1:.1f}%")

print("\n" + "="*60)
print("TASK 2 TRAINING (Digits 5-9) - WITH C-FLAT")
print("="*60)

for epoch in range(n_epochs):
    train_loss = train_epoch_cflat(network, output_layer, cflat_optimizer, task2_train_loader)
    test_acc = evaluate(network, output_layer, task2_test_loader)
    print(f"Epoch {epoch+1:2d}: Loss={train_loss:.4f}, Test Acc={test_acc:.1f}%")

task2_accuracy_after_task2 = evaluate(network, output_layer, task2_test_loader)
print(f"\nTask 2 final test accuracy: {task2_accuracy_after_task2:.1f}%")

# Measure catastrophic forgetting
task1_accuracy_after_task2 = evaluate(network, output_layer, task1_test_loader)
forgetting = 100.0 * (task1_accuracy_after_task1 - task1_accuracy_after_task2) / task1_accuracy_after_task1

print("\n" + "="*60)
print("CONTINUAL LEARNING RESULTS (WITH C-FLAT)")
print("="*60)
print(f"Task 1 accuracy after Task 1: {task1_accuracy_after_task1:.1f}%")
print(f"Task 1 accuracy after Task 2: {task1_accuracy_after_task2:.1f}%")
print(f"Task 2 accuracy after Task 2: {task2_accuracy_after_task2:.1f}%")
print()
print(f"Catastrophic forgetting: {forgetting:.1f}%")

if forgetting < 20:
    print("✓ EXCELLENT: Minimal catastrophic forgetting")
    status = "EXCELLENT"
elif forgetting < 50:
    print("~ MODERATE: Some catastrophic forgetting")
    status = "MODERATE"
else:
    print("✗ POOR: High catastrophic forgetting")
    status = "POOR"

print("\n" + "="*60)
print("COMPARISON WITH BASELINE (Standard Adam)")
print("="*60)
print("Baseline results (from test_continual_learning_fast.py):")
print("  Task 1 after Task 1: 98.0%")
print("  Task 1 after Task 2: 2.0%")
print("  Forgetting: 98.0% (POOR)")
print()
print("C-Flat results (current test):")
print(f"  Task 1 after Task 1: {task1_accuracy_after_task1:.1f}%")
print(f"  Task 1 after Task 2: {task1_accuracy_after_task2:.1f}%")
print(f"  Forgetting: {forgetting:.1f}% ({status})")
print()
improvement = 98.0 - forgetting
print(f"Forgetting reduction: {improvement:.1f} percentage points")
if improvement > 0:
    print(f"Relative improvement: {improvement/98.0*100:.1f}%")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
