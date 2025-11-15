"""
Test continual learning with the properly deep architecture.

Tests:
- MNIST Task 1: Digits 0-4
- MNIST Task 2: Digits 5-9
- Proper train/test splits
- Learning curves
- Catastrophic forgetting measurement
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

# Create network with proper depth
print("\nCreating deep messy perceptron network...")
network = FastMessyPerceptronNetwork(
    n_perceptrons=2000,
    avg_degree=30,
    n_input_perceptrons=250,  # ~16x16 = 256
    n_output_perceptrons=125,  # 10 classes + headroom
    settling_iterations=60,  # NEW: covers longest recurrent paths
    seed=42
).to(device)

# Map 10 MNIST classes to output perceptrons
# Use first 10 output neurons for the classes
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

            # Flatten to match input perceptrons (250 < 784, so we'll downsample)
            # Simple downsampling: take every 3rd pixel approximately
            images_downsampled = images[:, ::3][:, :250]

            # Forward pass
            activations = network(images_downsampled)
            logits = output_layer(activations)
            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    return accuracy

def train_epoch(network, output_layer, optimizer, data_loader):
    """Train for one epoch."""
    network.train()
    output_layer.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
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
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)

# Optimizer
all_params = list(network.parameters()) + list(output_layer.parameters())
optimizer = optim.Adam(all_params, lr=0.001)

print("\n" + "="*60)
print("TASK 1 TRAINING (Digits 0-4)")
print("="*60)

n_epochs = 15
for epoch in range(n_epochs):
    train_loss = train_epoch(network, output_layer, optimizer, task1_train_loader)
    test_acc = evaluate(network, output_layer, task1_test_loader)
    print(f"Epoch {epoch+1:2d}: Loss={train_loss:.4f}, Test Acc={test_acc:.1f}%")

task1_accuracy_after_task1 = evaluate(network, output_layer, task1_test_loader)
print(f"\nTask 1 final test accuracy: {task1_accuracy_after_task1:.1f}%")

print("\n" + "="*60)
print("TASK 2 TRAINING (Digits 5-9)")
print("="*60)

for epoch in range(n_epochs):
    train_loss = train_epoch(network, output_layer, optimizer, task2_train_loader)
    test_acc = evaluate(network, output_layer, task2_test_loader)
    print(f"Epoch {epoch+1:2d}: Loss={train_loss:.4f}, Test Acc={test_acc:.1f}%")

task2_accuracy_after_task2 = evaluate(network, output_layer, task2_test_loader)
print(f"\nTask 2 final test accuracy: {task2_accuracy_after_task2:.1f}%")

# Measure catastrophic forgetting
task1_accuracy_after_task2 = evaluate(network, output_layer, task1_test_loader)
forgetting = 100.0 * (task1_accuracy_after_task1 - task1_accuracy_after_task2) / task1_accuracy_after_task1

print("\n" + "="*60)
print("CONTINUAL LEARNING RESULTS")
print("="*60)
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
