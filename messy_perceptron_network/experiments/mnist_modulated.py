"""
MNIST continual learning with ACTIVATION-DRIVEN plasticity modulation (HOPE design).

Tests whether emergent task routing through plasticity modulation can prevent
catastrophic forgetting.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from messy_perceptron_network.core.fast_network_modulated import FastMessyPerceptronNetwork
from messy_perceptron_network.training.modulated_trainer import ModulatedTrainer


class MNISTClassifier(nn.Module):
    """Wrapper around network with input/output projections."""
    def __init__(self, network, n_classes=10):
        super().__init__()
        self.network = network
        self.input_projection = nn.Linear(784, network.n_input_perceptrons)
        self.output_layer = nn.Linear(network.n_output_perceptrons, n_classes)

    def forward(self, x):
        # Flatten MNIST images
        x = x.view(x.size(0), -1)
        # Project to network input space
        projected = self.input_projection(x)
        # Network forward pass (computes plasticity rates from activations)
        network_output = self.network(projected)
        # Project to class logits
        logits = self.output_layer(network_output)
        return logits


def create_task_dataloaders(task_digits, batch_size=64, train=True):
    """Create dataloader for specific digits."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(
        root='./data',
        train=train,
        download=True,
        transform=transform
    )

    # Filter for task digits
    indices = [i for i, (_, label) in enumerate(dataset) if label in task_digits]
    task_dataset = Subset(dataset, indices)

    return DataLoader(task_dataset, batch_size=batch_size, shuffle=train)


def run_continual_learning_modulated(
    n_perceptrons=500,
    avg_degree=15,
    n_epochs_per_task=3,
    batch_size=64,
    base_lr=0.001,
    device='cpu',
    seed=42
):
    """
    Run continual learning experiment with activation-driven plasticity modulation.

    Task sequence:
    1. Train on digits 0-4
    2. Train on digits 5-9
    3. Return to digits 0-4

    Measure catastrophic forgetting and task routing.
    """
    print("\n" + "="*70)
    print("MNIST Continual Learning with Activation-Driven Plasticity Modulation")
    print("="*70)
    print("\nConfiguration:")
    print(f"  Perceptrons: {n_perceptrons}")
    print(f"  Avg degree: {avg_degree}")
    print(f"  Epochs per task: {n_epochs_per_task}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {base_lr}")
    print(f"  Device: {device}\n")

    # Create network with activation-driven plasticity modulation
    print("Creating network with activation-driven plasticity modulation...")
    network = FastMessyPerceptronNetwork(
        n_perceptrons=n_perceptrons,
        avg_degree=avg_degree,
        n_input_perceptrons=min(784, n_perceptrons // 2),
        n_output_perceptrons=min(200, n_perceptrons // 4),
        settling_iterations=7,
        default_plasticity=0.5,
        seed=seed
    )

    # Create classifier
    classifier = MNISTClassifier(network, n_classes=10)

    # Create optimizer
    optimizer = torch.optim.Adam(classifier.parameters(), lr=base_lr)

    # Create trainer
    trainer = ModulatedTrainer(
        classifier=classifier,
        network=network,
        optimizer=optimizer,
        device=device
    )

    # Load data
    print("\nLoading MNIST data...")
    task_1_train = create_task_dataloaders([0, 1, 2, 3, 4], batch_size, train=True)
    task_1_test = create_task_dataloaders([0, 1, 2, 3, 4], batch_size, train=False)
    task_2_train = create_task_dataloaders([5, 6, 7, 8, 9], batch_size, train=True)
    task_2_test = create_task_dataloaders([5, 6, 7, 8, 9], batch_size, train=False)

    print(f"Created dataloaders:")
    print(f"  Task 1 (0-4): {len(task_1_train.dataset)} train, {len(task_1_test.dataset)} test")
    print(f"  Task 2 (5-9): {len(task_2_train.dataset)} train, {len(task_2_test.dataset)} test")

    print(f"\nTask sequence: task_1_digits_0-4 -> task_2_digits_5-9 -> task_1_digits_0-4\n")

    results = {}

    # Task 1: Train on digits 0-4
    print("\n" + "="*70)
    print("Task 1/3: task_1_digits_0-4")
    print("="*70 + "\n")

    for epoch in range(n_epochs_per_task):
        print(f"Epoch {epoch+1}/{n_epochs_per_task}")
        train_metrics = trainer.train_epoch(task_1_train)
        test_metrics = trainer.evaluate(task_1_test)
        print(f"  Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.4f}")
        print(f"  Test:  Loss={test_metrics['loss']:.4f}, Acc={test_metrics['accuracy']:.4f}\n")

    # Evaluate on both tasks
    print("\nPerformance on all tasks after task_1_digits_0-4:")
    task_1_perf = trainer.evaluate(task_1_test)
    task_2_perf = trainer.evaluate(task_2_test)
    print(f"  task_1_digits_0-4: Loss={task_1_perf['loss']:.4f}, Acc={task_1_perf['accuracy']:.4f}")
    print(f"  task_2_digits_5-9: Loss={task_2_perf['loss']:.4f}, Acc={task_2_perf['accuracy']:.4f}")

    results['task_1_after_task_1'] = task_1_perf['accuracy']

    # Get plasticity stats
    plasticity_stats = network.get_plasticity_stats()
    print(f"\nPlasticity Rates (Task 1):")
    print(f"  Mean: {plasticity_stats['mean']:.3f}")
    print(f"  Std:  {plasticity_stats['std']:.3f}")
    print(f"  Min:  {plasticity_stats['min']:.3f}")
    print(f"  Max:  {plasticity_stats['max']:.3f}")

    # Task 2: Train on digits 5-9
    print("\n" + "="*70)
    print("Task 2/3: task_2_digits_5-9")
    print("="*70 + "\n")

    for epoch in range(n_epochs_per_task):
        print(f"Epoch {epoch+1}/{n_epochs_per_task}")
        train_metrics = trainer.train_epoch(task_2_train)
        test_metrics = trainer.evaluate(task_2_test)
        print(f"  Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.4f}")
        print(f"  Test:  Loss={test_metrics['loss']:.4f}, Acc={test_metrics['accuracy']:.4f}\n")

    # Evaluate on both tasks (CHECK FOR CATASTROPHIC FORGETTING)
    print("\nPerformance on all tasks after task_2_digits_5-9:")
    task_1_perf = trainer.evaluate(task_1_test)
    task_2_perf = trainer.evaluate(task_2_test)
    print(f"  task_1_digits_0-4: Loss={task_1_perf['loss']:.4f}, Acc={task_1_perf['accuracy']:.4f}")
    print(f"  task_2_digits_5-9: Loss={task_2_perf['loss']:.4f}, Acc={task_2_perf['accuracy']:.4f}")

    results['task_1_after_task_2'] = task_1_perf['accuracy']
    results['task_2_after_task_2'] = task_2_perf['accuracy']

    # Get plasticity stats
    plasticity_stats = network.get_plasticity_stats()
    print(f"\nPlasticity Rates (Task 2):")
    print(f"  Mean: {plasticity_stats['mean']:.3f}")
    print(f"  Std:  {plasticity_stats['std']:.3f}")
    print(f"  Min:  {plasticity_stats['min']:.3f}")
    print(f"  Max:  {plasticity_stats['max']:.3f}")

    # Task 3: Return to digits 0-4
    print("\n" + "="*70)
    print("Task 3/3: task_1_digits_0-4 (return)")
    print("="*70 + "\n")

    for epoch in range(n_epochs_per_task):
        print(f"Epoch {epoch+1}/{n_epochs_per_task}")
        train_metrics = trainer.train_epoch(task_1_train)
        test_metrics = trainer.evaluate(task_1_test)
        print(f"  Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.4f}")
        print(f"  Test:  Loss={test_metrics['loss']:.4f}, Acc={test_metrics['accuracy']:.4f}\n")

    # Final evaluation
    print("\nPerformance on all tasks after returning to task_1_digits_0-4:")
    task_1_perf = trainer.evaluate(task_1_test)
    task_2_perf = trainer.evaluate(task_2_test)
    print(f"  task_1_digits_0-4: Loss={task_1_perf['loss']:.4f}, Acc={task_1_perf['accuracy']:.4f}")
    print(f"  task_2_digits_5-9: Loss={task_2_perf['loss']:.4f}, Acc={task_2_perf['accuracy']:.4f}")

    results['task_1_after_return'] = task_1_perf['accuracy']
    results['task_2_after_return'] = task_2_perf['accuracy']

    # Get final plasticity stats
    plasticity_stats = network.get_plasticity_stats()
    print(f"\nPlasticity Rates (Final):")
    print(f"  Mean: {plasticity_stats['mean']:.3f}")
    print(f"  Std:  {plasticity_stats['std']:.3f}")
    print(f"  Min:  {plasticity_stats['min']:.3f}")
    print(f"  Max:  {plasticity_stats['max']:.3f}")

    # Calculate continual learning metrics
    print("\n" + "="*70)
    print("Continual Learning Metrics")
    print("="*70 + "\n")

    print("Task 1 (0-4) Performance:")
    print(f"  After initial training: {results['task_1_after_task_1']:.4f}")
    print(f"  After training on task 2: {results['task_1_after_task_2']:.4f}")
    print(f"  After returning: {results['task_1_after_return']:.4f}")

    catastrophic_forgetting = results['task_1_after_task_1'] - results['task_1_after_task_2']
    backward_transfer = results['task_1_after_return'] - results['task_1_after_task_2']

    print(f"\nCatastrophic Forgetting: {catastrophic_forgetting:.4f} ({catastrophic_forgetting/results['task_1_after_task_1']*100:.2f}%)")
    print(f"Backward Transfer: {backward_transfer:.4f} ({backward_transfer/results['task_1_after_task_1']*100:.2f}%)")

    return results


if __name__ == "__main__":
    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = run_continual_learning_modulated(
        n_perceptrons=500,
        avg_degree=15,
        n_epochs_per_task=3,
        batch_size=64,
        base_lr=0.001,
        device=device,
        seed=42
    )

    print(f"\n{'='*70}")
    print("Experiment Complete!")
    print(f"{'='*70}\n")
