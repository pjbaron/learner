"""
MNIST continual learning with probabilistic resource-based updates.

Simple idea: Prevent oscillatory traps where contradictory gradients flip weights.
- Frequently updated parameters → low resources → resist further changes
- Rarely updated parameters → high resources → remain flexible
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from messy_perceptron_network.core.fast_network_modulated import FastMessyPerceptronNetwork
from messy_perceptron_network.training.probabilistic_trainer import ProbabilisticResourceTrainer


class MNISTClassifier(nn.Module):
    """Wrapper around network with input/output projections."""
    def __init__(self, network, n_classes=10):
        super().__init__()
        self.network = network
        self.input_projection = nn.Linear(784, network.n_input_perceptrons)
        self.output_layer = nn.Linear(network.n_output_perceptrons, n_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        projected = self.input_projection(x)
        network_output = self.network(projected)
        return self.output_layer(network_output)


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

    indices = [i for i, (_, label) in enumerate(dataset) if label in task_digits]
    task_dataset = Subset(dataset, indices)

    return DataLoader(task_dataset, batch_size=batch_size, shuffle=train)


def run_continual_learning_probabilistic(
    n_perceptrons=500,
    avg_degree=15,
    n_epochs_per_task=3,
    batch_size=64,
    base_lr=0.001,
    initial_resource=1.0,
    consumption_amount=0.01,
    significance_threshold=0.001,
    recovery_rate=0.005,
    device='cpu',
    seed=42
):
    """
    Run continual learning with probabilistic resource-based updates.

    Goal: Prevent oscillatory traps where contradictory gradients flip weights.
    Only deplete resources for parameters with significant gradients.
    """
    print("\n" + "="*70)
    print("MNIST Continual Learning with Probabilistic Resources")
    print("="*70)
    print("\nConfiguration:")
    print(f"  Perceptrons: {n_perceptrons}")
    print(f"  Avg degree: {avg_degree}")
    print(f"  Epochs per task: {n_epochs_per_task}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {base_lr}")
    print(f"  Initial resource: {initial_resource}")
    print(f"  Consumption (per significant update): {consumption_amount}")
    print(f"  Significance threshold: {significance_threshold}")
    print(f"  Recovery rate: {recovery_rate}")
    print(f"  Device: {device}\n")

    # Create network
    print("Creating network...")
    network = FastMessyPerceptronNetwork(
        n_perceptrons=n_perceptrons,
        avg_degree=avg_degree,
        n_input_perceptrons=min(784, n_perceptrons // 2),
        n_output_perceptrons=min(200, n_perceptrons // 4),
        settling_iterations=7,
        seed=seed
    )

    classifier = MNISTClassifier(network, n_classes=10)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=base_lr)

    # Create trainer with probabilistic resources
    trainer = ProbabilisticResourceTrainer(
        network=classifier,
        optimizer=optimizer,
        initial_resource=initial_resource,
        consumption_amount=consumption_amount,
        significance_threshold=significance_threshold,
        recovery_rate=recovery_rate,
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

    results = {}

    # Task 1
    print("\n" + "="*70)
    print("Task 1/3: Digits 0-4")
    print("="*70 + "\n")

    for epoch in range(n_epochs_per_task):
        print(f"Epoch {epoch+1}/{n_epochs_per_task}")
        train_metrics = trainer.train_epoch(task_1_train)
        test_metrics = trainer.evaluate(task_1_test)
        print(f"  Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.4f}")
        print(f"  Test:  Loss={test_metrics['loss']:.4f}, Acc={test_metrics['accuracy']:.4f}\n")

    print("\nPerformance on all tasks after Task 1:")
    task_1_perf = trainer.evaluate(task_1_test)
    task_2_perf = trainer.evaluate(task_2_test)
    print(f"  Task 1 (0-4): Acc={task_1_perf['accuracy']:.4f}")
    print(f"  Task 2 (5-9): Acc={task_2_perf['accuracy']:.4f}")
    results['task_1_after_task_1'] = task_1_perf['accuracy']

    # Task 2
    print("\n" + "="*70)
    print("Task 2/3: Digits 5-9")
    print("="*70 + "\n")

    for epoch in range(n_epochs_per_task):
        print(f"Epoch {epoch+1}/{n_epochs_per_task}")
        train_metrics = trainer.train_epoch(task_2_train)
        test_metrics = trainer.evaluate(task_2_test)
        print(f"  Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.4f}")
        print(f"  Test:  Loss={test_metrics['loss']:.4f}, Acc={test_metrics['accuracy']:.4f}\n")

    print("\nPerformance on all tasks after Task 2:")
    task_1_perf = trainer.evaluate(task_1_test)
    task_2_perf = trainer.evaluate(task_2_test)
    print(f"  Task 1 (0-4): Acc={task_1_perf['accuracy']:.4f}")
    print(f"  Task 2 (5-9): Acc={task_2_perf['accuracy']:.4f}")
    results['task_1_after_task_2'] = task_1_perf['accuracy']

    # Task 3: Return to Task 1
    print("\n" + "="*70)
    print("Task 3/3: Return to Digits 0-4")
    print("="*70 + "\n")

    for epoch in range(n_epochs_per_task):
        print(f"Epoch {epoch+1}/{n_epochs_per_task}")
        train_metrics = trainer.train_epoch(task_1_train)
        test_metrics = trainer.evaluate(task_1_test)
        print(f"  Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.4f}")
        print(f"  Test:  Loss={test_metrics['loss']:.4f}, Acc={test_metrics['accuracy']:.4f}\n")

    print("\nFinal Performance:")
    task_1_perf = trainer.evaluate(task_1_test)
    task_2_perf = trainer.evaluate(task_2_test)
    print(f"  Task 1 (0-4): Acc={task_1_perf['accuracy']:.4f}")
    print(f"  Task 2 (5-9): Acc={task_2_perf['accuracy']:.4f}")
    results['task_1_final'] = task_1_perf['accuracy']

    # Metrics
    print("\n" + "="*70)
    print("Continual Learning Metrics")
    print("="*70)

    forgetting = results['task_1_after_task_1'] - results['task_1_after_task_2']
    print(f"\nCatastrophic Forgetting: {forgetting:.4f} ({forgetting/results['task_1_after_task_1']*100:.2f}%)")

    return results


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    results = run_continual_learning_probabilistic(
        n_perceptrons=500,
        avg_degree=15,
        n_epochs_per_task=3,
        batch_size=64,
        base_lr=0.001,
        initial_resource=1.0,
        consumption_amount=0.01,      # Fixed small consumption
        significance_threshold=0.001, # Only deplete if gradient significant
        recovery_rate=0.005,          # Slow recovery
        device=device,
        seed=42
    )

    print(f"\n{'='*70}")
    print("Experiment Complete!")
    print(f"{'='*70}\n")
