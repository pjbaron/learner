"""
MNIST Continual Learning with Plasticity Resources.

Tests the resource-based plasticity mechanism for preventing catastrophic forgetting.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from messy_perceptron_network.core.fast_network_with_resources import FastMessyPerceptronNetworkWithResources
from messy_perceptron_network.training.resource_trainer import ResourceBasedTrainer


def load_mnist_split(root='./data', train=True, digits=None):
    """Load MNIST dataset, optionally filtering by digit class."""
    try:
        dataset = datasets.MNIST(
            root=root,
            train=train,
            download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
    except:
        print(f"Downloading MNIST data to {root}...")
        dataset = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )

    if digits is not None:
        indices = [i for i, (_, label) in enumerate(dataset) if label in digits]
        dataset = Subset(dataset, indices)

    return dataset


def create_mnist_dataloaders(batch_size=32):
    """Create dataloaders for continual learning."""
    task1_train = load_mnist_split(train=True, digits=[0, 1, 2, 3, 4])
    task1_test = load_mnist_split(train=False, digits=[0, 1, 2, 3, 4])

    task2_train = load_mnist_split(train=True, digits=[5, 6, 7, 8, 9])
    task2_test = load_mnist_split(train=False, digits=[5, 6, 7, 8, 9])

    dataloaders = {
        'task_1_digits_0-4': (
            DataLoader(task1_train, batch_size=batch_size, shuffle=True),
            DataLoader(task1_test, batch_size=batch_size, shuffle=False)
        ),
        'task_2_digits_5-9': (
            DataLoader(task2_train, batch_size=batch_size, shuffle=True),
            DataLoader(task2_test, batch_size=batch_size, shuffle=False)
        ),
    }

    print(f"Created dataloaders:")
    print(f"  Task 1 (0-4): {len(task1_train)} train, {len(task1_test)} test")
    print(f"  Task 2 (5-9): {len(task2_train)} train, {len(task2_test)} test")

    return dataloaders


class MNISTClassifier(nn.Module):
    """Classifier wrapper with input/output projections."""

    def __init__(self, network, n_classes=10):
        super(MNISTClassifier, self).__init__()
        self.network = network
        self.input_projection = nn.Linear(784, network.n_input_perceptrons)
        self.output_layer = nn.Linear(network.n_output_perceptrons, n_classes)

    def forward(self, x):
        projected = self.input_projection(x)
        network_out = self.network(projected)
        logits = self.output_layer(network_out)
        return logits


def run_continual_learning_with_resources(
    n_perceptrons=500,
    avg_degree=15,
    n_epochs_per_task=3,
    batch_size=64,
    base_lr=0.001,
    initial_resource=1.0,
    depletion_rate=0.1,
    recovery_rate=0.01,
    device='cpu',
    seed=42
):
    """Run continual learning experiment with plasticity resources."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*70}")
    print("MNIST Continual Learning with Plasticity Resources")
    print(f"{'='*70}\n")

    print(f"Configuration:")
    print(f"  Perceptrons: {n_perceptrons}")
    print(f"  Avg degree: {avg_degree}")
    print(f"  Epochs per task: {n_epochs_per_task}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {base_lr}")
    print(f"  Initial resource: {initial_resource}")
    print(f"  Depletion rate: {depletion_rate}")
    print(f"  Recovery rate: {recovery_rate}")
    print(f"  Device: {device}\n")

    # Create network with resources
    print("Creating network with plasticity resources...")
    network = FastMessyPerceptronNetworkWithResources(
        n_perceptrons=n_perceptrons,
        avg_degree=avg_degree,
        n_input_perceptrons=min(784, n_perceptrons // 2),
        n_output_perceptrons=min(200, n_perceptrons // 4),
        settling_iterations=7,
        initial_resource=initial_resource,
        depletion_rate=depletion_rate,
        recovery_rate=recovery_rate,
        seed=seed
    )

    # Create classifier
    classifier = MNISTClassifier(network, n_classes=10)

    # Create trainer
    trainer = ResourceBasedTrainer(
        network,
        classifier,
        base_lr=base_lr,
        device=device
    )

    # Load data
    print("\nLoading MNIST data...")
    dataloaders = create_mnist_dataloaders(batch_size=batch_size)

    task_sequence = [
        'task_1_digits_0-4',
        'task_2_digits_5-9',
        'task_1_digits_0-4',
    ]

    print(f"\nTask sequence: {' -> '.join(task_sequence)}\n")

    results = {'task_performance': []}

    # Train on each task
    for task_idx, task_name in enumerate(task_sequence):
        print(f"\n{'='*70}")
        print(f"Task {task_idx + 1}/{len(task_sequence)}: {task_name}")
        print(f"{'='*70}\n")

        train_loader, test_loader = dataloaders[task_name]

        for epoch in range(n_epochs_per_task):
            print(f"Epoch {epoch + 1}/{n_epochs_per_task}")
            epoch_stats = trainer.train_epoch(train_loader, verbose=True)
            print(f"  Train: Loss={epoch_stats['avg_loss']:.4f}, "
                  f"Acc={epoch_stats['avg_accuracy']:.4f}")

            eval_stats = trainer.evaluate(test_loader)
            print(f"  Test:  Loss={eval_stats['loss']:.4f}, "
                  f"Acc={eval_stats['accuracy']:.4f}\n")

        # Evaluate on all tasks
        print(f"\nPerformance on all tasks after {task_name}:")
        all_task_perf = {}
        for eval_task_name, (_, eval_test_loader) in dataloaders.items():
            eval_stats = trainer.evaluate(eval_test_loader)
            all_task_perf[eval_task_name] = eval_stats
            print(f"  {eval_task_name}: "
                  f"Loss={eval_stats['loss']:.4f}, "
                  f"Acc={eval_stats['accuracy']:.4f}")

        # Resource summary
        resource_summary = trainer.get_resource_summary()
        print(f"\nPlasticity Resources:")
        print(f"  Mean: {resource_summary['current_mean']:.3f}")
        print(f"  Min: {resource_summary['current_min']:.3f}")

        results['task_performance'].append({
            'task_trained': task_name,
            'all_task_performance': all_task_perf,
            'resource_summary': resource_summary,
        })

    # Compute metrics
    print(f"\n{'='*70}")
    print("Continual Learning Metrics")
    print(f"{'='*70}\n")

    task1_name = 'task_1_digits_0-4'
    perf_after_task1 = results['task_performance'][0]['all_task_performance'][task1_name]['accuracy']
    perf_after_task2 = results['task_performance'][1]['all_task_performance'][task1_name]['accuracy']
    perf_after_return = results['task_performance'][2]['all_task_performance'][task1_name]['accuracy']

    forgetting = perf_after_task1 - perf_after_task2
    forgetting_pct = 100 * forgetting / perf_after_task1
    backward_transfer = perf_after_return - perf_after_task2
    backward_transfer_pct = 100 * backward_transfer / perf_after_task1

    print(f"Task 1 (0-4) Performance:")
    print(f"  After initial training: {perf_after_task1:.4f}")
    print(f"  After training on task 2: {perf_after_task2:.4f}")
    print(f"  After returning: {perf_after_return:.4f}\n")

    print(f"Catastrophic Forgetting: {forgetting:.4f} ({forgetting_pct:.2f}%)")
    print(f"Backward Transfer: {backward_transfer:.4f} ({backward_transfer_pct:.2f}%)\n")

    return results


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = run_continual_learning_with_resources(
        n_perceptrons=500,
        avg_degree=15,
        n_epochs_per_task=3,
        batch_size=64,
        base_lr=0.001,
        initial_resource=1.0,
        depletion_rate=100.0,     # Moderate with binary gating
        recovery_rate=0.001,      # Slow recovery
        device=device,
        seed=42
    )

    print(f"\n{'='*70}")
    print("Experiment Complete!")
    print(f"{'='*70}\n")
