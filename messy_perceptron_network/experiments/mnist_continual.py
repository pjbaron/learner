"""
MNIST Continual Learning Experiment.

Tests the messy perceptron network on sequential MNIST tasks to demonstrate
resistance to catastrophic forgetting.

Task sequence:
1. Train on digits 0-4
2. Train on digits 5-9
3. Return to digits 0-4

Metrics:
- Catastrophic forgetting: accuracy drop on 0-4 after training on 5-9
- Backward transfer: accuracy recovery when returning to 0-4
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from messy_perceptron_network import (
    MessyPerceptronNetwork,
    MessyPerceptronTrainer,
    ContinualLearner,
)


def load_mnist_split(root='./data', train=True, digits=None):
    """
    Load MNIST dataset, optionally filtering by digit class.

    Args:
        root: Data directory
        train: Load training set if True, test set if False
        digits: List of digits to include (e.g., [0,1,2,3,4]). If None, include all.

    Returns:
        Dataset with filtered digits
    """
    # Download and load MNIST (using alternative mirror)
    from torchvision.datasets.utils import download_url

    # Try to load existing data first, download only if needed
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
        # If data doesn't exist, download from alternative source
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
        # Filter by digit class
        indices = [i for i, (_, label) in enumerate(dataset) if label in digits]
        dataset = Subset(dataset, indices)

    return dataset


def flatten_mnist(batch):
    """
    Flatten MNIST images for the network.

    Args:
        batch: Tuple of (images, labels) where images are (batch, 1, 28, 28)

    Returns:
        Tuple of (flattened_images, labels)
    """
    images, labels = batch
    flattened = images.view(images.size(0), -1)  # (batch, 784)
    return flattened, labels


def create_mnist_dataloaders(batch_size=32):
    """
    Create dataloaders for continual learning on MNIST.

    Returns:
        Dictionary mapping task names to (train_loader, test_loader) tuples
    """
    # Task 1: Digits 0-4
    task1_train = load_mnist_split(train=True, digits=[0, 1, 2, 3, 4])
    task1_test = load_mnist_split(train=False, digits=[0, 1, 2, 3, 4])

    # Task 2: Digits 5-9
    task2_train = load_mnist_split(train=True, digits=[5, 6, 7, 8, 9])
    task2_test = load_mnist_split(train=False, digits=[5, 6, 7, 8, 9])

    # Create dataloaders
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
    """
    Wrapper around MessyPerceptronNetwork for MNIST classification.

    Adds an output layer to map from network outputs to class logits.
    """

    def __init__(self, network, n_classes=10):
        """
        Initialize MNIST classifier.

        Args:
            network: MessyPerceptronNetwork instance
            n_classes: Number of output classes (default: 10)
        """
        super(MNISTClassifier, self).__init__()
        self.network = network
        self.output_layer = nn.Linear(network.n_output_perceptrons, n_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input images (batch, 784)

        Returns:
            logits: Class logits (batch, n_classes)
        """
        # Get network outputs
        network_out = self.network(x)

        # Map to class logits
        logits = self.output_layer(network_out)

        return logits


class MNISTContinualTrainer:
    """
    Trainer for MNIST continual learning experiments.

    Wraps MessyPerceptronTrainer to handle MNIST-specific tasks like
    classification accuracy.
    """

    def __init__(self, classifier, base_lr=0.001, device='cpu'):
        """
        Initialize MNIST trainer.

        Args:
            classifier: MNISTClassifier instance
            base_lr: Base learning rate
            device: Device for training
        """
        self.classifier = classifier
        self.device = torch.device(device)
        self.classifier.to(self.device)

        # Optimizer for both network and output layer
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=base_lr)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, inputs, labels):
        """
        Single training step.

        Args:
            inputs: Input images (batch, 784)
            labels: Target labels (batch,)

        Returns:
            Dictionary with training statistics
        """
        self.classifier.train()
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()

        # Forward pass
        logits = self.classifier(inputs)

        # Compute loss
        loss = self.criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.classifier.parameters(),
            1.0
        )

        # Update
        self.optimizer.step()

        # Compute accuracy
        _, predicted = torch.max(logits, 1)
        accuracy = (predicted == labels).float().mean().item()

        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'gradient_norm': grad_norm.item(),
        }

    def train_epoch(self, dataloader, verbose=True):
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader with (images, labels) batches
            verbose: Print progress

        Returns:
            Dictionary with epoch statistics
        """
        self.classifier.train()

        epoch_losses = []
        epoch_accuracies = []

        for batch_idx, (images, labels) in enumerate(dataloader):
            # Flatten images
            inputs = images.view(images.size(0), -1)

            # Training step
            stats = self.train_step(inputs, labels)

            epoch_losses.append(stats['loss'])
            epoch_accuracies.append(stats['accuracy'])

            if verbose and batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}: "
                      f"Loss={stats['loss']:.4f}, "
                      f"Acc={stats['accuracy']:.4f}")

        return {
            'avg_loss': np.mean(epoch_losses),
            'avg_accuracy': np.mean(epoch_accuracies),
        }

    def evaluate(self, dataloader):
        """
        Evaluate on a dataset.

        Args:
            dataloader: DataLoader with (images, labels) batches

        Returns:
            Dictionary with evaluation statistics
        """
        self.classifier.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in dataloader:
                inputs = images.view(images.size(0), -1).to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                logits = self.classifier(inputs)

                # Compute loss
                loss = self.criterion(logits, labels)
                total_loss += loss.item() * inputs.size(0)

                # Compute accuracy
                _, predicted = torch.max(logits, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += inputs.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
        }


def run_continual_learning_experiment(
    n_perceptrons=1000,
    avg_degree=20,
    n_epochs_per_task=5,
    batch_size=32,
    base_lr=0.001,
    device='cpu',
    seed=42
):
    """
    Run the full continual learning experiment.

    Args:
        n_perceptrons: Number of perceptrons in network
        avg_degree: Average connections per perceptron
        n_epochs_per_task: Number of epochs to train on each task
        batch_size: Batch size
        base_lr: Learning rate
        device: Device for training
        seed: Random seed

    Returns:
        Dictionary with experiment results
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*70}")
    print("MNIST Continual Learning Experiment")
    print(f"{'='*70}\n")

    print(f"Configuration:")
    print(f"  Perceptrons: {n_perceptrons}")
    print(f"  Avg degree: {avg_degree}")
    print(f"  Epochs per task: {n_epochs_per_task}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {base_lr}")
    print(f"  Device: {device}\n")

    # Create network
    print("Creating messy perceptron network...")
    network = MessyPerceptronNetwork(
        n_perceptrons=n_perceptrons,
        avg_degree=avg_degree,
        n_input_perceptrons=200,  # Subset of 784 MNIST pixels
        n_output_perceptrons=100,  # Map to 10 classes via output layer
        settling_iterations=7,
        seed=seed
    )

    # Create classifier
    classifier = MNISTClassifier(network, n_classes=10)

    # Create trainer
    trainer = MNISTContinualTrainer(classifier, base_lr=base_lr, device=device)

    # Load data
    print("\nLoading MNIST data...")
    dataloaders = create_mnist_dataloaders(batch_size=batch_size)

    # Task sequence
    task_sequence = [
        'task_1_digits_0-4',
        'task_2_digits_5-9',
        'task_1_digits_0-4',  # Return to first task
    ]

    print(f"\nTask sequence: {' -> '.join(task_sequence)}\n")

    # Track results
    results = {
        'task_performance': [],
        'task_sequence': task_sequence,
    }

    # Train on each task
    for task_idx, task_name in enumerate(task_sequence):
        print(f"\n{'='*70}")
        print(f"Task {task_idx + 1}/{len(task_sequence)}: {task_name}")
        print(f"{'='*70}\n")

        train_loader, test_loader = dataloaders[task_name]

        # Train
        for epoch in range(n_epochs_per_task):
            print(f"Epoch {epoch + 1}/{n_epochs_per_task}")
            epoch_stats = trainer.train_epoch(train_loader, verbose=True)
            print(f"  Train: Loss={epoch_stats['avg_loss']:.4f}, "
                  f"Acc={epoch_stats['avg_accuracy']:.4f}")

            # Evaluate on current task
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

        results['task_performance'].append({
            'task_trained': task_name,
            'all_task_performance': all_task_perf,
        })

    # Compute continual learning metrics
    print(f"\n{'='*70}")
    print("Continual Learning Metrics")
    print(f"{'='*70}\n")

    # Get performance on task 1 at different stages
    task1_name = 'task_1_digits_0-4'

    # After training on task 1
    perf_after_task1 = results['task_performance'][0]['all_task_performance'][task1_name]['accuracy']

    # After training on task 2 (catastrophic forgetting)
    perf_after_task2 = results['task_performance'][1]['all_task_performance'][task1_name]['accuracy']

    # After returning to task 1 (backward transfer)
    perf_after_return = results['task_performance'][2]['all_task_performance'][task1_name]['accuracy']

    # Compute metrics
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

    results['metrics'] = {
        'perf_after_task1': perf_after_task1,
        'perf_after_task2': perf_after_task2,
        'perf_after_return': perf_after_return,
        'forgetting': forgetting,
        'forgetting_pct': forgetting_pct,
        'backward_transfer': backward_transfer,
        'backward_transfer_pct': backward_transfer_pct,
    }

    return results


if __name__ == "__main__":
    # Run experiment with smaller network for testing
    results = run_continual_learning_experiment(
        n_perceptrons=500,  # Smaller for faster testing
        avg_degree=15,
        n_epochs_per_task=3,
        batch_size=32,
        base_lr=0.001,
        device='cpu',
        seed=42
    )

    print(f"\n{'='*70}")
    print("Experiment Complete!")
    print(f"{'='*70}\n")
