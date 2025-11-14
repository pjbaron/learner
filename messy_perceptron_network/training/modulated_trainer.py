"""
Trainer for activation-driven plasticity modulation (HOPE design).

Key difference from resource-based trainer:
- No depletion/recovery steps
- Plasticity rates computed dynamically from activations during forward pass
- Gradients scaled by these activation-driven rates
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict


class ModulatedTrainer:
    """
    Trainer for networks with activation-driven plasticity modulation.

    Implements the HOPE training protocol:
    1. Forward pass (settling) computes activations
    2. Plasticity rates α = sigmoid(Σ(w_plasticity × a)) computed per perceptron
    3. Backward pass computes gradients
    4. Gradients scaled by plasticity rates: grad *= α
    5. Optimizer step with scaled gradients
    """

    def __init__(self,
                 classifier: nn.Module,
                 network,  # The FastMessyPerceptronNetwork inside classifier
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module = None,
                 gradient_clip_norm: float = 1.0,
                 device: str = 'cpu'):
        """
        Initialize trainer.

        Args:
            classifier: Wrapper with network + input/output projections
            network: The core messy perceptron network
            optimizer: Optimizer for all parameters
            criterion: Loss function (default: CrossEntropyLoss)
            gradient_clip_norm: Max gradient norm
            device: Device to train on
        """
        self.classifier = classifier
        self.network = network
        self.optimizer = optimizer
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.gradient_clip_norm = gradient_clip_norm
        self.device = device
        self.step_count = 0

        # Move to device
        self.classifier.to(device)

    def train_step(self, inputs: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Single training step with activation-driven plasticity modulation.

        Args:
            inputs: (batch, input_dim)
            labels: (batch,) class labels

        Returns:
            Dictionary with loss, accuracy, and plasticity stats
        """
        self.classifier.train()
        self.optimizer.zero_grad()

        # Move data to device
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # Forward pass (computes plasticity rates from activations)
        logits = self.classifier(inputs)
        loss = self.criterion(logits, labels)

        # Backward pass
        loss.backward()

        # CRITICAL: Apply activation-driven plasticity modulation
        # This scales gradients by α values computed during forward pass
        self.network.apply_plasticity_modulation()

        # DEBUG: Check if plasticity modulation connections are learning
        if self.step_count < 2:
            print(f"\n  DEBUG Step {self.step_count}:")
            if self.network.plasticity_weights.grad is not None:
                print(f"    Plasticity weight grad mean: {self.network.plasticity_weights.grad.abs().mean().item():.6f}")
                print(f"    Plasticity weight grad max: {self.network.plasticity_weights.grad.abs().max().item():.6f}")
            else:
                print(f"    Plasticity weight grad: NONE!")
            if self.network.signal_weights.grad is not None:
                print(f"    Signal weight grad mean: {self.network.signal_weights.grad.abs().mean().item():.6f}")
            else:
                print(f"    Signal weight grad: NONE!")
            print()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.classifier.parameters(),
            self.gradient_clip_norm
        )

        # Update weights
        self.optimizer.step()

        # Compute accuracy
        _, predicted = torch.max(logits, 1)
        accuracy = (predicted == labels).float().mean().item()

        # Get plasticity stats
        plasticity_stats = self.network.get_plasticity_stats()

        self.step_count += 1

        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'grad_norm': grad_norm.item(),
            'plasticity_mean': plasticity_stats['mean'],
            'plasticity_std': plasticity_stats['std'],
            'plasticity_min': plasticity_stats['min'],
            'plasticity_max': plasticity_stats['max'],
        }

    def train_epoch(self, dataloader: DataLoader, verbose: bool = True) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader for training data
            verbose: Print progress

        Returns:
            Average metrics over epoch
        """
        total_loss = 0.0
        total_acc = 0.0
        total_plasticity = 0.0
        n_batches = 0

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            metrics = self.train_step(inputs, labels)

            total_loss += metrics['loss']
            total_acc += metrics['accuracy']
            total_plasticity += metrics['plasticity_mean']
            n_batches += 1

            if verbose and batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}: "
                      f"Loss={metrics['loss']:.4f}, "
                      f"Acc={metrics['accuracy']:.4f}, "
                      f"Plasticity={metrics['plasticity_mean']:.3f}")

        return {
            'loss': total_loss / n_batches,
            'accuracy': total_acc / n_batches,
            'plasticity_mean': total_plasticity / n_batches,
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate on a dataset.

        Args:
            dataloader: DataLoader for evaluation data

        Returns:
            Average metrics
        """
        self.classifier.eval()

        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            logits = self.classifier(inputs)
            loss = self.criterion(logits, labels)

            _, predicted = torch.max(logits, 1)
            accuracy = (predicted == labels).float().mean().item()

            total_loss += loss.item()
            total_acc += accuracy
            n_batches += 1

        return {
            'loss': total_loss / n_batches,
            'accuracy': total_acc / n_batches,
        }
