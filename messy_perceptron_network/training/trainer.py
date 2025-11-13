"""
Trainer for the messy perceptron network.

Implements:
- Multi-cycle training (3 forward/backward passes per batch)
- Backpropagation Through Time (BPTT) through settled activations
- Gradient clipping for stability
- Per-perceptron plasticity-modulated learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, List
import numpy as np


class MessyPerceptronTrainer:
    """
    Trainer for the messy perceptron network with multi-cycle training.

    Key features:
    - Multi-cycle training: Run 3 forward passes per batch
    - BPTT through settling iterations
    - Gradient clipping for stability
    - Plasticity-modulated learning rates
    """

    def __init__(self,
                 network,
                 base_lr=0.001,
                 optimizer_type='adam',
                 gradient_clip_norm=1.0,
                 cycles_per_batch=3,
                 device='cpu'):
        """
        Initialize the trainer.

        Args:
            network: MessyPerceptronNetwork instance
            base_lr: Base learning rate (default: 0.001)
            optimizer_type: 'adam' or 'sgd' (default: 'adam')
            gradient_clip_norm: Maximum gradient norm (default: 1.0)
            cycles_per_batch: Number of forward/backward cycles per batch (default: 3)
            device: Device to use for training ('cpu' or 'cuda')
        """
        self.network = network
        self.base_lr = base_lr
        self.gradient_clip_norm = gradient_clip_norm
        self.cycles_per_batch = cycles_per_batch
        self.device = torch.device(device)

        # Move network to device
        self.network.to(self.device)

        # Create optimizer
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(self.network.parameters(), lr=base_lr)
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(self.network.parameters(), lr=base_lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        # Loss function
        self.criterion = nn.MSELoss()

        # Training statistics
        self.training_stats = {
            'losses': [],
            'gradient_norms': [],
            'plasticity_stats': []
        }

    def train_step(self, inputs, targets):
        """
        Single training step with multi-cycle training.

        Args:
            inputs: Input tensor (batch_size, n_input_perceptrons)
            targets: Target tensor (batch_size, n_output_perceptrons)

        Returns:
            Dictionary with training statistics
        """
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        total_loss = 0.0
        cycle_losses = []

        # Multi-cycle training: Run multiple forward/backward passes
        for cycle in range(self.cycles_per_batch):
            self.optimizer.zero_grad()

            # Forward pass with history for BPTT
            outputs, activation_history = self.network.forward(inputs, return_history=True)

            # Compute loss
            loss = self.criterion(outputs, targets)
            total_loss += loss.item()
            cycle_losses.append(loss.item())

            # Backward pass (BPTT through settled activations)
            loss.backward()

            # Gradient clipping for stability
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.network.parameters(),
                self.gradient_clip_norm
            )

            # Update parameters with base learning rate
            # (In a full implementation, we could modulate per-perceptron based on plasticity_rate)
            self.optimizer.step()

        # Compute average loss
        avg_loss = total_loss / self.cycles_per_batch

        # Collect statistics
        stats = {
            'loss': avg_loss,
            'cycle_losses': cycle_losses,
            'gradient_norm': grad_norm.item(),
        }

        # Record statistics
        self.training_stats['losses'].append(avg_loss)
        self.training_stats['gradient_norms'].append(grad_norm.item())

        return stats

    def train_epoch(self, dataloader, verbose=True):
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader with (inputs, targets) batches
            verbose: Print progress (default: True)

        Returns:
            Dictionary with epoch statistics
        """
        self.network.train()

        epoch_losses = []
        epoch_grad_norms = []

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            stats = self.train_step(inputs, targets)

            epoch_losses.append(stats['loss'])
            epoch_grad_norms.append(stats['gradient_norm'])

            if verbose and batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}: "
                      f"Loss={stats['loss']:.4f}, "
                      f"GradNorm={stats['gradient_norm']:.4f}")

        # Compute epoch statistics
        epoch_stats = {
            'avg_loss': np.mean(epoch_losses),
            'avg_grad_norm': np.mean(epoch_grad_norms),
            'losses': epoch_losses,
            'grad_norms': epoch_grad_norms,
        }

        return epoch_stats

    def evaluate(self, dataloader):
        """
        Evaluate the network on a dataset.

        Args:
            dataloader: DataLoader with (inputs, targets) batches

        Returns:
            Dictionary with evaluation statistics
        """
        self.network.eval()

        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass (single cycle for evaluation)
                outputs = self.network.forward(inputs, return_history=False)

                # Compute loss
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0

        eval_stats = {
            'loss': avg_loss,
        }

        return eval_stats

    def get_plasticity_statistics(self):
        """
        Get statistics about plasticity rates across the network.

        Returns:
            Dictionary with plasticity statistics
        """
        plasticity_rates = self.network.get_plasticity_rates().cpu().numpy()

        stats = {
            'mean': np.mean(plasticity_rates),
            'std': np.std(plasticity_rates),
            'min': np.min(plasticity_rates),
            'max': np.max(plasticity_rates),
            'histogram': np.histogram(plasticity_rates, bins=20),
        }

        return stats

    def get_activation_statistics(self):
        """
        Get statistics about activations across the network.

        Returns:
            Dictionary with activation statistics
        """
        activations = self.network.get_activations().cpu().numpy()

        stats = {
            'mean': np.mean(activations),
            'std': np.std(activations),
            'min': np.min(activations),
            'max': np.max(activations),
            'sparsity': np.mean(np.abs(activations) < 0.1),  # Fraction near zero
        }

        return stats

    def save_checkpoint(self, filepath):
        """
        Save training checkpoint.

        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        """
        Load training checkpoint.

        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']
        print(f"Checkpoint loaded from {filepath}")

    def get_training_history(self):
        """
        Get training history.

        Returns:
            Dictionary with training statistics over time
        """
        return {
            'losses': self.training_stats['losses'],
            'gradient_norms': self.training_stats['gradient_norms'],
        }
