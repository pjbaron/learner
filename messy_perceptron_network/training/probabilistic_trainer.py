"""
Probabilistic resource-based trainer.

Simple idea:
- Each parameter has resource Y (0 to X)
- Update probability = f(Y)
- Each update consumes Y
- Y recovers over time

Purpose: Prevent oscillatory traps where contradictory gradients flip weights back and forth.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict


class ProbabilisticResourceTrainer:
    """
    Trainer with probabilistic updates based on plasticity resources.

    Prevents oscillations by making frequently-updated parameters harder to change.
    """

    def __init__(self,
                 network,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module = None,
                 initial_resource: float = 1.0,
                 depletion_rate: float = 0.1,
                 recovery_rate: float = 0.01,
                 gradient_clip_norm: float = 1.0,
                 device: str = 'cpu'):
        """
        Initialize trainer.

        Args:
            network: Network with parameters to train
            optimizer: Optimizer for parameters
            criterion: Loss function
            initial_resource: Starting resource level X (default: 1.0)
            depletion_rate: How much resource consumed per update
            recovery_rate: How much resource recovers per step
            gradient_clip_norm: Max gradient norm
            device: Device to train on
        """
        self.network = network
        self.optimizer = optimizer
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.gradient_clip_norm = gradient_clip_norm
        self.device = device
        self.step_count = 0

        # Resource tracking
        self.initial_resource = initial_resource
        self.depletion_rate = depletion_rate
        self.recovery_rate = recovery_rate

        # Initialize resources for each parameter
        self.param_resources = {}
        for name, param in network.named_parameters():
            if param.requires_grad:
                self.param_resources[name] = torch.ones_like(param.data) * initial_resource

        # Move to device
        self.network.to(device)

    def train_step(self, inputs: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Single training step with probabilistic updates.

        Args:
            inputs: (batch, input_dim)
            labels: (batch,) class labels

        Returns:
            Dictionary with metrics
        """
        self.network.train()
        self.optimizer.zero_grad()

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # Forward pass
        logits = self.network(inputs)
        loss = self.criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.network.parameters(),
            self.gradient_clip_norm
        )

        # CRITICAL: Probabilistic gradient masking based on resources
        self._apply_probabilistic_updates()

        # Update weights
        self.optimizer.step()

        # Deplete resources for updated parameters
        self._deplete_resources()

        # Recover resources
        self._recover_resources()

        # Compute accuracy
        _, predicted = torch.max(logits, 1)
        accuracy = (predicted == labels).float().mean().item()

        # Get resource stats
        resource_stats = self._get_resource_stats()

        self.step_count += 1

        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'grad_norm': grad_norm.item(),
            'resource_mean': resource_stats['mean'],
            'resource_min': resource_stats['min'],
        }

    def _apply_probabilistic_updates(self):
        """
        Mask gradients probabilistically based on resource levels.

        Higher resources → higher probability of update
        Lower resources → lower probability of update (avoid oscillations)
        """
        for name, param in self.network.named_parameters():
            if param.grad is not None and name in self.param_resources:
                resources = self.param_resources[name]

                # Update probability = resource level (normalized to 0-1)
                update_prob = resources.clamp(0.0, 1.0)

                # Bernoulli sampling: each parameter independently decides to update
                update_mask = torch.bernoulli(update_prob)

                # Mask gradients
                param.grad *= update_mask

    def _deplete_resources(self):
        """Deplete resources for parameters that just updated."""
        with torch.no_grad():
            for name, param in self.network.named_parameters():
                if param.grad is not None and name in self.param_resources:
                    # Deplete proportional to gradient magnitude
                    depletion = self.depletion_rate * param.grad.abs()
                    self.param_resources[name] -= depletion
                    self.param_resources[name].clamp_(min=0.0)

    def _recover_resources(self):
        """Passively recover resources toward initial level."""
        with torch.no_grad():
            for name in self.param_resources:
                self.param_resources[name] += self.recovery_rate
                self.param_resources[name].clamp_(max=self.initial_resource)

    def _get_resource_stats(self) -> Dict[str, float]:
        """Get statistics about current resource levels."""
        all_resources = torch.cat([r.flatten() for r in self.param_resources.values()])
        return {
            'mean': all_resources.mean().item(),
            'std': all_resources.std().item(),
            'min': all_resources.min().item(),
            'max': all_resources.max().item(),
        }

    def train_epoch(self, dataloader: DataLoader, verbose: bool = True) -> Dict[str, float]:
        """Train for one epoch."""
        total_loss = 0.0
        total_acc = 0.0
        total_resource = 0.0
        n_batches = 0

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            metrics = self.train_step(inputs, labels)

            total_loss += metrics['loss']
            total_acc += metrics['accuracy']
            total_resource += metrics['resource_mean']
            n_batches += 1

            if verbose and batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}: "
                      f"Loss={metrics['loss']:.4f}, "
                      f"Acc={metrics['accuracy']:.4f}, "
                      f"Resource={metrics['resource_mean']:.3f}")

        return {
            'loss': total_loss / n_batches,
            'accuracy': total_acc / n_batches,
            'resource_mean': total_resource / n_batches,
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on a dataset."""
        self.network.eval()

        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            logits = self.network(inputs)
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
