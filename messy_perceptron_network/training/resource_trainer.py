"""
Trainer for networks with plasticity resources.

Handles:
- Gradient scaling by plasticity resources
- Resource depletion on updates
- Passive resource recovery
"""

import torch
import torch.nn as nn
import numpy as np


class ResourceBasedTrainer:
    """
    Trainer that manages plasticity resources during learning.

    Training cycle:
    1. Forward pass
    2. Compute loss
    3. Backward pass (compute gradients)
    4. Scale gradients by plasticity resources
    5. Optimizer step
    6. Deplete resources based on updates
    7. Recover resources (passive)
    """

    def __init__(self,
                 network,
                 classifier,
                 base_lr=0.001,
                 optimizer_type='adam',
                 gradient_clip_norm=1.0,
                 device='cpu'):
        """
        Initialize resource-based trainer.

        Args:
            network: Network with plasticity resources
            classifier: Classifier wrapping the network
            base_lr: Base learning rate
            optimizer_type: 'adam' or 'sgd'
            gradient_clip_norm: Max gradient norm
            device: Device for training
        """
        self.network = network
        self.classifier = classifier
        self.base_lr = base_lr
        self.gradient_clip_norm = gradient_clip_norm
        self.device = torch.device(device)

        self.classifier.to(self.device)

        # Optimizer for all parameters
        if optimizer_type.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=base_lr)
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(self.classifier.parameters(), lr=base_lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        self.criterion = nn.CrossEntropyLoss()

        # Training statistics
        self.step_count = 0
        self.training_stats = {
            'losses': [],
            'accuracies': [],
            'resource_stats': [],
        }

    def train_step(self, inputs, labels):
        """
        Single training step with plasticity resources.

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
        loss = self.criterion(logits, labels)

        # Backward pass
        loss.backward()

        # CRITICAL: Scale gradients by plasticity resources
        self.network.apply_plasticity_resources()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.classifier.parameters(),
            self.gradient_clip_norm
        )

        # Update weights
        self.optimizer.step()

        # CRITICAL: Deplete resources based on weight changes
        self.network.deplete_resources(self.base_lr)

        # CRITICAL: Recover resources passively
        self.network.recover_resources()

        # Compute accuracy
        _, predicted = torch.max(logits, 1)
        accuracy = (predicted == labels).float().mean().item()

        # Get resource statistics
        resource_stats = self.network.get_resource_statistics()

        self.step_count += 1

        stats = {
            'loss': loss.item(),
            'accuracy': accuracy,
            'gradient_norm': grad_norm.item(),
            'resource_mean': resource_stats['mean'],
            'resource_min': resource_stats['min'],
        }

        self.training_stats['losses'].append(loss.item())
        self.training_stats['accuracies'].append(accuracy)
        self.training_stats['resource_stats'].append(resource_stats)

        return stats

    def train_epoch(self, dataloader, verbose=True):
        """Train for one epoch."""
        self.classifier.train()

        epoch_losses = []
        epoch_accuracies = []

        for batch_idx, (images, labels) in enumerate(dataloader):
            inputs = images.view(images.size(0), -1)
            stats = self.train_step(inputs, labels)

            epoch_losses.append(stats['loss'])
            epoch_accuracies.append(stats['accuracy'])

            if verbose and batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}: "
                      f"Loss={stats['loss']:.4f}, "
                      f"Acc={stats['accuracy']:.4f}, "
                      f"Resource={stats['resource_mean']:.3f}")

        return {
            'avg_loss': np.mean(epoch_losses),
            'avg_accuracy': np.mean(epoch_accuracies),
        }

    def evaluate(self, dataloader):
        """Evaluate on a dataset."""
        self.classifier.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in dataloader:
                inputs = images.view(images.size(0), -1).to(self.device)
                labels = labels.to(self.device)

                logits = self.classifier(inputs)
                loss = self.criterion(logits, labels)

                total_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(logits, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += inputs.size(0)

        return {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples,
        }

    def get_resource_summary(self):
        """Get summary of resource usage over training."""
        if len(self.training_stats['resource_stats']) == 0:
            return {}

        means = [s['mean'] for s in self.training_stats['resource_stats']]
        mins = [s['min'] for s in self.training_stats['resource_stats']]

        return {
            'current_mean': means[-1],
            'current_min': mins[-1],
            'mean_over_time': means,
            'min_over_time': mins,
        }
