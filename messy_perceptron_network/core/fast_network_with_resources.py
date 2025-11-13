"""
Fast vectorized network with plasticity resources.

Implements biologically-inspired plasticity depletion and recovery:
- Parameters have finite plasticity resources
- Resources deplete with learning (like neuromodulator depletion)
- Resources recover over time (like protein synthesis recovery)
- This naturally protects recently-learned knowledge
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

from .messy_graph import create_messy_graph


class FastMessyPerceptronNetworkWithResources(nn.Module):
    """
    Fast vectorized messy perceptron network with plasticity resources.

    Key features:
    - All parameters have associated plasticity resources (0-1)
    - Resources deplete proportional to weight updates
    - Resources recover passively over time
    - Effective learning rate = base_lr * plasticity_resource
    """

    def __init__(self,
                 n_perceptrons=2000,
                 avg_degree=30,
                 n_input_perceptrons=200,
                 n_output_perceptrons=200,
                 settling_iterations=7,
                 beta=0.9,
                 default_plasticity=0.5,
                 initial_resource=1.0,
                 depletion_rate=0.1,
                 recovery_rate=0.01,
                 seed=None):
        """
        Initialize network with plasticity resources.

        Args:
            n_perceptrons: Number of perceptrons
            avg_degree: Average connections per perceptron
            n_input_perceptrons: Number receiving external input
            n_output_perceptrons: Number providing output
            settling_iterations: Settling iterations for forward pass
            beta: EMA decay for activation history
            default_plasticity: Default plasticity rate
            initial_resource: Initial plasticity resource (default: 1.0)
            depletion_rate: How fast resources deplete (default: 0.1)
            recovery_rate: How fast resources recover per step (default: 0.01)
            seed: Random seed
        """
        super(FastMessyPerceptronNetworkWithResources, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.n_perceptrons = n_perceptrons
        self.n_input_perceptrons = n_input_perceptrons
        self.n_output_perceptrons = n_output_perceptrons
        self.settling_iterations = settling_iterations
        self.beta = beta
        self.default_plasticity = default_plasticity

        # Plasticity resource parameters
        self.initial_resource = initial_resource
        self.depletion_rate = depletion_rate
        self.recovery_rate = recovery_rate

        # Learnable thresholds
        self.thresholds = nn.Parameter(torch.randn(n_perceptrons) * 0.1)

        # Generate graph
        print(f"Generating messy graph with {n_perceptrons} perceptrons and avg degree {avg_degree}...")
        edges = create_messy_graph(n_perceptrons, avg_degree, seed)

        # Build matrices and initialize resources
        self._build_matrices(edges)

        # Select input/output perceptrons
        all_indices = list(range(n_perceptrons))
        np.random.shuffle(all_indices)
        self.input_perceptron_indices = torch.tensor(all_indices[:n_input_perceptrons], dtype=torch.long)
        self.output_perceptron_indices = torch.tensor(all_indices[:n_output_perceptrons], dtype=torch.long)

        print(f"Network created: {n_perceptrons} perceptrons, "
              f"{len(edges['signal']) + len(edges['threshold_mod']) + len(edges['plasticity_mod'])} connections")
        print(f"Plasticity resources: initial={initial_resource}, depletion={depletion_rate}, recovery={recovery_rate}")

    def _build_matrices(self, edges: Dict[str, List[tuple]]):
        """Build sparse matrices and initialize plasticity resources."""
        n = self.n_perceptrons

        # Build signal connections
        signal_edges = edges['signal']
        if len(signal_edges) > 0:
            signal_src, signal_dst = zip(*signal_edges)
            self.signal_indices = torch.tensor([signal_src, signal_dst], dtype=torch.long)
            self.signal_weights = nn.Parameter(torch.randn(len(signal_edges)) * 0.1)
            # Plasticity resources for signal weights
            self.register_buffer('signal_resources',
                               torch.ones(len(signal_edges)) * self.initial_resource)
        else:
            self.signal_indices = torch.zeros((2, 0), dtype=torch.long)
            self.signal_weights = nn.Parameter(torch.zeros(0))
            self.register_buffer('signal_resources', torch.zeros(0))

        # Build threshold modulation
        threshold_edges = edges['threshold_mod']
        if len(threshold_edges) > 0:
            threshold_src, threshold_dst = zip(*threshold_edges)
            self.threshold_indices = torch.tensor([threshold_src, threshold_dst], dtype=torch.long)
            self.threshold_weights = nn.Parameter(torch.randn(len(threshold_edges)) * 0.01)
            self.register_buffer('threshold_resources',
                               torch.ones(len(threshold_edges)) * self.initial_resource)
        else:
            self.threshold_indices = torch.zeros((2, 0), dtype=torch.long)
            self.threshold_weights = nn.Parameter(torch.zeros(0))
            self.register_buffer('threshold_resources', torch.zeros(0))

        # Build plasticity modulation (not actively used but kept for structure)
        plasticity_edges = edges['plasticity_mod']
        if len(plasticity_edges) > 0:
            plasticity_src, plasticity_dst = zip(*plasticity_edges)
            self.plasticity_indices = torch.tensor([plasticity_src, plasticity_dst], dtype=torch.long)
            self.plasticity_weights = nn.Parameter(torch.randn(len(plasticity_edges)) * 0.01)
            self.register_buffer('plasticity_resources',
                               torch.ones(len(plasticity_edges)) * self.initial_resource)
        else:
            self.plasticity_indices = torch.zeros((2, 0), dtype=torch.long)
            self.plasticity_weights = nn.Parameter(torch.zeros(0))
            self.register_buffer('plasticity_resources', torch.zeros(0))

        # Plasticity resources for thresholds
        self.register_buffer('threshold_param_resources',
                           torch.ones(n) * self.initial_resource)

        print(f"  Signal: {len(signal_edges)} connections")
        print(f"  Threshold modulation: {len(threshold_edges)} connections")
        print(f"  Plasticity modulation: {len(plasticity_edges)} connections")

    def forward(self, inputs: torch.Tensor, return_history=False):
        """Forward pass (same as FastMessyPerceptronNetwork)."""
        batch_size = inputs.shape[0]
        device = inputs.device

        if device != self.thresholds.device:
            self.to(device)

        # Initialize activations
        activations = torch.zeros(batch_size, self.n_perceptrons, device=device)

        # Create sparse matrices
        signal_adj = torch.sparse_coo_tensor(
            self.signal_indices.to(device),
            self.signal_weights,
            (self.n_perceptrons, self.n_perceptrons),
            device=device
        )

        threshold_adj = torch.sparse_coo_tensor(
            self.threshold_indices.to(device),
            self.threshold_weights,
            (self.n_perceptrons, self.n_perceptrons),
            device=device
        )

        activation_history = [] if return_history else None

        # Prepare strong input drive (preserves gradients unlike clamping)
        input_drive = torch.zeros(batch_size, self.n_perceptrons, device=device)
        input_drive[:, self.input_perceptron_indices] = inputs[:, :self.n_input_perceptrons] * 10.0

        # Settling iterations
        for iteration in range(self.settling_iterations):
            z = torch.sparse.mm(signal_adj, activations.t()).t()
            delta_theta = torch.sparse.mm(threshold_adj, activations.t()).t()
            effective_thresholds = self.thresholds.unsqueeze(0) + delta_theta

            # Add input drive instead of clamping (preserves gradient flow)
            activations = torch.tanh(z + input_drive - effective_thresholds)

            if return_history:
                activation_history.append(activations.clone())

        outputs = activations[:, self.output_perceptron_indices]

        if return_history:
            return outputs, activation_history
        else:
            return outputs

    def apply_plasticity_resources(self):
        """
        Apply plasticity resources to gradients.

        Call this AFTER loss.backward() but BEFORE optimizer.step()
        """
        if self.signal_weights.grad is not None:
            self.signal_weights.grad *= self.signal_resources

        if self.threshold_weights.grad is not None:
            self.threshold_weights.grad *= self.threshold_resources

        if self.plasticity_weights.grad is not None:
            self.plasticity_weights.grad *= self.plasticity_resources

        if self.thresholds.grad is not None:
            self.thresholds.grad *= self.threshold_param_resources

    def deplete_resources(self, learning_rate):
        """
        Deplete plasticity resources based on weight changes.

        Call this AFTER optimizer.step()

        Note: We store previous weights to compute actual weight changes,
        which better reflects true plasticity usage than gradient magnitude.
        """
        with torch.no_grad():
            # Initialize previous weight storage if not exists
            if not hasattr(self, '_prev_signal_weights'):
                self._prev_signal_weights = self.signal_weights.data.clone()
                self._prev_threshold_weights = self.threshold_weights.data.clone()
                self._prev_plasticity_weights = self.plasticity_weights.data.clone()
                self._prev_thresholds = self.thresholds.data.clone()
                return  # Skip depletion on first call

            # Deplete based on actual weight changes
            weight_change = torch.abs(self.signal_weights.data - self._prev_signal_weights)
            depletion = self.depletion_rate * weight_change
            self.signal_resources -= depletion
            self.signal_resources.clamp_(min=0.0)
            self._prev_signal_weights = self.signal_weights.data.clone()

            weight_change = torch.abs(self.threshold_weights.data - self._prev_threshold_weights)
            depletion = self.depletion_rate * weight_change
            self.threshold_resources -= depletion
            self.threshold_resources.clamp_(min=0.0)
            self._prev_threshold_weights = self.threshold_weights.data.clone()

            weight_change = torch.abs(self.plasticity_weights.data - self._prev_plasticity_weights)
            depletion = self.depletion_rate * weight_change
            self.plasticity_resources -= depletion
            self.plasticity_resources.clamp_(min=0.0)
            self._prev_plasticity_weights = self.plasticity_weights.data.clone()

            weight_change = torch.abs(self.thresholds.data - self._prev_thresholds)
            depletion = self.depletion_rate * weight_change
            self.threshold_param_resources -= depletion
            self.threshold_param_resources.clamp_(min=0.0)
            self._prev_thresholds = self.thresholds.data.clone()

    def recover_resources(self):
        """
        Recover plasticity resources over time.

        Call this at the end of each training step.
        """
        with torch.no_grad():
            self.signal_resources += self.recovery_rate
            self.signal_resources.clamp_(max=1.0)

            self.threshold_resources += self.recovery_rate
            self.threshold_resources.clamp_(max=1.0)

            self.plasticity_resources += self.recovery_rate
            self.plasticity_resources.clamp_(max=1.0)

            self.threshold_param_resources += self.recovery_rate
            self.threshold_param_resources.clamp_(max=1.0)

    def get_resource_statistics(self):
        """Get statistics about current plasticity resources."""
        all_resources = torch.cat([
            self.signal_resources,
            self.threshold_resources,
            self.plasticity_resources,
            self.threshold_param_resources
        ])

        return {
            'mean': all_resources.mean().item(),
            'std': all_resources.std().item(),
            'min': all_resources.min().item(),
            'max': all_resources.max().item(),
            'signal_mean': self.signal_resources.mean().item(),
            'threshold_mean': self.threshold_resources.mean().item(),
        }

    def count_parameters(self):
        """Count total learnable parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_statistics(self):
        """Get network statistics."""
        total_connections = (len(self.signal_weights) +
                           len(self.threshold_weights) +
                           len(self.plasticity_weights))

        stats = {
            'n_perceptrons': self.n_perceptrons,
            'n_parameters': self.count_parameters(),
            'total_connections': total_connections,
        }
        stats.update(self.get_resource_statistics())

        return stats

    def __repr__(self):
        stats = self.get_statistics()
        return (f"FastMessyPerceptronNetworkWithResources(\n"
                f"  perceptrons={stats['n_perceptrons']},\n"
                f"  parameters={stats['n_parameters']},\n"
                f"  connections={stats['total_connections']},\n"
                f"  resource_mean={stats['mean']:.3f}\n"
                f")")
