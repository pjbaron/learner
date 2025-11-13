"""
Fast vectorized implementation of the messy perceptron network.

Uses sparse matrix operations for efficient GPU/CPU computation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict

from .messy_graph import create_messy_graph


class FastMessyPerceptronNetwork(nn.Module):
    """
    Fast vectorized version of the messy perceptron network.

    Uses sparse adjacency matrices for all three connection types,
    allowing batch computation of all perceptrons in parallel.
    """

    def __init__(self,
                 n_perceptrons=2000,
                 avg_degree=30,
                 n_input_perceptrons=200,
                 n_output_perceptrons=200,
                 settling_iterations=7,
                 beta=0.9,
                 default_plasticity=0.5,
                 seed=None):
        """
        Initialize the fast messy perceptron network.

        Args:
            n_perceptrons: Number of perceptrons (default: 2000)
            avg_degree: Average connections per perceptron (default: 30)
            n_input_perceptrons: Number of perceptrons that receive external input (default: 200)
            n_output_perceptrons: Number of perceptrons that provide output (default: 200)
            settling_iterations: Number of iterations for activation settling (default: 7)
            beta: EMA decay rate for activation history (default: 0.9)
            default_plasticity: Default plasticity rate (default: 0.5)
            seed: Random seed for reproducibility
        """
        super(FastMessyPerceptronNetwork, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.n_perceptrons = n_perceptrons
        self.n_input_perceptrons = n_input_perceptrons
        self.n_output_perceptrons = n_output_perceptrons
        self.settling_iterations = settling_iterations
        self.beta = beta
        self.default_plasticity = default_plasticity

        # Learnable thresholds for all perceptrons
        self.thresholds = nn.Parameter(torch.randn(n_perceptrons) * 0.1)

        # Generate messy graph structure
        print(f"Generating messy graph with {n_perceptrons} perceptrons and avg degree {avg_degree}...")
        edges = create_messy_graph(n_perceptrons, avg_degree, seed)

        # Build sparse adjacency matrices and weight parameters
        self._build_matrices(edges)

        # Randomly select input and output perceptrons
        all_indices = list(range(n_perceptrons))
        np.random.shuffle(all_indices)

        self.input_perceptron_indices = torch.tensor(all_indices[:n_input_perceptrons], dtype=torch.long)
        self.output_perceptron_indices = torch.tensor(all_indices[:n_output_perceptrons], dtype=torch.long)

        print(f"Network created: {n_perceptrons} perceptrons, "
              f"{len(edges['signal']) + len(edges['threshold_mod']) + len(edges['plasticity_mod'])} connections")
        print(f"Input perceptrons: {n_input_perceptrons}, Output perceptrons: {n_output_perceptrons}")

    def _build_matrices(self, edges: Dict[str, List[tuple]]):
        """
        Build sparse adjacency matrices for each connection type.

        Args:
            edges: Dictionary with 'signal', 'threshold_mod', 'plasticity_mod' edge lists
        """
        n = self.n_perceptrons

        # Build signal connections
        signal_edges = edges['signal']
        if len(signal_edges) > 0:
            signal_src, signal_dst = zip(*signal_edges)
            signal_indices = torch.tensor([signal_src, signal_dst], dtype=torch.long)
            self.signal_indices = signal_indices
            # Xavier initialization for signal weights
            self.signal_weights = nn.Parameter(torch.randn(len(signal_edges)) * 0.1)
        else:
            self.signal_indices = torch.zeros((2, 0), dtype=torch.long)
            self.signal_weights = nn.Parameter(torch.zeros(0))

        # Build threshold modulation connections
        threshold_edges = edges['threshold_mod']
        if len(threshold_edges) > 0:
            threshold_src, threshold_dst = zip(*threshold_edges)
            threshold_indices = torch.tensor([threshold_src, threshold_dst], dtype=torch.long)
            self.threshold_indices = threshold_indices
            # Small initialization for threshold modulation
            self.threshold_weights = nn.Parameter(torch.randn(len(threshold_edges)) * 0.01)
        else:
            self.threshold_indices = torch.zeros((2, 0), dtype=torch.long)
            self.threshold_weights = nn.Parameter(torch.zeros(0))

        # Build plasticity modulation connections (not used in forward pass currently)
        plasticity_edges = edges['plasticity_mod']
        if len(plasticity_edges) > 0:
            plasticity_src, plasticity_dst = zip(*plasticity_edges)
            plasticity_indices = torch.tensor([plasticity_src, plasticity_dst], dtype=torch.long)
            self.plasticity_indices = plasticity_indices
            # Small initialization for plasticity modulation
            self.plasticity_weights = nn.Parameter(torch.randn(len(plasticity_edges)) * 0.01)
        else:
            self.plasticity_indices = torch.zeros((2, 0), dtype=torch.long)
            self.plasticity_weights = nn.Parameter(torch.zeros(0))

        print(f"  Signal: {len(signal_edges)} connections")
        print(f"  Threshold modulation: {len(threshold_edges)} connections")
        print(f"  Plasticity modulation: {len(plasticity_edges)} connections")

    def forward(self, inputs: torch.Tensor, return_history=False):
        """
        Fast vectorized forward pass with iterative settling.

        Args:
            inputs: Input tensor of shape (batch_size, n_input_perceptrons)
            return_history: If True, return activation history for all settling iterations

        Returns:
            outputs: Output tensor of shape (batch_size, n_output_perceptrons)
        """
        batch_size = inputs.shape[0]
        device = inputs.device

        # Move to device if needed
        if device != self.thresholds.device:
            self.to(device)

        # Initialize activations to zero
        activations = torch.zeros(batch_size, self.n_perceptrons, device=device)

        # Set input activations
        activations[:, self.input_perceptron_indices] = inputs[:, :self.n_input_perceptrons]

        # Create sparse adjacency matrices on the correct device
        signal_adj = torch.sparse_coo_tensor(
            self.signal_indices.to(device),
            self.signal_weights.to(device),
            (self.n_perceptrons, self.n_perceptrons)
        )

        threshold_adj = torch.sparse_coo_tensor(
            self.threshold_indices.to(device),
            self.threshold_weights.to(device),
            (self.n_perceptrons, self.n_perceptrons)
        )

        # Settling iterations
        activation_history = [] if return_history else None

        for iteration in range(self.settling_iterations):
            # Compute signal inputs: z = A_signal @ activations^T
            # Shape: (batch, n_perceptrons)
            z = torch.sparse.mm(signal_adj, activations.t()).t()

            # Compute threshold modulation: delta_theta = A_threshold @ activations^T
            delta_theta = torch.sparse.mm(threshold_adj, activations.t()).t()

            # Compute effective thresholds
            effective_thresholds = self.thresholds.unsqueeze(0) + delta_theta

            # Compute new activations with tanh
            new_activations = torch.tanh(z - effective_thresholds)

            # Keep input activations fixed (clone to avoid in-place modification)
            activations = new_activations.clone()
            activations[:, self.input_perceptron_indices] = inputs[:, :self.n_input_perceptrons]

            if return_history:
                activation_history.append(activations.clone())

        # Extract outputs
        outputs = activations[:, self.output_perceptron_indices]

        if return_history:
            return outputs, activation_history
        else:
            return outputs

    def get_activations(self):
        """Get current activations (not maintained in fast version)."""
        return torch.zeros(self.n_perceptrons)

    def get_thresholds(self):
        """Get learned thresholds for all perceptrons."""
        return self.thresholds.data

    def get_plasticity_rates(self):
        """Get plasticity rates (simplified in fast version)."""
        return torch.ones(self.n_perceptrons) * self.default_plasticity

    def count_parameters(self):
        """Count total number of learnable parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_statistics(self):
        """Get network statistics."""
        total_connections = (len(self.signal_weights) +
                           len(self.threshold_weights) +
                           len(self.plasticity_weights))

        return {
            'n_perceptrons': self.n_perceptrons,
            'n_parameters': self.count_parameters(),
            'n_input_perceptrons': self.n_input_perceptrons,
            'n_output_perceptrons': self.n_output_perceptrons,
            'settling_iterations': self.settling_iterations,
            'total_connections': total_connections,
            'signal_connections': len(self.signal_weights),
            'threshold_modulation_connections': len(self.threshold_weights),
            'plasticity_modulation_connections': len(self.plasticity_weights),
        }

    def __repr__(self):
        stats = self.get_statistics()
        return (f"FastMessyPerceptronNetwork(\n"
                f"  perceptrons={stats['n_perceptrons']},\n"
                f"  parameters={stats['n_parameters']},\n"
                f"  connections={stats['total_connections']},\n"
                f"  settling_iterations={stats['settling_iterations']}\n"
                f")")
