"""
Messy Perceptron Network implementation.

Combines perceptrons and connections into a recurrent network with settling dynamics.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict

from .perceptron import Perceptron
from .connection import ConnectionManager, ConnectionType
from .messy_graph import create_messy_graph


class MessyPerceptronNetwork(nn.Module):
    """
    A recurrent neural network of perceptrons with three connection types.

    The network has no layers - just a messy graph of perceptrons with:
    - Signal connections (computation)
    - Threshold modulation connections (dynamic sensitivity)
    - Plasticity modulation connections (meta-learning)

    Forward pass uses iterative settling to handle the recurrent structure.
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
        Initialize the messy perceptron network.

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
        super(MessyPerceptronNetwork, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.n_perceptrons = n_perceptrons
        self.n_input_perceptrons = n_input_perceptrons
        self.n_output_perceptrons = n_output_perceptrons
        self.settling_iterations = settling_iterations

        # Create perceptrons
        self.perceptrons = nn.ModuleList([
            Perceptron(i, beta=beta, default_plasticity=default_plasticity)
            for i in range(n_perceptrons)
        ])

        # Generate messy graph structure
        print(f"Generating messy graph with {n_perceptrons} perceptrons and avg degree {avg_degree}...")
        edges = create_messy_graph(n_perceptrons, avg_degree, seed)

        # Create connection manager and build connections
        self.connection_manager = ConnectionManager()
        self._build_connections(edges)

        # Randomly select input and output perceptrons (can overlap)
        # ~10% of perceptrons for each
        all_indices = list(range(n_perceptrons))
        np.random.shuffle(all_indices)

        self.input_perceptron_indices = all_indices[:n_input_perceptrons]
        self.output_perceptron_indices = all_indices[:n_output_perceptrons]  # Allow overlap

        print(f"Network created: {len(self.perceptrons)} perceptrons, "
              f"{len(self.connection_manager)} connections")
        print(f"Input perceptrons: {n_input_perceptrons}, Output perceptrons: {n_output_perceptrons}")

    def _build_connections(self, edges: Dict[str, List[tuple]]):
        """
        Build connections between perceptrons based on edge list.

        Args:
            edges: Dictionary with 'signal', 'threshold_mod', 'plasticity_mod' edge lists
        """
        # Signal connections
        for src, dst in edges['signal']:
            self.connection_manager.add_connection(
                self.perceptrons[src],
                self.perceptrons[dst],
                ConnectionType.SIGNAL
            )

        # Threshold modulation connections
        for src, dst in edges['threshold_mod']:
            self.connection_manager.add_connection(
                self.perceptrons[src],
                self.perceptrons[dst],
                ConnectionType.THRESHOLD_MODULATION
            )

        # Plasticity modulation connections
        for src, dst in edges['plasticity_mod']:
            self.connection_manager.add_connection(
                self.perceptrons[src],
                self.perceptrons[dst],
                ConnectionType.PLASTICITY_MODULATION
            )

    def reset_state(self):
        """Reset all perceptron states (activations, history, plasticity)."""
        for perceptron in self.perceptrons:
            perceptron.reset_state()

    def forward(self, inputs: torch.Tensor, return_history=False):
        """
        Forward pass with iterative settling.

        Args:
            inputs: Input tensor of shape (batch_size, n_input_perceptrons)
            return_history: If True, return activation history for all settling iterations

        Returns:
            outputs: Output tensor of shape (batch_size, n_output_perceptrons)
            If return_history=True, also returns list of activation snapshots
        """
        batch_size = inputs.shape[0]
        device = inputs.device

        # Move perceptrons to correct device if needed
        if device != self.perceptrons[0].theta.device:
            self.to(device)

        # Store activation history if requested (for BPTT)
        activation_history = [] if return_history else None

        # Process each sample in the batch
        outputs = []
        for batch_idx in range(batch_size):
            # Initialize activations for this sample
            # Start with zeros, but set inputs
            current_activations = torch.zeros(self.n_perceptrons, device=device, requires_grad=True)

            # Set input activations (these don't need gradients from the network)
            with torch.no_grad():
                for i, input_idx in enumerate(self.input_perceptron_indices):
                    if i < inputs.shape[1]:
                        current_activations[input_idx] = inputs[batch_idx, i]

            # Settling iterations - unroll the recurrence for BPTT
            sample_history = []
            for iteration in range(self.settling_iterations):
                # Compute new activations based on current activations
                new_activations = self._compute_single_iteration(current_activations, inputs[batch_idx])

                # Record activations for this iteration
                if return_history:
                    sample_history.append(new_activations)

                current_activations = new_activations

            # Extract output activations
            output_activations = current_activations[self.output_perceptron_indices]
            outputs.append(output_activations)

            if return_history:
                activation_history.append(sample_history)

        # Stack outputs into batch
        output_tensor = torch.stack(outputs)

        if return_history:
            return output_tensor, activation_history
        else:
            return output_tensor

    def _compute_single_iteration(self, prev_activations, input_values):
        """
        Compute one iteration of settling.

        Args:
            prev_activations: Activation values from previous iteration (n_perceptrons,)
            input_values: Input values for this sample

        Returns:
            new_activations: New activation values (n_perceptrons,)
        """
        device = prev_activations.device
        new_activations = []

        # Update all perceptrons based on previous activations
        for p_idx, perceptron in enumerate(self.perceptrons):
            # Compute signal input
            z = torch.zeros(1, device=device, requires_grad=True)
            for source, weight in perceptron.signal_inputs:
                z = z + prev_activations[source.id] * weight.squeeze()

            # Compute threshold modulation
            delta_theta = torch.zeros(1, device=device, requires_grad=True)
            for source, weight in perceptron.threshold_mod_inputs:
                delta_theta = delta_theta + prev_activations[source.id] * weight.squeeze()

            # Compute effective threshold
            effective_threshold = perceptron.theta.squeeze() + delta_theta.squeeze()

            # Compute activation
            activation = torch.tanh(z.squeeze() - effective_threshold)

            # For input perceptrons, use the input value instead
            if p_idx in self.input_perceptron_indices:
                input_idx = self.input_perceptron_indices.index(p_idx)
                if input_idx < len(input_values):
                    activation = input_values[input_idx]

            new_activations.append(activation)

        return torch.stack(new_activations)

    def get_plasticity_rates(self):
        """
        Get current plasticity rates for all perceptrons.

        Returns:
            Tensor of shape (n_perceptrons,) with plasticity rates
        """
        return torch.tensor([p.get_plasticity_rate() for p in self.perceptrons])

    def get_activations(self):
        """
        Get current activations for all perceptrons.

        Returns:
            Tensor of shape (n_perceptrons,) with activations
        """
        return torch.stack([p.activation for p in self.perceptrons]).squeeze()

    def get_thresholds(self):
        """
        Get learned thresholds for all perceptrons.

        Returns:
            Tensor of shape (n_perceptrons,) with thresholds
        """
        return torch.stack([p.theta for p in self.perceptrons]).squeeze()

    def get_all_parameters(self):
        """
        Get all learnable parameters (perceptron thresholds and connection weights).

        Returns:
            List of all parameters for optimization
        """
        params = []
        # Perceptron thresholds
        for perceptron in self.perceptrons:
            params.append(perceptron.theta)
        # Connection weights
        params.extend(self.connection_manager.get_all_weights())
        return params

    def count_parameters(self):
        """Count total number of learnable parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_statistics(self):
        """
        Get network statistics.

        Returns:
            Dictionary with network statistics
        """
        stats = {
            'n_perceptrons': self.n_perceptrons,
            'n_parameters': self.count_parameters(),
            'n_input_perceptrons': self.n_input_perceptrons,
            'n_output_perceptrons': self.n_output_perceptrons,
            'settling_iterations': self.settling_iterations,
        }

        # Add connection statistics
        conn_stats = self.connection_manager.get_statistics()
        stats.update(conn_stats)

        return stats

    def __repr__(self):
        stats = self.get_statistics()
        return (f"MessyPerceptronNetwork(\n"
                f"  perceptrons={stats['n_perceptrons']},\n"
                f"  parameters={stats['n_parameters']},\n"
                f"  connections={stats['total_connections']},\n"
                f"  settling_iterations={stats['settling_iterations']}\n"
                f")")
