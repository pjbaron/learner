"""
Fast vectorized network with ACTIVATION-DRIVEN plasticity modulation (HOPE design).

Key difference from resource-based approach:
- Plasticity rates (α) computed from network activations during forward pass
- Different inputs activate different modulators → different plasticity patterns
- Enables emergent task routing and continual learning
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

from .messy_graph import create_messy_graph


class FastMessyPerceptronNetwork(nn.Module):
    """
    Fast vectorized messy perceptron network with activation-driven plasticity modulation.

    Key features (HOPE design):
    - Three connection types: signal, threshold modulation, plasticity modulation
    - Plasticity rate α computed from activations via plasticity modulation connections
    - Each perceptron's learning rate = base_lr × α (computed during forward pass)
    - Emergent task routing: different inputs → different modulators → different plasticity
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
        Initialize network with activation-driven plasticity modulation.

        Args:
            n_perceptrons: Number of perceptrons
            avg_degree: Average connections per perceptron
            n_input_perceptrons: Number receiving external input
            n_output_perceptrons: Number providing output
            settling_iterations: Settling iterations for forward pass
            beta: EMA decay for activation history
            default_plasticity: Default α when no plasticity modulation active
            seed: Random seed
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

        # Learnable thresholds (one per perceptron)
        self.thresholds = nn.Parameter(torch.randn(n_perceptrons) * 0.1)

        # Generate messy graph
        print(f"Generating messy graph with {n_perceptrons} perceptrons and avg degree {avg_degree}...")
        edges = create_messy_graph(n_perceptrons, avg_degree, seed)

        # Build sparse matrices for each connection type
        self._build_matrices(edges)

        # Select input/output perceptrons (random subsets, non-overlapping)
        all_indices = list(range(n_perceptrons))
        np.random.shuffle(all_indices)
        self.input_perceptron_indices = torch.tensor(all_indices[:n_input_perceptrons], dtype=torch.long)
        self.output_perceptron_indices = torch.tensor(all_indices[n_input_perceptrons:n_input_perceptrons+n_output_perceptrons], dtype=torch.long)

        # Storage for per-perceptron plasticity rates (computed during forward pass)
        self.register_buffer('plasticity_rates', torch.ones(n_perceptrons) * default_plasticity)

        print(f"Network created: {n_perceptrons} perceptrons, "
              f"{len(edges['signal']) + len(edges['threshold_mod']) + len(edges['plasticity_mod'])} connections")
        print(f"Input perceptrons: {n_input_perceptrons}, Output perceptrons: {n_output_perceptrons}")

    def _build_matrices(self, edges: Dict[str, List[tuple]]):
        """Build sparse adjacency matrices for each connection type."""

        # Signal connections (80%)
        signal_edges = edges['signal']
        if signal_edges:
            signal_src, signal_dst = zip(*signal_edges)
            self.signal_indices = torch.tensor([signal_src, signal_dst], dtype=torch.long)
            # Initialize weights (Xavier for signal connections)
            n_signal = len(signal_edges)
            self.signal_weights = nn.Parameter(torch.randn(n_signal) * np.sqrt(2.0 / self.n_perceptrons))
        else:
            self.signal_indices = torch.zeros((2, 0), dtype=torch.long)
            self.signal_weights = nn.Parameter(torch.zeros(0))

        # Threshold modulation connections (15%)
        threshold_edges = edges['threshold_mod']
        if threshold_edges:
            threshold_src, threshold_dst = zip(*threshold_edges)
            self.threshold_indices = torch.tensor([threshold_src, threshold_dst], dtype=torch.long)
            # Small random initialization for modulation
            n_threshold = len(threshold_edges)
            self.threshold_weights = nn.Parameter(torch.randn(n_threshold) * 0.01)
        else:
            self.threshold_indices = torch.zeros((2, 0), dtype=torch.long)
            self.threshold_weights = nn.Parameter(torch.zeros(0))

        # Plasticity modulation connections (5%)
        plasticity_edges = edges['plasticity_mod']
        if plasticity_edges:
            plasticity_src, plasticity_dst = zip(*plasticity_edges)
            self.plasticity_indices = torch.tensor([plasticity_src, plasticity_dst], dtype=torch.long)
            # Initialize near default plasticity (0.5 → sigmoid^-1(0.5) = 0)
            n_plasticity = len(plasticity_edges)
            self.plasticity_weights = nn.Parameter(torch.randn(n_plasticity) * 0.1)
        else:
            self.plasticity_indices = torch.zeros((2, 0), dtype=torch.long)
            self.plasticity_weights = nn.Parameter(torch.zeros(0))

        print(f"  Signal: {len(signal_edges)} connections")
        print(f"  Threshold modulation: {len(threshold_edges)} connections")
        print(f"  Plasticity modulation: {len(plasticity_edges)} connections")

    def forward(self, inputs: torch.Tensor, return_history=False):
        """
        Forward pass with iterative settling.

        Computes plasticity rates (α) from activations via plasticity modulation connections.

        Args:
            inputs: (batch, n_input_perceptrons)
            return_history: If True, return activation history

        Returns:
            outputs: (batch, n_output_perceptrons)
            activation_history (optional): List of (batch, n_perceptrons) per iteration
        """
        batch_size = inputs.shape[0]
        device = inputs.device

        if device != self.thresholds.device:
            self.to(device)

        # Initialize activations
        activations = torch.zeros(batch_size, self.n_perceptrons, device=device)

        # Create sparse matrices for efficient computation
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

        plasticity_adj = torch.sparse_coo_tensor(
            self.plasticity_indices.to(device),
            self.plasticity_weights,
            (self.n_perceptrons, self.n_perceptrons),
            device=device
        )

        activation_history = [] if return_history else None

        # Prepare strong input drive (preserves gradients)
        input_drive = torch.zeros(batch_size, self.n_perceptrons, device=device)
        input_drive[:, self.input_perceptron_indices] = inputs[:, :self.n_input_perceptrons] * 10.0

        # Compute plasticity modulation ONCE at start (simpler, allows gradients to flow)
        # Initialize with small random activations to bootstrap
        bootstrap_activations = torch.randn_like(activations) * 0.1
        plasticity_input = torch.sparse.mm(plasticity_adj, bootstrap_activations.t()).t()
        plasticity_rates = torch.sigmoid(plasticity_input)  # (batch, n_perceptrons)

        # Average across batch and expand for broadcasting
        avg_plasticity = plasticity_rates.mean(dim=0)  # (n_perceptrons,)

        # CRITICAL: Apply plasticity modulation to connection weights IN THE FORWARD PASS
        # This allows gradients to flow to both signal_weights AND plasticity_weights!
        target_indices = self.signal_indices[1]  # Target perceptron for each connection
        plasticity_mask = avg_plasticity[target_indices]  # (n_connections,)
        modulated_signal_weights = self.signal_weights * plasticity_mask

        # Rebuild signal matrix with modulated weights
        signal_adj = torch.sparse_coo_tensor(
            self.signal_indices.to(device),
            modulated_signal_weights,
            (self.n_perceptrons, self.n_perceptrons),
            device=device
        )

        # Settling iterations
        for iteration in range(self.settling_iterations):
            # Signal inputs with MODULATED weights: z = Σ((w_signal × α) × a)
            z = torch.sparse.mm(signal_adj, activations.t()).t()

            # Threshold modulation: Δθ = Σ(w_threshold × a)
            delta_theta = torch.sparse.mm(threshold_adj, activations.t()).t()

            # Compute effective threshold
            effective_thresholds = self.thresholds.unsqueeze(0) + delta_theta

            # Update activations with input drive
            activations = torch.tanh(z + input_drive - effective_thresholds)

            if return_history:
                activation_history.append(activations.clone())

        # Store plasticity rates for monitoring (detach since we don't need gradients to flow through this)
        self.plasticity_rates = avg_plasticity.detach()

        # Extract outputs
        outputs = activations[:, self.output_perceptron_indices]

        if return_history:
            return outputs, activation_history
        else:
            return outputs

    def apply_plasticity_modulation(self):
        """
        Placeholder - plasticity modulation now happens in forward pass.

        Kept for compatibility with trainer, but does nothing.
        Plasticity is applied by scaling weights: effective_w = w × α
        Gradients automatically flow to both signal_weights and plasticity_weights.
        """
        pass  # No-op: plasticity applied in forward pass

    def get_plasticity_stats(self):
        """Get statistics about current plasticity rates."""
        return {
            'mean': self.plasticity_rates.mean().item(),
            'std': self.plasticity_rates.std().item(),
            'min': self.plasticity_rates.min().item(),
            'max': self.plasticity_rates.max().item(),
        }
