"""
Perceptron implementation with learnable threshold and three connection types.

Each perceptron can receive three types of inputs:
1. Signal connections - standard weighted inputs for computation
2. Threshold modulation connections - dynamically adjust activation threshold
3. Plasticity modulation connections - control learning rate
"""

import torch
import torch.nn as nn
import numpy as np


class Perceptron(nn.Module):
    """
    A single perceptron with learnable threshold and three connection types.

    Internal State:
    - theta: Learnable threshold parameter
    - activation: Current activation value
    - history: Exponential moving average of activations
    - recent_gradient: Gradient from last backward pass
    - plasticity_rate: Current learning rate modulation (computed from inputs)

    Connection Types:
    - signal_inputs: Standard weighted connections for computation
    - threshold_mod_inputs: Connections that modulate the activation threshold
    - plasticity_mod_inputs: Connections that modulate the learning rate
    """

    def __init__(self, perceptron_id, beta=0.9, default_plasticity=0.5):
        """
        Initialize a perceptron.

        Args:
            perceptron_id: Unique identifier for this perceptron
            beta: EMA decay rate for activation history (default: 0.9)
            default_plasticity: Default plasticity rate when no modulation (default: 0.5)
        """
        super(Perceptron, self).__init__()

        self.id = perceptron_id
        self.beta = beta
        self.default_plasticity = default_plasticity

        # Learnable threshold - initialized with small random value
        self.theta = nn.Parameter(torch.randn(1) * 0.1)

        # State variables (not learnable)
        self.register_buffer('activation', torch.zeros(1))
        self.register_buffer('history', torch.zeros(1))
        self.register_buffer('plasticity_rate', torch.ones(1) * default_plasticity)

        # Gradient tracking
        self.recent_gradient = None

        # Connection lists will be populated by the network
        # Each connection is a tuple: (source_perceptron, weight_parameter)
        self.signal_inputs = []
        self.threshold_mod_inputs = []
        self.plasticity_mod_inputs = []

    def add_signal_connection(self, source_perceptron, weight):
        """Add a signal connection from source perceptron with given weight."""
        self.signal_inputs.append((source_perceptron, weight))

    def add_threshold_modulation_connection(self, source_perceptron, weight):
        """Add a threshold modulation connection from source perceptron."""
        self.threshold_mod_inputs.append((source_perceptron, weight))

    def add_plasticity_modulation_connection(self, source_perceptron, weight):
        """Add a plasticity modulation connection from source perceptron."""
        self.plasticity_mod_inputs.append((source_perceptron, weight))

    def compute_activation(self):
        """
        Compute the activation for this perceptron based on all inputs.

        Steps:
        1. Compute signal input: z = Σ(w_i × a_i)
        2. Compute threshold modulation: Δθ = Σ(w_j × a_j)
        3. Compute plasticity modulation: α = σ(Σ(w_k × a_k))
        4. Compute effective threshold: θ_eff = θ + Δθ
        5. Activate: a = tanh(z - θ_eff)
        6. Update history: h ← β×h + (1-β)×a

        Returns:
            activation: The computed activation value
        """
        # 1. Compute signal input
        z = torch.zeros(1, device=self.theta.device)
        for source, weight in self.signal_inputs:
            z += source.activation * weight

        # 2. Compute threshold modulation
        delta_theta = torch.zeros(1, device=self.theta.device)
        for source, weight in self.threshold_mod_inputs:
            delta_theta += source.activation * weight

        # 3. Compute plasticity modulation
        alpha_input = torch.zeros(1, device=self.theta.device)
        for source, weight in self.plasticity_mod_inputs:
            alpha_input += source.activation * weight

        # If no plasticity modulation inputs, use default
        if len(self.plasticity_mod_inputs) > 0:
            self.plasticity_rate = torch.sigmoid(alpha_input)
        else:
            self.plasticity_rate.fill_(self.default_plasticity)

        # 4. Compute effective threshold
        effective_threshold = self.theta + delta_theta

        # 5. Activate with tanh
        new_activation = torch.tanh(z - effective_threshold)
        self.activation = new_activation.detach().clone()

        # 6. Update history with exponential moving average
        self.history = self.beta * self.history + (1 - self.beta) * self.activation

        return new_activation

    def set_activation(self, value):
        """
        Directly set the activation value (used for input perceptrons).

        Args:
            value: The activation value to set
        """
        self.activation = torch.tensor([value], dtype=self.activation.dtype, device=self.activation.device)

    def get_activation(self):
        """Return the current activation value."""
        return self.activation.item()

    def get_plasticity_rate(self):
        """Return the current plasticity rate."""
        return self.plasticity_rate.item()

    def reset_state(self):
        """Reset the perceptron's state (activation, history, plasticity)."""
        self.activation.zero_()
        self.history.zero_()
        self.plasticity_rate.fill_(self.default_plasticity)
        self.recent_gradient = None

    def __repr__(self):
        return (f"Perceptron(id={self.id}, "
                f"theta={self.theta.item():.3f}, "
                f"activation={self.activation.item():.3f}, "
                f"plasticity={self.plasticity_rate.item():.3f})")
