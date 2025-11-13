"""
Connection types for the messy perceptron network.

Three types of connections:
1. SignalConnection - Standard weighted connections for computation
2. ThresholdModulationConnection - Modulates target perceptron's threshold
3. PlasticityModulationConnection - Modulates target perceptron's learning rate
"""

import torch
import torch.nn as nn
from enum import Enum


class ConnectionType(Enum):
    """Enum for the three connection types."""
    SIGNAL = "signal"
    THRESHOLD_MODULATION = "threshold_mod"
    PLASTICITY_MODULATION = "plasticity_mod"


class Connection:
    """
    A weighted connection between two perceptrons.

    Each connection has:
    - source: Source perceptron
    - target: Target perceptron
    - weight: Learnable weight parameter
    - connection_type: One of the three connection types
    """

    def __init__(self, source, target, connection_type, initial_weight=None):
        """
        Create a connection between two perceptrons.

        Args:
            source: Source perceptron
            target: Target perceptron
            connection_type: ConnectionType enum value
            initial_weight: Optional initial weight value (if None, uses default initialization)
        """
        self.source = source
        self.target = target
        self.connection_type = connection_type

        # Initialize weight based on connection type
        if initial_weight is not None:
            self.weight = nn.Parameter(torch.tensor([initial_weight], dtype=torch.float32))
        else:
            self.weight = self._initialize_weight()

    def _initialize_weight(self):
        """
        Initialize weight based on connection type.

        Signal connections: Xavier/Glorot initialization
        Threshold modulation: Small random values (scale 0.01)
        Plasticity modulation: Initialize near 0 (to start near default plasticity of 0.5)
        """
        if self.connection_type == ConnectionType.SIGNAL:
            # Xavier initialization for signal connections
            weight = torch.randn(1) * 0.1
        elif self.connection_type == ConnectionType.THRESHOLD_MODULATION:
            # Small random values for threshold modulation
            weight = torch.randn(1) * 0.01
        elif self.connection_type == ConnectionType.PLASTICITY_MODULATION:
            # Initialize near 0 so sigmoid(0) ≈ 0.5 (default plasticity)
            weight = torch.randn(1) * 0.01
        else:
            raise ValueError(f"Unknown connection type: {self.connection_type}")

        return nn.Parameter(weight)

    def get_weight_value(self):
        """Return the current weight value as a Python float."""
        return self.weight.item()

    def __repr__(self):
        return (f"Connection({self.source.id} -> {self.target.id}, "
                f"type={self.connection_type.value}, "
                f"weight={self.get_weight_value():.4f})")


class ConnectionManager:
    """
    Manages all connections in the network.

    This class handles:
    - Creating connections between perceptrons
    - Organizing connections by type
    - Collecting all learnable parameters
    - Providing statistics about the network connectivity
    """

    def __init__(self):
        """Initialize empty connection manager."""
        self.connections = []
        self.connections_by_type = {
            ConnectionType.SIGNAL: [],
            ConnectionType.THRESHOLD_MODULATION: [],
            ConnectionType.PLASTICITY_MODULATION: []
        }

    def add_connection(self, source, target, connection_type, initial_weight=None):
        """
        Add a connection between two perceptrons.

        Args:
            source: Source perceptron
            target: Target perceptron
            connection_type: ConnectionType enum value
            initial_weight: Optional initial weight

        Returns:
            The created Connection object
        """
        connection = Connection(source, target, connection_type, initial_weight)
        self.connections.append(connection)
        self.connections_by_type[connection_type].append(connection)

        # Register the connection with the target perceptron
        if connection_type == ConnectionType.SIGNAL:
            target.add_signal_connection(source, connection.weight)
        elif connection_type == ConnectionType.THRESHOLD_MODULATION:
            target.add_threshold_modulation_connection(source, connection.weight)
        elif connection_type == ConnectionType.PLASTICITY_MODULATION:
            target.add_plasticity_modulation_connection(source, connection.weight)

        return connection

    def get_all_weights(self):
        """Return list of all weight parameters for optimization."""
        return [conn.weight for conn in self.connections]

    def get_statistics(self):
        """
        Get statistics about the network connectivity.

        Returns:
            Dictionary with connectivity statistics
        """
        total = len(self.connections)
        stats = {
            'total_connections': total,
            'signal_connections': len(self.connections_by_type[ConnectionType.SIGNAL]),
            'threshold_modulation_connections': len(self.connections_by_type[ConnectionType.THRESHOLD_MODULATION]),
            'plasticity_modulation_connections': len(self.connections_by_type[ConnectionType.PLASTICITY_MODULATION]),
        }

        if total > 0:
            stats['signal_percentage'] = 100.0 * stats['signal_connections'] / total
            stats['threshold_mod_percentage'] = 100.0 * stats['threshold_modulation_connections'] / total
            stats['plasticity_mod_percentage'] = 100.0 * stats['plasticity_modulation_connections'] / total

        return stats

    def get_connections_by_type(self, connection_type):
        """Get all connections of a specific type."""
        return self.connections_by_type[connection_type]

    def __len__(self):
        """Return total number of connections."""
        return len(self.connections)

    def __repr__(self):
        stats = self.get_statistics()
        return (f"ConnectionManager({stats['total_connections']} connections: "
                f"{stats.get('signal_percentage', 0):.1f}% signal, "
                f"{stats.get('threshold_mod_percentage', 0):.1f}% threshold_mod, "
                f"{stats.get('plasticity_mod_percentage', 0):.1f}% plasticity_mod)")
