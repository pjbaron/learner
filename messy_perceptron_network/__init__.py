"""
Messy Perceptron Network: A biologically-inspired neural network for continual learning.

This package implements a proof-of-concept neural network architecture that achieves
continual learning through emergent nested optimization. The architecture consists of
a single component type - perceptrons - connected in a messy, recurrent graph where
different connection types create emergent hierarchies of learning.

Key Components:
- Perceptrons with learnable thresholds
- Three connection types (signal, threshold modulation, plasticity modulation)
- Messy recurrent graph topology
- Iterative settling for forward pass
- Multi-cycle training with BPTT

Usage:
    from messy_perceptron_network import MessyPerceptronNetwork, MessyPerceptronTrainer

    # Create network
    network = MessyPerceptronNetwork(
        n_perceptrons=2000,
        avg_degree=30,
        n_input_perceptrons=200,
        n_output_perceptrons=200
    )

    # Create trainer
    trainer = MessyPerceptronTrainer(network, base_lr=0.001)

    # Train
    trainer.train_step(inputs, targets)
"""

__version__ = "0.1.0"
__author__ = "Messy Perceptron Network Team"

from .core import (
    Perceptron,
    Connection,
    ConnectionType,
    ConnectionManager,
    MessyGraphGenerator,
    create_messy_graph,
    MessyPerceptronNetwork,
)

from .training import (
    MessyPerceptronTrainer,
    ContinualLearner,
)

from .utils import (
    GraphAnalyzer,
    StabilityMonitor,
    initialize_weights_carefully,
    detect_and_fix_nans,
)

__all__ = [
    # Core
    'Perceptron',
    'Connection',
    'ConnectionType',
    'ConnectionManager',
    'MessyGraphGenerator',
    'create_messy_graph',
    'MessyPerceptronNetwork',
    # Training
    'MessyPerceptronTrainer',
    'ContinualLearner',
    # Utils
    'GraphAnalyzer',
    'StabilityMonitor',
    'initialize_weights_carefully',
    'detect_and_fix_nans',
]
