#!/usr/bin/env python3
"""
Basic functionality test for the messy perceptron network.

Tests:
1. Network creation
2. Forward pass
3. Backward pass
4. Training step
5. Stability checks
"""

import torch
import torch.nn as nn
import sys
import os

# Add the project directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'messy_perceptron_network'))

from core import MessyPerceptronNetwork
from training import MessyPerceptronTrainer
from utils import StabilityMonitor, GraphAnalyzer, initialize_weights_carefully


def test_network_creation():
    """Test creating a small network."""
    print("\n" + "="*60)
    print("TEST 1: Network Creation")
    print("="*60)

    # Create a small network for testing
    network = MessyPerceptronNetwork(
        n_perceptrons=100,
        avg_degree=10,
        n_input_perceptrons=10,
        n_output_perceptrons=10,
        settling_iterations=5,
        seed=42
    )

    print("\nNetwork created successfully!")
    print(network)

    stats = network.get_statistics()
    print(f"\nNetwork statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    return network


def test_forward_pass(network):
    """Test forward pass."""
    print("\n" + "="*60)
    print("TEST 2: Forward Pass")
    print("="*60)

    # Create dummy input
    batch_size = 4
    n_inputs = network.n_input_perceptrons
    inputs = torch.randn(batch_size, n_inputs)

    print(f"\nInput shape: {inputs.shape}")

    # Forward pass
    outputs = network.forward(inputs)

    print(f"Output shape: {outputs.shape}")
    print(f"Output range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")

    assert outputs.shape == (batch_size, network.n_output_perceptrons), \
        f"Expected output shape {(batch_size, network.n_output_perceptrons)}, got {outputs.shape}"

    print("\n✓ Forward pass successful!")

    return outputs


def test_backward_pass(network, outputs):
    """Test backward pass."""
    print("\n" + "="*60)
    print("TEST 3: Backward Pass")
    print("="*60)

    # Create dummy targets
    targets = torch.randn_like(outputs)

    # Compute loss
    criterion = nn.MSELoss()
    loss = criterion(outputs, targets)

    print(f"\nLoss: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    # Check that gradients exist
    has_gradients = False
    for param in network.parameters():
        if param.grad is not None:
            has_gradients = True
            break

    assert has_gradients, "No gradients computed!"

    print("\n✓ Backward pass successful!")

    return loss


def test_training_step(network):
    """Test a complete training step."""
    print("\n" + "="*60)
    print("TEST 4: Training Step")
    print("="*60)

    # Create trainer
    trainer = MessyPerceptronTrainer(
        network,
        base_lr=0.01,
        optimizer_type='adam',
        cycles_per_batch=2,  # Use 2 cycles for faster testing
        device='cpu'
    )

    # Create dummy data
    batch_size = 4
    inputs = torch.randn(batch_size, network.n_input_perceptrons)
    targets = torch.randn(batch_size, network.n_output_perceptrons)

    # Training step
    stats = trainer.train_step(inputs, targets)

    print(f"\nTraining stats:")
    print(f"  Loss: {stats['loss']:.4f}")
    print(f"  Gradient norm: {stats['gradient_norm']:.4f}")
    print(f"  Cycle losses: {[f'{l:.4f}' for l in stats['cycle_losses']]}")

    print("\n✓ Training step successful!")

    return trainer


def test_stability_check(network, trainer):
    """Test stability monitoring."""
    print("\n" + "="*60)
    print("TEST 5: Stability Check")
    print("="*60)

    monitor = StabilityMonitor(network)

    # Do a few training steps
    for i in range(5):
        inputs = torch.randn(4, network.n_input_perceptrons)
        targets = torch.randn(4, network.n_output_perceptrons)
        trainer.train_step(inputs, targets)

    # Check stability
    monitor.print_status()

    issues = monitor.detect_issues()
    if issues:
        print(f"\n⚠️  Issues detected:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\n✓ No stability issues detected!")

    return monitor


def test_graph_analysis(network):
    """Test graph analysis."""
    print("\n" + "="*60)
    print("TEST 6: Graph Analysis")
    print("="*60)

    # Build edge dict from network
    edges = {
        'signal': [],
        'threshold_mod': [],
        'plasticity_mod': []
    }

    from core.connection import ConnectionType

    for conn in network.connection_manager.connections:
        edge = (conn.source.id, conn.target.id)
        if conn.connection_type == ConnectionType.SIGNAL:
            edges['signal'].append(edge)
        elif conn.connection_type == ConnectionType.THRESHOLD_MODULATION:
            edges['threshold_mod'].append(edge)
        elif conn.connection_type == ConnectionType.PLASTICITY_MODULATION:
            edges['plasticity_mod'].append(edge)

    analyzer = GraphAnalyzer(network.n_perceptrons, edges)

    print("\nComputing loop statistics...")
    loop_stats = analyzer.compute_loop_statistics(max_loop_length=20, sample_size=20)
    print(f"  Loops found: {loop_stats['num_loops']}")
    if loop_stats['num_loops'] > 0:
        print(f"  Loop length range: {loop_stats['min_length']} - {loop_stats['max_length']}")
        print(f"  Average loop length: {loop_stats['mean_length']:.2f}")

    print("\nComputing degree statistics...")
    degree_stats = analyzer.compute_degree_statistics()
    print(f"  Average in-degree: {degree_stats['in_degree']['mean']:.2f}")
    print(f"  Average out-degree: {degree_stats['out_degree']['mean']:.2f}")

    print("\nAnalyzing connectivity...")
    conn_stats = analyzer.analyze_connectivity()
    print(f"  Strongly connected: {conn_stats['is_strongly_connected']}")
    print(f"  Largest component: {conn_stats['largest_component_size']} nodes "
          f"({conn_stats['largest_component_fraction']*100:.1f}%)")

    print("\n✓ Graph analysis complete!")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("MESSY PERCEPTRON NETWORK - BASIC FUNCTIONALITY TEST")
    print("="*60)

    try:
        # Test 1: Create network
        network = test_network_creation()

        # Test 2: Forward pass
        outputs = test_forward_pass(network)

        # Test 3: Backward pass
        loss = test_backward_pass(network, outputs)

        # Test 4: Training step
        trainer = test_training_step(network)

        # Test 5: Stability check
        monitor = test_stability_check(network, trainer)

        # Test 6: Graph analysis
        test_graph_analysis(network)

        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60 + "\n")

        return True

    except Exception as e:
        print("\n" + "="*60)
        print("TEST FAILED! ✗")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
