"""
Stability utilities for training the messy perceptron network.

Tools for:
- Gradient monitoring
- NaN/Inf detection
- Activation monitoring
- Weight initialization
"""

import torch
import numpy as np
from typing import List, Dict


class StabilityMonitor:
    """
    Monitor training stability and detect issues.

    Tracks:
    - Gradient magnitudes
    - NaN/Inf values
    - Activation statistics
    - Weight statistics
    """

    def __init__(self, network):
        """
        Initialize stability monitor.

        Args:
            network: MessyPerceptronNetwork instance
        """
        self.network = network
        self.history = {
            'gradient_norms': [],
            'gradient_max': [],
            'activation_stats': [],
            'weight_stats': [],
            'nan_detected': [],
        }

    def check_gradients(self):
        """
        Check gradient magnitudes for all parameters.

        Returns:
            Dictionary with gradient statistics
        """
        total_norm = 0.0
        max_grad = 0.0
        nan_detected = False
        inf_detected = False

        for param in self.network.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                max_grad = max(max_grad, param.grad.data.abs().max().item())

                if torch.isnan(param.grad.data).any():
                    nan_detected = True
                if torch.isinf(param.grad.data).any():
                    inf_detected = True

        total_norm = total_norm ** 0.5

        stats = {
            'total_norm': total_norm,
            'max_gradient': max_grad,
            'nan_detected': nan_detected,
            'inf_detected': inf_detected,
        }

        self.history['gradient_norms'].append(total_norm)
        self.history['gradient_max'].append(max_grad)
        self.history['nan_detected'].append(nan_detected or inf_detected)

        return stats

    def check_activations(self):
        """
        Check activation statistics.

        Returns:
            Dictionary with activation statistics
        """
        activations = self.network.get_activations().detach().cpu().numpy()

        stats = {
            'mean': np.mean(activations),
            'std': np.std(activations),
            'min': np.min(activations),
            'max': np.max(activations),
            'sparsity': np.mean(np.abs(activations) < 0.1),
            'dead_neurons': np.sum(np.abs(activations) < 0.001),
            'saturated_neurons': np.sum(np.abs(activations) > 0.99),
        }

        self.history['activation_stats'].append(stats)

        return stats

    def check_weights(self):
        """
        Check weight statistics.

        Returns:
            Dictionary with weight statistics
        """
        all_weights = []
        for param in self.network.parameters():
            all_weights.append(param.data.detach().cpu().numpy().flatten())

        all_weights = np.concatenate(all_weights)

        stats = {
            'mean': np.mean(all_weights),
            'std': np.std(all_weights),
            'min': np.min(all_weights),
            'max': np.max(all_weights),
        }

        self.history['weight_stats'].append(stats)

        return stats

    def check_plasticity(self):
        """
        Check plasticity rate statistics.

        Returns:
            Dictionary with plasticity statistics
        """
        plasticity = self.network.get_plasticity_rates().detach().cpu().numpy()

        stats = {
            'mean': np.mean(plasticity),
            'std': np.std(plasticity),
            'min': np.min(plasticity),
            'max': np.max(plasticity),
        }

        return stats

    def full_check(self):
        """
        Perform full stability check.

        Returns:
            Dictionary with all statistics
        """
        return {
            'gradients': self.check_gradients(),
            'activations': self.check_activations(),
            'weights': self.check_weights(),
            'plasticity': self.check_plasticity(),
        }

    def detect_issues(self):
        """
        Detect stability issues.

        Returns:
            List of detected issues with descriptions
        """
        issues = []

        # Check gradients
        grad_stats = self.check_gradients()
        if grad_stats['nan_detected']:
            issues.append("NaN or Inf detected in gradients!")
        if grad_stats['total_norm'] > 100:
            issues.append(f"Very large gradient norm: {grad_stats['total_norm']:.2f}")
        if grad_stats['total_norm'] < 1e-6:
            issues.append(f"Very small gradient norm: {grad_stats['total_norm']:.2e} (possible vanishing gradients)")

        # Check activations
        act_stats = self.check_activations()
        if act_stats['dead_neurons'] > self.network.n_perceptrons * 0.5:
            issues.append(f"Many dead neurons: {act_stats['dead_neurons']}/{self.network.n_perceptrons}")
        if act_stats['saturated_neurons'] > self.network.n_perceptrons * 0.5:
            issues.append(f"Many saturated neurons: {act_stats['saturated_neurons']}/{self.network.n_perceptrons}")

        return issues

    def print_status(self):
        """Print current stability status."""
        stats = self.full_check()

        print("\n" + "="*60)
        print("Stability Status")
        print("="*60)

        print(f"\nGradients:")
        print(f"  Total norm: {stats['gradients']['total_norm']:.4f}")
        print(f"  Max gradient: {stats['gradients']['max_gradient']:.4f}")
        print(f"  NaN/Inf: {stats['gradients']['nan_detected']}")

        print(f"\nActivations:")
        print(f"  Mean: {stats['activations']['mean']:.4f}")
        print(f"  Std: {stats['activations']['std']:.4f}")
        print(f"  Range: [{stats['activations']['min']:.4f}, {stats['activations']['max']:.4f}]")
        print(f"  Sparsity: {stats['activations']['sparsity']:.2%}")
        print(f"  Dead neurons: {stats['activations']['dead_neurons']}")

        print(f"\nWeights:")
        print(f"  Mean: {stats['weights']['mean']:.4f}")
        print(f"  Std: {stats['weights']['std']:.4f}")
        print(f"  Range: [{stats['weights']['min']:.4f}, {stats['weights']['max']:.4f}]")

        print(f"\nPlasticity:")
        print(f"  Mean: {stats['plasticity']['mean']:.4f}")
        print(f"  Std: {stats['plasticity']['std']:.4f}")
        print(f"  Range: [{stats['plasticity']['min']:.4f}, {stats['plasticity']['max']:.4f}]")

        # Check for issues
        issues = self.detect_issues()
        if issues:
            print(f"\n⚠️  WARNINGS:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"\n✓ No stability issues detected")

        print("="*60 + "\n")


def initialize_weights_carefully(network):
    """
    Carefully initialize network weights for stability.

    Args:
        network: MessyPerceptronNetwork instance
    """
    for perceptron in network.perceptrons:
        # Initialize threshold to small random value
        torch.nn.init.normal_(perceptron.theta, mean=0.0, std=0.1)

    # Connection weights are already initialized in Connection class
    # but we can verify they're reasonable
    for conn in network.connection_manager.connections:
        if torch.abs(conn.weight).item() > 1.0:
            # Clip very large weights
            with torch.no_grad():
                conn.weight.clamp_(-1.0, 1.0)

    print("Weights initialized carefully for stability.")


def detect_and_fix_nans(network):
    """
    Detect and fix NaN values in network parameters.

    Args:
        network: MessyPerceptronNetwork instance

    Returns:
        True if NaNs were found and fixed, False otherwise
    """
    nans_found = False

    for name, param in network.named_parameters():
        if torch.isnan(param.data).any():
            print(f"⚠️  NaN detected in {name}, resetting to small random values")
            torch.nn.init.normal_(param.data, mean=0.0, std=0.01)
            nans_found = True

        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"⚠️  NaN detected in {name} gradient, zeroing gradient")
            param.grad.zero_()
            nans_found = True

    return nans_found
