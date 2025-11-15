"""
Estimate required settling iterations for recurrent network.

Simple heuristic: longest path ≈ max(input_distance) + max(backward_loop_jump)
"""

import torch
import numpy as np
from messy_perceptron_network.core.fast_network import FastMessyPerceptronNetwork

print("Creating deep messy perceptron network...")
network = FastMessyPerceptronNetwork(
    n_perceptrons=2000,
    avg_degree=30,
    n_input_perceptrons=250,
    n_output_perceptrons=125,
    settling_iterations=30,
    seed=42
)

print("\n" + "="*60)
print("SETTLING ITERATIONS ESTIMATE")
print("="*60)

# Get neuron distances
neuron_distances = network.neuron_distances
input_indices = network.input_perceptron_indices.numpy()
output_indices = network.output_perceptron_indices.numpy()

# Analyze input distances
input_dists = neuron_distances[input_indices]
max_input_dist = input_dists.max()
avg_input_dist = input_dists.mean()

print(f"\nInput neuron distances to outputs:")
print(f"  Average: {avg_input_dist:.1f} hops")
print(f"  Maximum: {max_input_dist:.0f} hops")

# Analyze backward loop magnitudes
signal_indices = network.signal_indices.numpy()
backward_jumps = []

for src, dst in zip(signal_indices[0], signal_indices[1]):
    src_dist = neuron_distances[src]
    dst_dist = neuron_distances[dst]

    # Backward connection = increasing distance
    if src_dist < dst_dist:
        jump = dst_dist - src_dist
        backward_jumps.append(jump)

if backward_jumps:
    max_backward_jump = max(backward_jumps)
    avg_backward_jump = np.mean(backward_jumps)
    print(f"\nBackward recurrent loops:")
    print(f"  Count: {len(backward_jumps)}")
    print(f"  Average jump: {avg_backward_jump:.1f} hops")
    print(f"  Maximum jump: {max_backward_jump:.0f} hops")

    # Estimate longest path
    # Worst case: start at furthest input, hit max backward loop, then traverse back
    estimated_longest = max_input_dist + max_backward_jump
    print(f"\nEstimated longest path:")
    print(f"  Max input distance + max backward jump = {max_input_dist:.0f} + {max_backward_jump:.0f}")
    print(f"  = {estimated_longest:.0f} hops")

    print(f"\n{'='*60}")
    print(f"RECOMMENDATION")
    print(f"{'='*60}")
    print(f"Current settling iterations: {network.settling_iterations}")
    print(f"Estimated longest path: {estimated_longest:.0f} hops")

    recommended = int(np.ceil(estimated_longest * 1.1))  # 10% buffer
    if network.settling_iterations >= estimated_longest:
        print(f"✓ ADEQUATE: Current iterations likely sufficient")
        if network.settling_iterations < recommended:
            print(f"  Suggest: {recommended} iterations for safety margin")
    else:
        print(f"✗ INSUFFICIENT: Increase to at least {int(estimated_longest)} iterations")
        print(f"  Recommended: {recommended} iterations (with 10% buffer)")
else:
    print("\nNo backward loops found!")
