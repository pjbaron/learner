"""Test gradient flow in FastMessyPerceptronNetwork."""
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from messy_perceptron_network.core.fast_network_with_resources import FastMessyPerceptronNetworkWithResources

# Create small network
print("Creating network...")
network = FastMessyPerceptronNetworkWithResources(
    n_perceptrons=100,
    avg_degree=10,
    n_input_perceptrons=20,
    n_output_perceptrons=20,
    settling_iterations=3,
    seed=42
)

# Simple classifier wrapper
class TestClassifier(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.input_proj = nn.Linear(50, network.n_input_perceptrons)
        self.output_layer = nn.Linear(network.n_output_perceptrons, 5)

    def forward(self, x):
        proj = self.input_proj(x)
        net_out = self.network(proj)
        return self.output_layer(net_out)

print("\nTesting gradient flow...")
classifier = TestClassifier(network)
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

# Single training step
inputs = torch.randn(8, 50)
labels = torch.randint(0, 5, (8,))
criterion = nn.CrossEntropyLoss()

optimizer.zero_grad()
outputs = classifier(inputs)
loss = criterion(outputs, labels)
loss.backward()

# Check gradients
print('\nGradient flow check:')
print(f'  Output layer: {classifier.output_layer.weight.grad.abs().mean().item():.6f}')
print(f'  Input proj: {classifier.input_proj.weight.grad.abs().mean().item():.6f}')
if network.signal_weights.grad is not None:
    print(f'  Network signal weights: {network.signal_weights.grad.abs().mean().item():.6f}')
else:
    print(f'  Network signal weights: None')
if network.thresholds.grad is not None:
    print(f'  Network thresholds: {network.thresholds.grad.abs().mean().item():.6f}')
else:
    print(f'  Network thresholds: None')
