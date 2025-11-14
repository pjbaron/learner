"""
Hyperparameter search for resource depletion parameters.

Search over:
- significance_threshold: How big gradient must be to count as "significant"
- consumption_amount: How much resource consumed per significant update
- recovery_rate: How fast resources recover

Goal: Find settings that reduce catastrophic forgetting.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from messy_perceptron_network.core.fast_network_modulated import FastMessyPerceptronNetwork
from messy_perceptron_network.training.probabilistic_trainer import ProbabilisticResourceTrainer


class MNISTClassifier(nn.Module):
    def __init__(self, network, n_classes=10):
        super().__init__()
        self.network = network
        self.input_projection = nn.Linear(784, network.n_input_perceptrons)
        self.output_layer = nn.Linear(network.n_output_perceptrons, n_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        projected = self.input_projection(x)
        network_output = self.network(projected)
        return self.output_layer(network_output)


def create_task_dataloaders(task_digits, batch_size=64, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    indices = [i for i, (_, label) in enumerate(dataset) if label in task_digits]
    task_dataset = Subset(dataset, indices)
    return DataLoader(task_dataset, batch_size=batch_size, shuffle=train)


def run_single_experiment(
    significance_threshold,
    consumption_amount,
    recovery_rate,
    n_epochs=2,  # Reduced for speed
    device='cpu',
    seed=42
):
    """Run single experiment and return key metrics."""

    # Create network
    network = FastMessyPerceptronNetwork(
        n_perceptrons=500,
        avg_degree=15,
        n_input_perceptrons=250,
        n_output_perceptrons=125,
        settling_iterations=7,
        seed=seed
    )

    classifier = MNISTClassifier(network, n_classes=10)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    trainer = ProbabilisticResourceTrainer(
        network=classifier,
        optimizer=optimizer,
        initial_resource=1.0,
        consumption_amount=consumption_amount,
        significance_threshold=significance_threshold,
        recovery_rate=recovery_rate,
        device=device
    )

    # Load data
    task_1_train = create_task_dataloaders([0, 1, 2, 3, 4], 64, train=True)
    task_1_test = create_task_dataloaders([0, 1, 2, 3, 4], 64, train=False)
    task_2_train = create_task_dataloaders([5, 6, 7, 8, 9], 64, train=True)
    task_2_test = create_task_dataloaders([5, 6, 7, 8, 9], 64, train=False)

    # Task 1
    for epoch in range(n_epochs):
        trainer.train_epoch(task_1_train, verbose=False)
    task_1_perf_initial = trainer.evaluate(task_1_test)

    # Get resource stats after Task 1
    resource_stats_task1 = trainer._get_resource_stats()

    # Task 2
    for epoch in range(n_epochs):
        trainer.train_epoch(task_2_train, verbose=False)
    task_1_perf_after_task2 = trainer.evaluate(task_1_test)
    task_2_perf = trainer.evaluate(task_2_test)

    # Get resource stats after Task 2
    resource_stats_task2 = trainer._get_resource_stats()

    # Calculate catastrophic forgetting
    forgetting = task_1_perf_initial['accuracy'] - task_1_perf_after_task2['accuracy']
    forgetting_pct = (forgetting / task_1_perf_initial['accuracy']) * 100

    return {
        'task_1_initial': task_1_perf_initial['accuracy'],
        'task_1_after_task_2': task_1_perf_after_task2['accuracy'],
        'task_2_final': task_2_perf['accuracy'],
        'catastrophic_forgetting': forgetting,
        'forgetting_pct': forgetting_pct,
        'resource_mean_task1': resource_stats_task1['mean'],
        'resource_min_task1': resource_stats_task1['min'],
        'resource_mean_task2': resource_stats_task2['mean'],
        'resource_min_task2': resource_stats_task2['min'],
    }


def hyperparameter_search():
    """Search over hyperparameter grid."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("\nHyperparameter Search for Resource Depletion")
    print("="*70)

    # Define search grid (from gradient analysis: mean ~0.0005, max ~0.005)
    # Reduced grid for faster results
    thresholds = [0.0001, 0.0005, 0.001]
    consumptions = [0.01, 0.05, 0.1]
    recovery_rates = [0.001, 0.005]

    results = []
    total_experiments = len(thresholds) * len(consumptions) * len(recovery_rates)
    experiment_num = 0

    print(f"\nRunning {total_experiments} experiments...")
    print(f"Grid: thresholds={len(thresholds)}, consumptions={len(consumptions)}, recovery={len(recovery_rates)}\n")

    for threshold in thresholds:
        for consumption in consumptions:
            for recovery in recovery_rates:
                experiment_num += 1

                print(f"[{experiment_num}/{total_experiments}] Testing: "
                      f"threshold={threshold:.4f}, consumption={consumption:.2f}, recovery={recovery:.3f}", flush=True)

                try:
                    metrics = run_single_experiment(
                        significance_threshold=threshold,
                        consumption_amount=consumption,
                        recovery_rate=recovery,
                        device=device
                    )

                    result = {
                        'threshold': threshold,
                        'consumption': consumption,
                        'recovery': recovery,
                        **metrics
                    }
                    results.append(result)

                    print(f"  → Forgetting: {metrics['forgetting_pct']:.1f}%, "
                          f"Task1: {metrics['task_1_initial']:.3f}→{metrics['task_1_after_task_2']:.3f}, "
                          f"Resources: {metrics['resource_mean_task2']:.3f} (min: {metrics['resource_min_task2']:.3f})", flush=True)

                except Exception as e:
                    print(f"  → ERROR: {e}")
                    continue

    # Sort by catastrophic forgetting (lower is better)
    results.sort(key=lambda x: x['forgetting_pct'])

    # Print top 10 results
    print("\n" + "="*70)
    print("TOP 10 RESULTS (by lowest catastrophic forgetting)")
    print("="*70)

    for i, r in enumerate(results[:10], 1):
        print(f"\n{i}. Forgetting: {r['forgetting_pct']:.2f}%")
        print(f"   threshold={r['threshold']:.4f}, consumption={r['consumption']:.2f}, recovery={r['recovery']:.3f}")
        print(f"   Task 1: {r['task_1_initial']:.3f} → {r['task_1_after_task_2']:.3f}")
        print(f"   Task 2: {r['task_2_final']:.3f}")
        print(f"   Resources after Task 2: mean={r['resource_mean_task2']:.3f}, min={r['resource_min_task2']:.3f}")

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hyperparameter_search_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nFull results saved to: {filename}")

    return results


if __name__ == "__main__":
    results = hyperparameter_search()
