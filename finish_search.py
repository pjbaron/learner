"""Complete the final 3 experiments and compile all results."""
import sys
sys.path.insert(0, '.')
from hyperparameter_search import run_single_experiment
import json

# Manual results from experiments 1-15 (from previous run)
results = [
    {'threshold': 0.0001, 'consumption': 0.01, 'recovery': 0.001, 'forgetting_pct': 99.7, 'task_1_initial': 0.984, 'task_1_after_task_2': 0.003, 'resource_mean_task2': 0.679},
    {'threshold': 0.0001, 'consumption': 0.01, 'recovery': 0.005, 'forgetting_pct': 100.0, 'task_1_initial': 0.978, 'task_1_after_task_2': 0.000, 'resource_mean_task2': 0.972},
    {'threshold': 0.0001, 'consumption': 0.05, 'recovery': 0.001, 'forgetting_pct': 100.0, 'task_1_initial': 0.980, 'task_1_after_task_2': 0.000, 'resource_mean_task2': 0.585},
    {'threshold': 0.0001, 'consumption': 0.05, 'recovery': 0.005, 'forgetting_pct': 100.0, 'task_1_initial': 0.981, 'task_1_after_task_2': 0.000, 'resource_mean_task2': 0.699},
    {'threshold': 0.0001, 'consumption': 0.10, 'recovery': 0.001, 'forgetting_pct': 99.6, 'task_1_initial': 0.976, 'task_1_after_task_2': 0.004, 'resource_mean_task2': 0.552},
    {'threshold': 0.0001, 'consumption': 0.10, 'recovery': 0.005, 'forgetting_pct': 99.3, 'task_1_initial': 0.980, 'task_1_after_task_2': 0.007, 'resource_mean_task2': 0.634},
    {'threshold': 0.0005, 'consumption': 0.01, 'recovery': 0.001, 'forgetting_pct': 99.9, 'task_1_initial': 0.982, 'task_1_after_task_2': 0.001, 'resource_mean_task2': 0.771},
    {'threshold': 0.0005, 'consumption': 0.01, 'recovery': 0.005, 'forgetting_pct': 100.0, 'task_1_initial': 0.981, 'task_1_after_task_2': 0.000, 'resource_mean_task2': 0.996},
    {'threshold': 0.0005, 'consumption': 0.05, 'recovery': 0.001, 'forgetting_pct': 100.0, 'task_1_initial': 0.980, 'task_1_after_task_2': 0.000, 'resource_mean_task2': 0.613},
    {'threshold': 0.0005, 'consumption': 0.05, 'recovery': 0.005, 'forgetting_pct': 100.0, 'task_1_initial': 0.983, 'task_1_after_task_2': 0.000, 'resource_mean_task2': 0.789},
    {'threshold': 0.0005, 'consumption': 0.10, 'recovery': 0.001, 'forgetting_pct': 99.8, 'task_1_initial': 0.980, 'task_1_after_task_2': 0.002, 'resource_mean_task2': 0.583},
    {'threshold': 0.0005, 'consumption': 0.10, 'recovery': 0.005, 'forgetting_pct': 99.9, 'task_1_initial': 0.980, 'task_1_after_task_2': 0.001, 'resource_mean_task2': 0.687},
    {'threshold': 0.0010, 'consumption': 0.01, 'recovery': 0.001, 'forgetting_pct': 100.0, 'task_1_initial': 0.983, 'task_1_after_task_2': 0.000, 'resource_mean_task2': 0.850},
    {'threshold': 0.0010, 'consumption': 0.01, 'recovery': 0.005, 'forgetting_pct': 100.0, 'task_1_initial': 0.980, 'task_1_after_task_2': 0.000, 'resource_mean_task2': 0.998},
]

# Run remaining 3 experiments (16, 17, 18)
remaining = [
    (0.0010, 0.05, 0.001),  # 16
    (0.0010, 0.05, 0.005),  # 17
    (0.0010, 0.10, 0.001),  # 18
]

print("Running final 3 experiments...")
for i, (threshold, consumption, recovery) in enumerate(remaining, 16):
    print(f"\n[{i}/18] threshold={threshold:.4f}, consumption={consumption:.2f}, recovery={recovery:.3f}")

    try:
        metrics = run_single_experiment(
            significance_threshold=threshold,
            consumption_amount=consumption,
            recovery_rate=recovery,
            n_epochs=2,
            device='cpu',
            seed=42
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
              f"Resources: {metrics['resource_mean_task2']:.3f}")
    except Exception as e:
        print(f"  → ERROR: {e}")

# Sort and display results
results.sort(key=lambda x: x.get('forgetting_pct', 100))

print("\n" + "="*70)
print("COMPLETE RESULTS - TOP 10")
print("="*70)
for i, r in enumerate(results[:10], 1):
    print(f"\n{i}. Forgetting: {r.get('forgetting_pct', 'N/A'):.2f}%")
    print(f"   threshold={r['threshold']:.4f}, consumption={r['consumption']:.2f}, recovery={r['recovery']:.3f}")
    print(f"   Resources: {r.get('resource_mean_task2', 'N/A'):.3f}")

# Save results
filename = "hyperparameter_results_complete.json"
with open(filename, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n\nResults saved to: {filename}")
