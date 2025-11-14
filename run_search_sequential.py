"""Run hyperparameter search sequentially with manual iteration control."""

import sys
sys.path.insert(0, '.')
from hyperparameter_search import run_single_experiment
import json
from datetime import datetime

# Define grid
thresholds = [0.0001, 0.0005, 0.001]
consumptions = [0.01, 0.05, 0.1]
recovery_rates = [0.001, 0.005]

results = []
total = len(thresholds) * len(consumptions) * len(recovery_rates)
count = 0

print(f"Running {total} experiments sequentially...\n")

for threshold in thresholds:
    for consumption in consumptions:
        for recovery in recovery_rates:
            count += 1
            print(f"[{count}/{total}] threshold={threshold:.4f}, consumption={consumption:.2f}, recovery={recovery:.3f}")

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
                      f"Resources: {metrics['resource_mean_task2']:.3f}\n")

            except Exception as e:
                print(f"  → ERROR: {e}\n")

# Sort and display top results
results.sort(key=lambda x: x['forgetting_pct'])

print("\n" + "="*70)
print("TOP 5 RESULTS")
print("="*70)
for i, r in enumerate(results[:5], 1):
    print(f"\n{i}. Forgetting: {r['forgetting_pct']:.2f}%")
    print(f"   threshold={r['threshold']:.4f}, consumption={r['consumption']:.2f}, recovery={r['recovery']:.3f}")
    print(f"   Resources: {r['resource_mean_task2']:.3f} (min: {r['resource_min_task2']:.3f})")

# Save results
filename = f"hyperparameter_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(filename, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {filename}")
