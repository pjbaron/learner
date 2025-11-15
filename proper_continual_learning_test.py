"""Proper continual learning test with train/test splits and learning curves."""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from messy_perceptron_network.core.fast_network import FastMessyPerceptronNetwork
import numpy as np

def create_task_splits(dataset, task_labels, train_size=2000, test_size=500):
    """Create train/test splits for a task."""
    indices = [i for i, (_, label) in enumerate(dataset) if label in task_labels]

    # Shuffle and split
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:train_size+test_size]

    return train_indices, test_indices


def run_proper_test(max_epochs=20, early_stop_patience=3):
    """Run proper continual learning test with train/test splits."""
    print("="*70)
    print("PROPER CONTINUAL LEARNING TEST")
    print("Train/Test Splits, Learning Curve Tracking")
    print("="*70)

    # Create network
    network = FastMessyPerceptronNetwork(
        n_perceptrons=500,
        avg_degree=15,
        n_input_perceptrons=250,
        n_output_perceptrons=125,
        settling_iterations=7,
        seed=42
    )

    # Load MNIST
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

    # Task 1: Digits 0-4
    task1_train_idx, task1_test_idx = create_task_splits(
        train_dataset,
        task_labels=[0, 1, 2, 3, 4],
        train_size=2000,
        test_size=500
    )
    task1_train_loader = DataLoader(Subset(train_dataset, task1_train_idx), batch_size=32, shuffle=True)
    task1_test_loader = DataLoader(Subset(train_dataset, task1_test_idx), batch_size=32, shuffle=False)

    # Task 2: Digits 5-9
    task2_train_idx, task2_test_idx = create_task_splits(
        train_dataset,
        task_labels=[5, 6, 7, 8, 9],
        train_size=2000,
        test_size=500
    )
    task2_train_loader = DataLoader(Subset(train_dataset, task2_train_idx), batch_size=32, shuffle=True)
    task2_test_loader = DataLoader(Subset(train_dataset, task2_test_idx), batch_size=32, shuffle=False)

    # Optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()

    def train_epoch(loader, task_name):
        """Train for one epoch."""
        network.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.view(batch_x.size(0), -1)

            # Remap labels to 0-4
            if task_name == "Task 2":
                batch_y = batch_y - 5

            optimizer.zero_grad()
            outputs = network(batch_x)
            logits = outputs[:, :5]
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)

        return total_loss / len(loader), correct / total

    def evaluate(loader, task_name):
        """Evaluate on test set."""
        network.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.view(batch_x.size(0), -1)

                if task_name == "Task 2":
                    batch_y = batch_y - 5

                outputs = network(batch_x)
                logits = outputs[:, :5]
                pred = logits.argmax(dim=1)
                correct += (pred == batch_y).sum().item()
                total += batch_y.size(0)

        return correct / total

    # Train Task 1 with learning curve tracking
    print("\n" + "="*70)
    print("TASK 1 TRAINING (digits 0-4)")
    print("="*70)

    task1_train_accs = []
    task1_test_accs = []
    best_test_acc = 0
    patience_counter = 0

    for epoch in range(max_epochs):
        train_loss, train_acc = train_epoch(task1_train_loader, "Task 1")
        test_acc = evaluate(task1_test_loader, "Task 1")

        task1_train_accs.append(train_acc)
        task1_test_accs.append(test_acc)

        print(f"Epoch {epoch+1:2d}: Loss={train_loss:.4f}, "
              f"Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}")

        # Early stopping check
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience and test_acc >= 0.85:
            print(f"Early stopping: test accuracy plateaued at {test_acc:.3f}")
            break

    task1_final_test_acc = task1_test_accs[-1]
    print(f"\nTask 1 Final Test Accuracy: {task1_final_test_acc:.3f}")

    # Predict epochs to 90% if not reached
    if task1_final_test_acc < 0.90:
        # Simple linear extrapolation from last 5 epochs
        if len(task1_test_accs) >= 5:
            recent_accs = task1_test_accs[-5:]
            improvement_rate = (recent_accs[-1] - recent_accs[0]) / 5
            if improvement_rate > 0.001:
                epochs_to_90 = int((0.90 - task1_final_test_acc) / improvement_rate)
                print(f"Prediction: ~{epochs_to_90} more epochs to reach 90% test accuracy")
                print(f"  (based on recent improvement rate: {improvement_rate*100:.2f}% per epoch)")
            else:
                print("Learning appears to have plateaued - may not reach 90%")
        else:
            print("Not enough data to predict convergence")
    else:
        print(f"✓ Reached 90%+ test accuracy!")

    # Train Task 2 with learning curve tracking
    print("\n" + "="*70)
    print("TASK 2 TRAINING (digits 5-9)")
    print("="*70)

    task2_train_accs = []
    task2_test_accs = []
    best_test_acc = 0
    patience_counter = 0

    for epoch in range(max_epochs):
        train_loss, train_acc = train_epoch(task2_train_loader, "Task 2")
        test_acc = evaluate(task2_test_loader, "Task 2")

        task2_train_accs.append(train_acc)
        task2_test_accs.append(test_acc)

        print(f"Epoch {epoch+1:2d}: Loss={train_loss:.4f}, "
              f"Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}")

        # Early stopping check
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience and test_acc >= 0.85:
            print(f"Early stopping: test accuracy plateaued at {test_acc:.3f}")
            break

    task2_final_test_acc = task2_test_accs[-1]
    print(f"\nTask 2 Final Test Accuracy: {task2_final_test_acc:.3f}")

    # Measure catastrophic forgetting on TEST set
    task1_test_acc_after = evaluate(task1_test_loader, "Task 1")

    print("\n" + "="*70)
    print("CATASTROPHIC FORGETTING ANALYSIS (on TEST set)")
    print("="*70)
    print(f"Task 1 test accuracy: {task1_final_test_acc:.3f} → {task1_test_acc_after:.3f}")
    print(f"Task 2 test accuracy: {task2_final_test_acc:.3f}")

    forgetting_pct = ((task1_final_test_acc - task1_test_acc_after) / task1_final_test_acc * 100) if task1_final_test_acc > 0 else 100.0
    retention_pct = (task1_test_acc_after / task1_final_test_acc * 100) if task1_final_test_acc > 0 else 0

    print(f"\nForgetting: {forgetting_pct:.1f}%")
    print(f"Retention: {retention_pct:.1f}%")

    print("\n" + "="*70)
    print("LEARNING CURVE SUMMARY")
    print("="*70)
    print(f"Task 1 epochs trained: {len(task1_test_accs)}")
    print(f"Task 1 train/test gap: {task1_train_accs[-1] - task1_test_accs[-1]:.3f}")
    print(f"Task 2 epochs trained: {len(task2_test_accs)}")
    print(f"Task 2 train/test gap: {task2_train_accs[-1] - task2_test_accs[-1]:.3f}")

    if forgetting_pct < 10:
        print("\n✓✓ Excellent! Catastrophic forgetting < 10%")
    elif forgetting_pct < 50:
        print("\n✓ Good progress - forgetting < 50%")
    elif forgetting_pct < 90:
        print("\n⚠ Moderate forgetting - needs consolidation")
    else:
        print("\n✗ Severe catastrophic forgetting - major issue")

    return {
        'task1_test_accs': task1_test_accs,
        'task2_test_accs': task2_test_accs,
        'task1_final': task1_final_test_acc,
        'task1_after_task2': task1_test_acc_after,
        'task2_final': task2_final_test_acc,
        'forgetting_pct': forgetting_pct
    }


if __name__ == '__main__':
    results = run_proper_test(max_epochs=20, early_stop_patience=3)
