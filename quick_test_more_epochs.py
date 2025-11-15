"""Quick test with more epochs to verify learning capability."""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from messy_perceptron_network.core.fast_network import FastMessyPerceptronNetwork

def run_extended_test():
    """Run 5-epoch test on Task 1 and Task 2."""
    print("="*70)
    print("EXTENDED TEST: 5 Epochs per Task")
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

    # Task 1: Digits 0-4 (more samples)
    task1_indices = [i for i, (_, label) in enumerate(train_dataset) if label < 5]
    task1_dataset = Subset(train_dataset, task1_indices[:2000])
    task1_loader = DataLoader(task1_dataset, batch_size=32, shuffle=True)

    # Task 2: Digits 5-9
    task2_indices = [i for i, (_, label) in enumerate(train_dataset) if label >= 5]
    task2_dataset = Subset(train_dataset, task2_indices[:2000])
    task2_loader = DataLoader(task2_dataset, batch_size=32, shuffle=True)

    # Optimizer with slightly higher LR
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
        """Evaluate on a task."""
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

    # Train Task 1
    print("\nTraining Task 1 (digits 0-4)...")
    for epoch in range(5):
        loss, acc = train_epoch(task1_loader, "Task 1")
        print(f"  Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.3f}")

    task1_acc_initial = evaluate(task1_loader, "Task 1")
    print(f"Task 1 final accuracy: {task1_acc_initial:.3f}")

    # Train Task 2
    print("\nTraining Task 2 (digits 5-9)...")
    for epoch in range(5):
        loss, acc = train_epoch(task2_loader, "Task 2")
        print(f"  Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.3f}")

    task2_acc = evaluate(task2_loader, "Task 2")
    task1_acc_after = evaluate(task1_loader, "Task 1")

    print(f"\nTask 2 final accuracy: {task2_acc:.3f}")
    print(f"Task 1 accuracy after Task 2: {task1_acc_after:.3f}")

    # Compute forgetting
    forgetting_pct = ((task1_acc_initial - task1_acc_after) / task1_acc_initial * 100) if task1_acc_initial > 0 else 100.0

    print("\n" + "="*70)
    print(f"CATASTROPHIC FORGETTING: {forgetting_pct:.1f}%")
    print(f"Task 1: {task1_acc_initial:.3f} → {task1_acc_after:.3f}")
    print(f"Task 2: {task2_acc:.3f}")
    print("="*70)

    if forgetting_pct < 50:
        print("✓ Forgetting < 50% - Architecture fix appears successful!")
    elif forgetting_pct < 90:
        print("⚠ Moderate forgetting - Some improvement but not solved")
    else:
        print("✗ High forgetting persists - Problem not resolved")

if __name__ == '__main__':
    run_extended_test()
