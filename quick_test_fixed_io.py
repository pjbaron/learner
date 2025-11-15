"""Quick test of baseline network with fixed I/O overlap."""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from messy_perceptron_network.core.fast_network import FastMessyPerceptronNetwork

def run_quick_test():
    """Run a quick 2-epoch test on Task 1 and Task 2."""
    print("="*70)
    print("QUICK TEST: Baseline Network with Fixed I/O")
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
    task1_indices = [i for i, (_, label) in enumerate(train_dataset) if label < 5]
    task1_dataset = Subset(train_dataset, task1_indices[:1000])  # Limit for speed
    task1_loader = DataLoader(task1_dataset, batch_size=32, shuffle=True)

    # Task 2: Digits 5-9
    task2_indices = [i for i, (_, label) in enumerate(train_dataset) if label >= 5]
    task2_dataset = Subset(train_dataset, task2_indices[:1000])  # Limit for speed
    task2_loader = DataLoader(task2_dataset, batch_size=32, shuffle=True)

    # Simple optimizer and loss
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    def train_epoch(loader, task_name):
        """Train for one epoch."""
        network.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.view(batch_x.size(0), -1)

            # Remap labels to 0-4 for both tasks
            if task_name == "Task 2":
                batch_y = batch_y - 5

            optimizer.zero_grad()
            outputs = network(batch_x)

            # Use only first 5 output neurons for classification
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
    for epoch in range(2):
        loss, acc = train_epoch(task1_loader, "Task 1")
        print(f"  Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.3f}")

    task1_acc_initial = evaluate(task1_loader, "Task 1")
    print(f"Task 1 final accuracy: {task1_acc_initial:.3f}")

    # Train Task 2
    print("\nTraining Task 2 (digits 5-9)...")
    for epoch in range(2):
        loss, acc = train_epoch(task2_loader, "Task 2")
        print(f"  Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.3f}")

    task2_acc = evaluate(task2_loader, "Task 2")
    task1_acc_after = evaluate(task1_loader, "Task 1")

    print(f"\nTask 2 final accuracy: {task2_acc:.3f}")
    print(f"Task 1 accuracy after Task 2: {task1_acc_after:.3f}")

    # Compute forgetting
    forgetting = (task1_acc_initial - task1_acc_after) / task1_acc_initial if task1_acc_initial > 0 else 1.0

    print("\n" + "="*70)
    print(f"CATASTROPHIC FORGETTING: {forgetting*100:.1f}%")
    print(f"Task 1: {task1_acc_initial:.3f} → {task1_acc_after:.3f}")
    print("="*70)

    return forgetting

if __name__ == '__main__':
    run_quick_test()
