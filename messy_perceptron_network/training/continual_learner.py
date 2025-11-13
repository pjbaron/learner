"""
Continual learning harness for sequential task training.

Manages:
- Sequential task presentation (e.g., MNIST digits 0-4, then 5-9, then back to 0-4)
- Task-specific evaluation
- Catastrophic forgetting metrics
- Task switching dynamics
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional


class ContinualLearner:
    """
    Manages continual learning across sequential tasks.

    Tracks performance on all tasks over time to measure:
    - Catastrophic forgetting
    - Forward transfer
    - Backward transfer
    """

    def __init__(self, trainer, task_dataloaders: Dict[str, Tuple]):
        """
        Initialize continual learner.

        Args:
            trainer: MessyPerceptronTrainer instance
            task_dataloaders: Dictionary mapping task names to (train_loader, test_loader) tuples
        """
        self.trainer = trainer
        self.task_dataloaders = task_dataloaders
        self.task_names = list(task_dataloaders.keys())

        # History of performance on each task over time
        self.performance_history = {
            task_name: [] for task_name in self.task_names
        }

        # Current task being trained
        self.current_task = None
        self.training_step = 0

    def train_task(self, task_name, n_epochs, verbose=True):
        """
        Train on a specific task for n_epochs.

        Args:
            task_name: Name of the task to train on
            n_epochs: Number of epochs to train
            verbose: Print progress

        Returns:
            Dictionary with training statistics
        """
        if task_name not in self.task_dataloaders:
            raise ValueError(f"Unknown task: {task_name}")

        self.current_task = task_name
        train_loader, test_loader = self.task_dataloaders[task_name]

        if verbose:
            print(f"\n{'='*60}")
            print(f"Training on task: {task_name}")
            print(f"{'='*60}")

        task_stats = {
            'task_name': task_name,
            'epoch_losses': [],
            'epoch_accuracies': [],
        }

        for epoch in range(n_epochs):
            if verbose:
                print(f"\nEpoch {epoch+1}/{n_epochs}")

            # Train for one epoch
            epoch_stats = self.trainer.train_epoch(train_loader, verbose=verbose)
            task_stats['epoch_losses'].append(epoch_stats['avg_loss'])

            # Evaluate on this task
            eval_stats = self.trainer.evaluate(test_loader)

            if verbose:
                print(f"  Train Loss: {epoch_stats['avg_loss']:.4f}")
                print(f"  Test Loss: {eval_stats['loss']:.4f}")

            # Evaluate on all tasks to track forgetting
            if (epoch + 1) % max(1, n_epochs // 5) == 0 or epoch == n_epochs - 1:
                all_task_performance = self.evaluate_all_tasks()
                self._record_performance(all_task_performance)

                if verbose:
                    print(f"\n  Performance on all tasks:")
                    for task, perf in all_task_performance.items():
                        print(f"    {task}: Loss={perf['loss']:.4f}")

            self.training_step += 1

        return task_stats

    def train_sequence(self, task_sequence: List[str], epochs_per_task: int, verbose=True):
        """
        Train on a sequence of tasks.

        Args:
            task_sequence: List of task names in order
            epochs_per_task: Number of epochs to train on each task
            verbose: Print progress

        Returns:
            Dictionary with continual learning statistics
        """
        sequence_stats = {
            'task_sequence': task_sequence,
            'epochs_per_task': epochs_per_task,
            'task_stats': [],
        }

        for task_name in task_sequence:
            task_stats = self.train_task(task_name, epochs_per_task, verbose=verbose)
            sequence_stats['task_stats'].append(task_stats)

        # Compute continual learning metrics
        cl_metrics = self.compute_continual_learning_metrics()
        sequence_stats['continual_learning_metrics'] = cl_metrics

        if verbose:
            print(f"\n{'='*60}")
            print("Continual Learning Metrics:")
            print(f"{'='*60}")
            for metric_name, value in cl_metrics.items():
                print(f"{metric_name}: {value:.4f}")

        return sequence_stats

    def evaluate_all_tasks(self):
        """
        Evaluate on all tasks.

        Returns:
            Dictionary mapping task names to evaluation statistics
        """
        all_task_performance = {}

        for task_name in self.task_names:
            _, test_loader = self.task_dataloaders[task_name]
            eval_stats = self.trainer.evaluate(test_loader)
            all_task_performance[task_name] = eval_stats

        return all_task_performance

    def _record_performance(self, task_performance: Dict):
        """Record performance on all tasks at current training step."""
        for task_name, stats in task_performance.items():
            self.performance_history[task_name].append({
                'training_step': self.training_step,
                'current_task': self.current_task,
                'loss': stats['loss'],
            })

    def compute_continual_learning_metrics(self):
        """
        Compute continual learning metrics.

        Metrics:
        - Average performance across all tasks at end
        - Catastrophic forgetting (performance drop on earlier tasks)
        - Forward transfer (initial performance on new tasks)
        - Backward transfer (performance recovery on earlier tasks)

        Returns:
            Dictionary with metrics
        """
        metrics = {}

        # Get final performance on each task
        final_performance = {}
        for task_name, history in self.performance_history.items():
            if len(history) > 0:
                final_performance[task_name] = history[-1]['loss']

        # Average final performance
        if len(final_performance) > 0:
            metrics['avg_final_loss'] = np.mean(list(final_performance.values()))

        # Catastrophic forgetting: compare first task performance
        # when it was trained vs at the end
        if len(self.task_names) > 0:
            first_task = self.task_names[0]
            history = self.performance_history[first_task]

            if len(history) > 1:
                # Find performance right after training on first task
                initial_perf = history[0]['loss']
                # Performance at end
                final_perf = history[-1]['loss']

                metrics['forgetting'] = final_perf - initial_perf
                metrics['forgetting_percentage'] = 100 * (final_perf - initial_perf) / (initial_perf + 1e-8)

        return metrics

    def get_performance_history(self):
        """
        Get complete performance history.

        Returns:
            Dictionary mapping task names to performance over time
        """
        return self.performance_history

    def plot_performance_over_time(self, save_path=None):
        """
        Plot performance on all tasks over time.

        Args:
            save_path: Optional path to save plot
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        for task_name, history in self.performance_history.items():
            if len(history) > 0:
                steps = [h['training_step'] for h in history]
                losses = [h['loss'] for h in history]
                ax.plot(steps, losses, marker='o', label=task_name)

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('Performance on All Tasks Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

        plt.close()
