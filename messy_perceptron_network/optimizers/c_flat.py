"""
C-Flat Optimizer for Continual Learning (NeurIPS 2024)

Implementation of "Make Continual Learning Stronger via C-Flat"
Paper: https://arxiv.org/abs/2404.00986
GitHub: https://github.com/WanNaa/C-Flat

C-Flat promotes flatter loss landscapes tailored for continual learning,
mitigating catastrophic forgetting through sharpness-aware optimization.

Key features:
- Zeroth-order sharpness: Measures max loss change in neighborhood
- First-order flatness: Constrains gradient norm in neighborhood
- Plug-and-play: Works with any base optimizer (Adam, SGD, etc.)
- One-line integration: Minimal code changes required

Hyperparameters (from paper):
- rho (ρ): 0.2 - Perturbation radius for sharpness computation
- lambda (λ): 0.2 - Weight for first-order flatness term
"""

import torch
import torch.nn as nn
from typing import Callable, Optional


class CFlatOptimizer:
    """
    C-Flat: Continual Flatness Optimizer

    Wraps a base optimizer to perform sharpness-aware optimization
    for continual learning. Seeks flat minima that generalize better
    across sequential tasks.

    Usage:
        base_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer = CFlatOptimizer(
            params=model.parameters(),
            base_optimizer=base_optimizer,
            model=model,
            rho=0.2,
            lambda_=0.2
        )

        # In training loop:
        def loss_fn():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            return outputs, loss

        optimizer.set_closure(loss_fn)
        outputs, loss = optimizer.step()
    """

    def __init__(self,
                 params,
                 base_optimizer: torch.optim.Optimizer,
                 model: nn.Module,
                 rho: float = 0.2,
                 lambda_: float = 0.2,
                 adaptive: bool = False):
        """
        Initialize C-Flat optimizer.

        Args:
            params: Model parameters to optimize
            base_optimizer: Underlying optimizer (Adam, SGD, etc.)
            model: The neural network model
            rho: Perturbation radius for sharpness computation (default: 0.2)
            lambda_: Weight for first-order flatness regularization (default: 0.2)
            adaptive: If True, use adaptive perturbation scaling (ASAM-style)
        """
        self.base_optimizer = base_optimizer
        self.model = model
        self.rho = rho
        self.lambda_ = lambda_
        self.adaptive = adaptive
        self.closure = None

        # Store parameter groups
        self.param_groups = base_optimizer.param_groups

        # State for storing original parameters
        self.state = {}

    def set_closure(self, closure: Callable):
        """
        Set the closure function that computes loss.

        Args:
            closure: Function that computes and returns (outputs, loss)
                    The loss should be a scalar tensor or list of loss tensors.
        """
        self.closure = closure

    @torch.no_grad()
    def _get_perturbation(self):
        """
        Compute perturbation vector for sharpness-aware optimization.

        Returns epsilon that maximizes loss within rho-ball around current params.
        Uses gradient ascent direction scaled by rho.
        """
        # Collect all gradients and parameters
        grad_norm = self._grad_norm()

        perturbations = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if self.adaptive:
                    # Adaptive perturbation (ASAM-style)
                    # Scale by parameter norm for better conditioning
                    scale = self.rho / (grad_norm + 1e-12)
                    scale *= torch.norm(p)
                    epsilon = scale * p.grad.sign()
                else:
                    # Standard perturbation
                    scale = self.rho / (grad_norm + 1e-12)
                    epsilon = scale * p.grad

                perturbations.append(epsilon)

        return perturbations

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        """Compute L2 norm of gradients across all parameters."""
        norm = torch.norm(
            torch.stack([
                p.grad.norm()
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ])
        )
        return norm

    @torch.no_grad()
    def _apply_perturbation(self, perturbations):
        """Apply perturbation to parameters (ascent step)."""
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Store original parameter
                if id(p) not in self.state:
                    self.state[id(p)] = {}
                self.state[id(p)]['old_p'] = p.data.clone()

                # Apply perturbation
                p.data.add_(perturbations[idx])
                idx += 1

    @torch.no_grad()
    def _restore_parameters(self):
        """Restore parameters to pre-perturbation state."""
        for group in self.param_groups:
            for p in group['params']:
                if id(p) in self.state and 'old_p' in self.state[id(p)]:
                    p.data = self.state[id(p)]['old_p']

    def step(self):
        """
        Perform C-Flat optimization step.

        This implements the two-pass sharpness-aware optimization:
        1. Compute gradient and perturbation
        2. Move to worst-case point in neighborhood (ascent)
        3. Compute gradient at perturbed point
        4. Update parameters using combined gradients (descent)

        Returns:
            outputs: Model outputs from the closure
            loss: Computed loss value(s)
        """
        if self.closure is None:
            raise RuntimeError("Must call set_closure() before step()")

        # Enable gradient computation for first pass
        with torch.enable_grad():
            # First forward-backward: Compute gradient at current position
            outputs, loss = self.closure()

            # Handle loss as list or single tensor
            if isinstance(loss, list):
                total_loss = sum(loss)
            else:
                total_loss = loss

            total_loss.backward()

        # Compute perturbation direction (toward high loss)
        perturbations = self._get_perturbation()

        # Apply perturbation (ascent step to worst-case in neighborhood)
        self._apply_perturbation(perturbations)

        # Zero gradients for second pass
        self.base_optimizer.zero_grad()

        # Second forward-backward: Compute gradient at perturbed position
        with torch.enable_grad():
            outputs_perturbed, loss_perturbed = self.closure()

            if isinstance(loss_perturbed, list):
                total_loss_perturbed = sum(loss_perturbed)
            else:
                total_loss_perturbed = loss_perturbed

            # First backward to get gradients at perturbed point
            total_loss_perturbed.backward(create_graph=(self.lambda_ > 0))

            # Add first-order flatness regularization
            # This penalizes large gradients in the neighborhood
            if self.lambda_ > 0:
                # Compute gradient norm at perturbed point as regularization
                grad_norm_perturbed = torch.tensor(0.0, device=total_loss_perturbed.device)
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is not None:
                            grad_norm_perturbed = grad_norm_perturbed + (p.grad.norm() ** 2)
                grad_norm_perturbed = torch.sqrt(grad_norm_perturbed)

                # Zero gradients and add flatness regularization
                self.base_optimizer.zero_grad()

                # Combined loss with flatness regularization
                total_loss_combined = total_loss_perturbed + self.lambda_ * grad_norm_perturbed
                total_loss_combined.backward()

        # Restore parameters to original position
        self._restore_parameters()

        # Update parameters using gradients from perturbed position
        # This moves toward flatter minima
        self.base_optimizer.step()

        # Zero gradients for next iteration
        self.base_optimizer.zero_grad()

        return outputs, loss

    def zero_grad(self):
        """Zero all gradients."""
        self.base_optimizer.zero_grad()

    def state_dict(self):
        """Return optimizer state dict."""
        return {
            'base_optimizer': self.base_optimizer.state_dict(),
            'rho': self.rho,
            'lambda_': self.lambda_,
            'adaptive': self.adaptive
        }

    def load_state_dict(self, state_dict):
        """Load optimizer state dict."""
        self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        self.rho = state_dict['rho']
        self.lambda_ = state_dict['lambda_']
        self.adaptive = state_dict.get('adaptive', False)
