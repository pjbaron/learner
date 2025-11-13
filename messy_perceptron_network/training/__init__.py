"""
Training infrastructure for the messy perceptron network.
"""

from .trainer import MessyPerceptronTrainer
from .continual_learner import ContinualLearner

__all__ = [
    'MessyPerceptronTrainer',
    'ContinualLearner',
]
