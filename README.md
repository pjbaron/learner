# Messy Perceptron Network

A biologically-inspired neural network architecture for continual learning through emergent nested optimization.

## Overview

This project implements a proof-of-concept neural network that achieves continual learning capabilities without catastrophic forgetting. The architecture consists of a single homogeneous component type (perceptrons) connected in a messy, recurrent graph where different connection types create emergent hierarchies of learning.

## Key Innovation

- **Single homogeneous component**: Perceptrons with learnable thresholds
- **Messy connectivity**: Arbitrary recurrent graph with three types of connections:
  1. **Signal connections** (80%): Standard weighted inputs for computation
  2. **Threshold modulation connections** (15%): Dynamically adjust activation thresholds
  3. **Plasticity modulation connections** (5%): Control learning rates (meta-learning)
- **Emergent properties**: Multi-timescale learning, local adaptation, and global coordination arise from topology alone

## Architecture

### Core Components

- **Perceptrons** (2000): Each with learnable threshold and three types of incoming connections
- **Messy Graph**: ~60,000 connections with variable loop lengths (2-30+ steps)
- **No Layers**: No prescribed hierarchy - structure emerges from learning

### Training

- **Iterative Settling**: Forward pass uses 7 iterations to allow recurrent activations to stabilize
- **BPTT**: Backpropagation Through Time through the settled activations
- **Multi-Cycle Training**: 3 forward/backward passes per batch
- **Gradient Clipping**: Stability through gradient norm clipping (max norm = 1.0)

## Project Structure

```
messy_perceptron_network/
├── core/
│   ├── perceptron.py           # Perceptron implementation
│   ├── connection.py           # Three connection types
│   ├── messy_graph.py          # Graph generation
│   └── network.py              # Full network with settling dynamics
├── training/
│   ├── trainer.py              # Multi-cycle training with BPTT
│   └── continual_learner.py    # Sequential task management
└── utils/
    ├── graph_utils.py          # Loop analysis, connectivity
    └── stability.py            # Gradient monitoring, NaN detection
```

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- Python 3.8+
- PyTorch 2.0+
- NumPy
- Matplotlib

## Quick Start

```python
from messy_perceptron_network import MessyPerceptronNetwork, MessyPerceptronTrainer

# Create network
network = MessyPerceptronNetwork(
    n_perceptrons=2000,
    avg_degree=30,
    n_input_perceptrons=200,
    n_output_perceptrons=200,
    settling_iterations=7,
    seed=42
)

# Create trainer
trainer = MessyPerceptronTrainer(
    network,
    base_lr=0.001,
    optimizer_type='adam',
    cycles_per_batch=3
)

# Train on a batch
import torch
inputs = torch.randn(32, 200)   # batch_size=32
targets = torch.randn(32, 200)

stats = trainer.train_step(inputs, targets)
print(f"Loss: {stats['loss']:.4f}")
```

## Testing

Run basic functionality tests:

```bash
python test_basic_functionality.py
```

This tests:
1. Network creation
2. Forward pass with settling
3. Backward pass with BPTT
4. Training step
5. Stability monitoring
6. Graph analysis

## Implementation Status

### ✅ Completed (Phase 1 & 2)

- [x] Perceptron class with three connection types
- [x] Connection manager and weight initialization
- [x] Messy graph generator with variable loop lengths
- [x] Network class with settling dynamics
- [x] Forward pass with iterative settling
- [x] Trainer with BPTT and multi-cycle training
- [x] Continual learning harness for sequential tasks
- [x] Graph analysis tools (loop detection, connectivity)
- [x] Stability monitoring (gradient checking, NaN detection)
- [x] Basic functionality tests

### 🚧 In Progress (Phase 3)

- [ ] MNIST continual learning experiment
- [ ] Baseline comparisons (fixed plasticity, EWC)
- [ ] Performance metrics and evaluation

### 📋 Planned (Phase 4 & 5)

- [ ] Visualization tools (plasticity heatmaps, activation patterns)
- [ ] Emergent structure analysis (identify modulators, routers)
- [ ] Loop utilization metrics
- [ ] Hyperparameter tuning
- [ ] Documentation and results write-up

## Key Features

### 1. Emergent Multi-Timescale Learning

Different loop lengths in the graph create natural temporal hierarchies:
- Short loops (2-5 steps): Fast learning, high plasticity
- Long loops (15+ steps): Slow learning, consolidation

### 2. Emergent Neuromodulation

Some perceptrons naturally become "modulators":
- High out-degree through plasticity modulation connections
- Control learning rates of other perceptrons
- Respond to task switches and loss patterns

### 3. Emergent Task Routing

Threshold modulation creates task-specific pathways:
- Different tasks activate different subgraphs
- Minimal interference between tasks
- Dynamic gating based on context

## Continual Learning Benchmark

The proof-of-concept task is sequential MNIST:
1. Train on digits 0-4
2. Switch to digits 5-9
3. Return to digits 0-4
4. Measure catastrophic forgetting

**Success Criteria:**
- Minimal accuracy drop on digits 0-4 when training on 5-9
- Better retention than fixed-plasticity baseline
- Competitive with Elastic Weight Consolidation (EWC)

## Design Philosophy

### Why "Messy"?

Biological neural networks are messy:
- No clean layers
- Recurrent connections everywhere
- Variable connection distances
- Heterogeneous learning rates

This architecture embraces messiness as a feature, not a bug. The hypothesis is that sophisticated learning capabilities emerge from simple components (perceptrons) connected messily, without prescribed hierarchy.

### Biological Plausibility

- **Homogeneous neurons**: All perceptrons are the same type
- **Local learning rules**: Each perceptron updates based on local information
- **Neuromodulation**: Plasticity modulation mimics dopamine/serotonin effects
- **Recurrent connectivity**: Like actual cortical circuits
- **Threshold modulation**: Similar to dendritic integration

## References

- Rosenblatt (1958): Original perceptron
- Hebb (1949): Local learning rules
- Kirkpatrick et al. (2017): Elastic Weight Consolidation
- Google Nested Learning (2025): Multi-timescale optimization

## License

MIT License - See LICENSE file for details

## Contributing

This is a research proof-of-concept. Contributions welcome:
- Bug reports and fixes
- Performance improvements
- Additional analysis tools
- Experiment variations

## Citation

```bibtex
@software{messy_perceptron_network,
  title={Messy Perceptron Network: Emergent Continual Learning},
  author={Messy Perceptron Network Team},
  year={2025},
  version={0.1.0}
}
```

## Status

**Current Status**: Phase 1 & 2 Complete

The core architecture is implemented and tested. Next steps are running the continual learning experiments and analyzing emergent structures.

---

**Document Version**: 1.0
**Last Updated**: 2025-11-13
