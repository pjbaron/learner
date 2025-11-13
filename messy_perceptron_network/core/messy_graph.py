"""
Messy graph generator for the perceptron network.

Creates a weakly-connected directed graph with:
- Variable loop lengths (2 to 30+ steps)
- Three types of connections (signal, threshold modulation, plasticity modulation)
- No prescribed layers or hierarchy
- Optional distance constraints
"""

import numpy as np
import random
from typing import List, Tuple, Optional, Dict


class MessyGraphGenerator:
    """
    Generates messy recurrent graphs for the perceptron network.

    The graph has:
    - n_perceptrons nodes
    - ~avg_degree edges per node on average
    - 80% signal, 15% threshold modulation, 5% plasticity modulation connections
    - Variable loop lengths from 2 to 30+ steps
    - Weak connectivity (all nodes can influence each other through some path)
    """

    def __init__(self,
                 n_perceptrons=2000,
                 avg_degree=30,
                 signal_ratio=0.80,
                 threshold_mod_ratio=0.15,
                 plasticity_mod_ratio=0.05,
                 distance_constraint=None,
                 seed=None):
        """
        Initialize the messy graph generator.

        Args:
            n_perceptrons: Number of perceptrons in the network (default: 2000)
            avg_degree: Average number of outgoing connections per perceptron (default: 30)
            signal_ratio: Proportion of signal connections (default: 0.80)
            threshold_mod_ratio: Proportion of threshold modulation connections (default: 0.15)
            plasticity_mod_ratio: Proportion of plasticity modulation connections (default: 0.05)
            distance_constraint: Optional maximum distance for connections (default: None)
            seed: Random seed for reproducibility (default: None)
        """
        self.n_perceptrons = n_perceptrons
        self.avg_degree = avg_degree
        self.signal_ratio = signal_ratio
        self.threshold_mod_ratio = threshold_mod_ratio
        self.plasticity_mod_ratio = plasticity_mod_ratio
        self.distance_constraint = distance_constraint

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Validate ratios
        total_ratio = signal_ratio + threshold_mod_ratio + plasticity_mod_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"Connection type ratios must sum to 1.0, got {total_ratio}")

    def generate(self) -> Dict[str, List[Tuple[int, int]]]:
        """
        Generate a messy graph with three connection types.

        Returns:
            Dictionary with keys 'signal', 'threshold_mod', 'plasticity_mod',
            each containing a list of (source, target) edges
        """
        edges = {
            'signal': [],
            'threshold_mod': [],
            'plasticity_mod': []
        }

        # Step 1: Create backbone for weak connectivity
        # This ensures all perceptrons can influence each other through some path
        backbone_edges = self._create_backbone()

        # Step 2: Add random connections including backward edges (creates loops)
        all_edges = self._add_random_connections(backbone_edges)

        # Step 3: Assign connection types to edges
        edges = self._assign_connection_types(all_edges)

        # Step 4: Verify graph properties
        self._verify_graph(edges)

        return edges

    def _create_backbone(self) -> List[Tuple[int, int]]:
        """
        Create a backbone structure to ensure weak connectivity.

        Creates a forward chain plus some random forward connections.
        """
        backbone = []

        # Forward chain: 0->1->2->...->n-1
        for i in range(self.n_perceptrons - 1):
            backbone.append((i, i + 1))

        # Add some random forward connections to strengthen connectivity
        n_extra_backbone = self.n_perceptrons // 10
        for _ in range(n_extra_backbone):
            src = random.randint(0, self.n_perceptrons - 2)
            # Connect to a node further ahead
            dst = random.randint(src + 2, min(src + 20, self.n_perceptrons - 1))
            if (src, dst) not in backbone:
                backbone.append((src, dst))

        return backbone

    def _get_valid_targets(self, src: int) -> List[int]:
        """
        Get valid target nodes for a given source node.

        Applies distance constraint if specified.
        """
        if self.distance_constraint is None:
            # All other nodes are valid targets
            valid = list(range(self.n_perceptrons))
            valid.remove(src)  # No self-loops
            return valid
        else:
            # Only nodes within distance constraint
            valid = []
            for dst in range(self.n_perceptrons):
                if dst != src:  # No self-loops
                    distance = abs(dst - src)
                    if distance <= self.distance_constraint:
                        valid.append(dst)
            return valid

    def _add_random_connections(self, backbone_edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Add random connections to reach target average degree.

        Includes backward edges to create loops of various lengths.
        """
        edge_set = set(backbone_edges)
        target_total = int(self.n_perceptrons * self.avg_degree)

        # Add random edges until we reach target
        while len(edge_set) < target_total:
            src = random.randint(0, self.n_perceptrons - 1)
            valid_targets = self._get_valid_targets(src)

            if len(valid_targets) == 0:
                continue

            dst = random.choice(valid_targets)

            # Add the edge if it doesn't exist
            if (src, dst) not in edge_set:
                edge_set.add((src, dst))

        return list(edge_set)

    def _assign_connection_types(self, edges: List[Tuple[int, int]]) -> Dict[str, List[Tuple[int, int]]]:
        """
        Assign connection types to edges based on specified ratios.

        80% signal, 15% threshold modulation, 5% plasticity modulation.
        """
        # Shuffle edges for random assignment
        shuffled_edges = edges.copy()
        random.shuffle(shuffled_edges)

        total_edges = len(shuffled_edges)
        n_signal = int(total_edges * self.signal_ratio)
        n_threshold = int(total_edges * self.threshold_mod_ratio)
        # Remaining edges are plasticity modulation

        result = {
            'signal': shuffled_edges[:n_signal],
            'threshold_mod': shuffled_edges[n_signal:n_signal + n_threshold],
            'plasticity_mod': shuffled_edges[n_signal + n_threshold:]
        }

        return result

    def _verify_graph(self, edges: Dict[str, List[Tuple[int, int]]]):
        """
        Verify that the generated graph has the expected properties.

        Checks:
        - Total number of edges is approximately correct
        - Connection type ratios are approximately correct
        - No duplicate edges
        """
        total_edges = sum(len(edges[key]) for key in edges)
        expected_edges = int(self.n_perceptrons * self.avg_degree)

        # Check total edges (allow 5% tolerance)
        if abs(total_edges - expected_edges) > expected_edges * 0.05:
            print(f"Warning: Expected ~{expected_edges} edges, got {total_edges}")

        # Check for duplicate edges
        all_edges = []
        for edge_list in edges.values():
            all_edges.extend(edge_list)

        if len(all_edges) != len(set(all_edges)):
            print("Warning: Duplicate edges detected")

        # Print statistics
        print(f"Graph generated: {self.n_perceptrons} perceptrons, {total_edges} connections")
        print(f"  Signal: {len(edges['signal'])} ({100*len(edges['signal'])/total_edges:.1f}%)")
        print(f"  Threshold modulation: {len(edges['threshold_mod'])} ({100*len(edges['threshold_mod'])/total_edges:.1f}%)")
        print(f"  Plasticity modulation: {len(edges['plasticity_mod'])} ({100*len(edges['plasticity_mod'])/total_edges:.1f}%)")

    def analyze_loops(self, edges: Dict[str, List[Tuple[int, int]]]) -> Dict:
        """
        Analyze the loop structure of the generated graph.

        Returns:
            Dictionary with loop statistics
        """
        # Build adjacency list for all edge types combined
        adj = {i: [] for i in range(self.n_perceptrons)}
        for edge_type in edges:
            for src, dst in edges[edge_type]:
                adj[src].append(dst)

        # Find loops using DFS from each node
        loop_lengths = []

        def find_loops_from(start, max_depth=30):
            """Find all loops starting from a given node."""
            visited = {start: 0}
            stack = [(start, 0)]
            loops = []

            while stack:
                node, depth = stack.pop()

                if depth >= max_depth:
                    continue

                for neighbor in adj[node]:
                    if neighbor == start and depth > 0:
                        # Found a loop back to start
                        loops.append(depth + 1)
                    elif neighbor not in visited or visited[neighbor] > depth + 1:
                        visited[neighbor] = depth + 1
                        stack.append((neighbor, depth + 1))

            return loops

        # Sample a subset of nodes to find loops (checking all is expensive)
        sample_size = min(100, self.n_perceptrons)
        sampled_nodes = random.sample(range(self.n_perceptrons), sample_size)

        for node in sampled_nodes:
            loops = find_loops_from(node)
            loop_lengths.extend(loops)

        if len(loop_lengths) == 0:
            return {
                'num_loops_found': 0,
                'min_loop_length': None,
                'max_loop_length': None,
                'avg_loop_length': None,
                'loop_length_distribution': {}
            }

        # Compute statistics
        loop_dist = {}
        for length in loop_lengths:
            loop_dist[length] = loop_dist.get(length, 0) + 1

        return {
            'num_loops_found': len(loop_lengths),
            'min_loop_length': min(loop_lengths),
            'max_loop_length': max(loop_lengths),
            'avg_loop_length': np.mean(loop_lengths),
            'loop_length_distribution': loop_dist
        }


def create_messy_graph(n_perceptrons=2000, avg_degree=30, seed=None):
    """
    Convenience function to create a messy graph.

    Args:
        n_perceptrons: Number of perceptrons (default: 2000)
        avg_degree: Average connections per perceptron (default: 30)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with 'signal', 'threshold_mod', 'plasticity_mod' edge lists
    """
    generator = MessyGraphGenerator(
        n_perceptrons=n_perceptrons,
        avg_degree=avg_degree,
        seed=seed
    )
    edges = generator.generate()
    return edges
