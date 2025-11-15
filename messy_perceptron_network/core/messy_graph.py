"""
Messy graph generator for the perceptron network.

Creates a deep recurrent graph with:
- 20-30 depth levels from input to output
- Forward connections advancing through depth levels
- Backward recurrent loops jumping back through levels
- Three types of connections (signal, threshold modulation, plasticity modulation)
"""

import numpy as np
import random
from typing import List, Tuple, Optional, Dict


class MessyGraphGenerator:
    """
    Generates deep messy recurrent graphs for the perceptron network.

    The graph has:
    - n_perceptrons nodes assigned to ~30 depth levels
    - Forward connections advancing through levels
    - Backward recurrent loops for multi-timescale learning
    - ~avg_degree edges per node on average
    - 80% signal, 15% threshold modulation, 5% plasticity modulation connections
    """

    def __init__(self,
                 n_perceptrons=2000,
                 avg_degree=30,
                 n_input_perceptrons=250,
                 n_output_perceptrons=125,
                 signal_ratio=0.80,
                 threshold_mod_ratio=0.15,
                 plasticity_mod_ratio=0.05,
                 seed=None):
        """
        Initialize the messy graph generator.

        Args:
            n_perceptrons: Number of perceptrons in the network (default: 2000)
            avg_degree: Average number of outgoing connections per perceptron (default: 30)
            n_input_perceptrons: Number of input neurons (default: 250)
            n_output_perceptrons: Number of output neurons (default: 125)
            signal_ratio: Proportion of signal connections (default: 0.80)
            threshold_mod_ratio: Proportion of threshold modulation connections (default: 0.15)
            plasticity_mod_ratio: Proportion of plasticity modulation connections (default: 0.05)
            seed: Random seed for reproducibility (default: None)
        """
        self.n_perceptrons = n_perceptrons
        self.avg_degree = avg_degree
        self.n_input_perceptrons = n_input_perceptrons
        self.n_output_perceptrons = n_output_perceptrons
        self.signal_ratio = signal_ratio
        self.threshold_mod_ratio = threshold_mod_ratio
        self.plasticity_mod_ratio = plasticity_mod_ratio

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Validate ratios
        total_ratio = signal_ratio + threshold_mod_ratio + plasticity_mod_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"Connection type ratios must sum to 1.0, got {total_ratio}")

        # Will be computed during generation
        self.neuron_distances = None  # Distance from each neuron to nearest output
        self.input_indices = None
        self.output_indices = None

    def generate(self) -> Dict[str, List[Tuple[int, int]]]:
        """
        Generate a deep messy graph with emergent layer structure.

        Uses distance-to-output to create natural depth without explicit layers.

        Returns:
            Dictionary with keys 'signal', 'threshold_mod', 'plasticity_mod',
            each containing a list of (source, target) edges
        """
        edges = {
            'signal': [],
            'threshold_mod': [],
            'plasticity_mod': []
        }

        # Step 1: Randomly select output neurons
        self.output_indices = random.sample(range(self.n_perceptrons), self.n_output_perceptrons)

        # Step 2: Build initial random connectivity to establish graph structure
        print("Building initial connectivity...")
        initial_edges = self._build_initial_connectivity()

        # Step 3: Compute distance-to-output for all neurons
        print("Computing distance-to-output for all neurons...")
        self.neuron_distances = self._compute_distances_to_outputs(initial_edges)

        # Step 4: Select inputs from neurons furthest from outputs
        self._select_input_neurons()

        # Step 5: Rebuild connections based on distance-to-output
        print("Rebuilding connections based on computed distances...")
        all_edges = self._build_distance_based_connections()

        # Step 6: Assign connection types to edges
        edges = self._assign_connection_types(all_edges)

        # Step 7: Verify graph properties
        self._verify_graph(edges)

        return edges

    def _build_initial_connectivity(self) -> List[Tuple[int, int]]:
        """
        Build initial connectivity with strong forward bias to create depth.

        Creates a sequential chain-like structure with skip connections
        to establish deep paths from inputs to outputs.
        """
        edges = []

        # Assign temporary positions to neurons (outputs at start for backwards BFS)
        # Shuffle non-output neurons to randomize which become deep vs shallow
        non_output_indices = [i for i in range(self.n_perceptrons) if i not in self.output_indices]
        random.shuffle(non_output_indices)

        # Build sequential chain from outputs backwards
        # This ensures outputs are reachable and creates depth
        for i in range(len(non_output_indices) - 1):
            src = non_output_indices[i]
            dst = non_output_indices[i + 1]
            edges.append((src, dst))

        # Connect chain to outputs
        for _ in range(100):  # Multiple connections to outputs
            src = random.choice(non_output_indices[:50])  # From early in chain
            dst = random.choice(self.output_indices)
            if (src, dst) not in edges:
                edges.append((src, dst))

        # Add skip connections (forward jumps of varying lengths)
        target_edges = int(self.n_perceptrons * self.avg_degree)
        while len(edges) < target_edges:
            # Pick from non-outputs
            src_idx = random.randint(50, len(non_output_indices) - 1)
            # Jump forward toward outputs (smaller index = closer to output)
            jump = random.randint(1, min(50, src_idx))
            dst_idx = src_idx - jump

            if dst_idx >= 0:
                if dst_idx < len(non_output_indices):
                    src = non_output_indices[src_idx]
                    dst = non_output_indices[dst_idx]
                else:
                    # Jump to output
                    src = non_output_indices[src_idx]
                    dst = random.choice(self.output_indices)

                if src != dst and (src, dst) not in edges:
                    edges.append((src, dst))

        return edges

    def _compute_distances_to_outputs(self, edges: List[Tuple[int, int]]) -> np.ndarray:
        """
        Compute shortest distance from each neuron to nearest output neuron.

        Uses BFS backwards from outputs through reversed edges.

        Returns:
            Array of distances (or np.inf if unreachable)
        """
        from collections import deque

        # Build reverse adjacency list (backwards from outputs)
        reverse_adj = {i: [] for i in range(self.n_perceptrons)}
        for src, dst in edges:
            reverse_adj[dst].append(src)  # Reverse direction

        # BFS from all outputs simultaneously
        distances = np.full(self.n_perceptrons, np.inf)
        queue = deque()

        for output_idx in self.output_indices:
            distances[output_idx] = 0
            queue.append((output_idx, 0))

        visited = set(self.output_indices)

        while queue:
            node, dist = queue.popleft()

            for predecessor in reverse_adj[node]:
                if predecessor not in visited:
                    visited.add(predecessor)
                    distances[predecessor] = dist + 1
                    queue.append((predecessor, dist + 1))

        return distances

    def _select_input_neurons(self):
        """
        Select input neurons from those at distance 20-30 from outputs.

        Targets the "sweet spot" for 20-30 hop paths to outputs.
        """
        # Get neurons at target distance range
        target_distances = list(range(20, 31))  # 20-30 hops from output
        candidates = [i for i in range(self.n_perceptrons)
                     if i not in self.output_indices and
                     int(self.neuron_distances[i]) in target_distances]

        # If not enough candidates in target range, expand search
        if len(candidates) < self.n_input_perceptrons:
            print(f"  Warning: Only {len(candidates)} neurons at distance 20-30, expanding search...")
            candidates = [i for i in range(self.n_perceptrons)
                         if i not in self.output_indices and
                         self.neuron_distances[i] >= 15]

        # Shuffle and take required number
        random.shuffle(candidates)
        self.input_indices = candidates[:self.n_input_perceptrons]

        avg_input_dist = np.mean([self.neuron_distances[i] for i in self.input_indices])
        print(f"Selected inputs with average distance-to-output: {avg_input_dist:.1f}")

    def _build_distance_based_connections(self) -> List[Tuple[int, int]]:
        """
        Build connections based on distance-to-output.

        Forward connections: src_distance > dst_distance (moving toward outputs)
        Backward loops: src_distance < dst_distance (recurrent, away from outputs)

        Creates ~60% forward, ~40% backward to maintain recurrence.
        """
        # Pre-compute neurons at each distance for efficient lookup
        distance_bins = {}
        for neuron_idx in range(self.n_perceptrons):
            dist = int(self.neuron_distances[neuron_idx])
            if not np.isinf(dist):
                if dist not in distance_bins:
                    distance_bins[dist] = []
                distance_bins[dist].append(neuron_idx)

        max_dist = max(distance_bins.keys()) if distance_bins else 0
        print(f"  Distance range: 0 to {max_dist} hops")

        edges = set()
        target_total = int(self.n_perceptrons * self.avg_degree)

        # 60% forward (toward outputs - decreasing distance)
        n_forward = int(target_total * 0.6)

        for _ in range(n_forward * 2):  # Extra attempts
            if len(edges) >= n_forward:
                break

            # Pick source from mid-to-high distance
            src_dist = random.randint(1, max_dist)
            if src_dist not in distance_bins:
                continue

            src = random.choice(distance_bins[src_dist])

            # Connect forward (decrease distance by 1-2 hops)
            # Mostly 1-hop (deep paths) with some 2-hop (shortcuts) for variety
            advance = 1 if random.random() < 0.9 else 2  # 90% single-hop, 10% double-hop
            dst_dist = max(0, src_dist - advance)
            if dst_dist not in distance_bins:
                continue

            dst = random.choice(distance_bins[dst_dist])

            if src != dst:
                edges.add((src, dst))

        # 40% backward (away from outputs - increasing distance)
        n_backward = target_total - len(edges)

        for _ in range(n_backward * 2):  # Extra attempts
            if len(edges) >= target_total:
                break

            # Pick source from low-to-mid distance
            src_dist = random.randint(0, max(1, max_dist - 5))
            if src_dist not in distance_bins:
                continue

            src = random.choice(distance_bins[src_dist])

            # Jump back 2-25 hops (recurrent loop)
            jump_back = random.choice([
                random.randint(2, 5),    # Short loops (more common)
                random.randint(6, 15),   # Medium loops
                random.randint(16, 25)   # Long loops (less common)
            ])

            dst_dist = min(max_dist, src_dist + jump_back)
            if dst_dist not in distance_bins:
                continue

            dst = random.choice(distance_bins[dst_dist])

            if src != dst:
                edges.add((src, dst))

        print(f"  Built {len(edges)} distance-based connections")
        return list(edges)

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
        - Forward/backward distribution based on distance-to-output
        """
        total_edges = sum(len(edges[key]) for key in edges)
        expected_edges = int(self.n_perceptrons * self.avg_degree)

        # Check total edges (allow 15% tolerance for distance-based generation)
        if abs(total_edges - expected_edges) > expected_edges * 0.15:
            print(f"Warning: Expected ~{expected_edges} edges, got {total_edges}")

        # Check for duplicate edges
        all_edges = []
        for edge_list in edges.values():
            all_edges.extend(edge_list)

        if len(all_edges) != len(set(all_edges)):
            print("Warning: Duplicate edges detected")

        # Analyze distance-based characteristics
        forward_count = 0  # Toward outputs (decreasing distance)
        backward_count = 0  # Away from outputs (increasing distance)
        lateral_count = 0   # Same distance

        for src, dst in all_edges:
            src_dist = self.neuron_distances[src]
            dst_dist = self.neuron_distances[dst]

            if src_dist > dst_dist:  # Moving closer to output
                forward_count += 1
            elif src_dist < dst_dist:  # Moving away from output (recurrent)
                backward_count += 1
            else:
                lateral_count += 1

        # Compute distance statistics
        input_dists = [self.neuron_distances[i] for i in self.input_indices]
        output_dists = [self.neuron_distances[i] for i in self.output_indices]

        # Print statistics
        print(f"\nGraph generated: {self.n_perceptrons} perceptrons, {total_edges} connections")
        print(f"  Input neurons: {len(self.input_indices)} at avg distance {np.mean(input_dists):.1f} from outputs")
        print(f"  Output neurons: {len(self.output_indices)} at distance 0 (by definition)")
        print(f"  Max distance-to-output: {np.max(self.neuron_distances[~np.isinf(self.neuron_distances)]):.0f} hops")
        print(f"  Forward connections (toward outputs): {forward_count} ({100*forward_count/total_edges:.1f}%)")
        print(f"  Backward connections (recurrent loops): {backward_count} ({100*backward_count/total_edges:.1f}%)")
        print(f"  Lateral connections (same distance): {lateral_count} ({100*lateral_count/total_edges:.1f}%)")
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


def create_messy_graph(n_perceptrons=2000, avg_degree=30, n_input_perceptrons=250, n_output_perceptrons=125, seed=None):
    """
    Convenience function to create a deep messy graph with emergent layer structure.

    Args:
        n_perceptrons: Number of perceptrons (default: 2000)
        avg_degree: Average connections per perceptron (default: 30)
        n_input_perceptrons: Number of input neurons (default: 250)
        n_output_perceptrons: Number of output neurons (default: 125)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (edges, neuron_distances, input_indices, output_indices) where:
        - edges: Dictionary with 'signal', 'threshold_mod', 'plasticity_mod' edge lists
        - neuron_distances: numpy array of distance-to-output for each neuron
        - input_indices: list of input neuron indices
        - output_indices: list of output neuron indices
    """
    generator = MessyGraphGenerator(
        n_perceptrons=n_perceptrons,
        avg_degree=avg_degree,
        n_input_perceptrons=n_input_perceptrons,
        n_output_perceptrons=n_output_perceptrons,
        seed=seed
    )
    edges = generator.generate()
    return edges, generator.neuron_distances, generator.input_indices, generator.output_indices
