"""
Graph analysis utilities for the messy perceptron network.

Tools for:
- Loop detection and analysis
- Connectivity analysis
- Path length computation
"""

import numpy as np
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple


class GraphAnalyzer:
    """
    Analyzes the structure of the messy perceptron graph.

    Provides tools for understanding emergent hierarchies through:
    - Loop length distribution
    - Connectivity patterns
    - Path analysis
    """

    def __init__(self, n_nodes, edges: Dict[str, List[Tuple[int, int]]]):
        """
        Initialize graph analyzer.

        Args:
            n_nodes: Number of nodes (perceptrons)
            edges: Dictionary with 'signal', 'threshold_mod', 'plasticity_mod' edge lists
        """
        self.n_nodes = n_nodes
        self.edges = edges

        # Build combined adjacency list (all edge types)
        self.adj_list = defaultdict(list)
        self.reverse_adj_list = defaultdict(list)

        for edge_type in edges:
            for src, dst in edges[edge_type]:
                self.adj_list[src].append(dst)
                self.reverse_adj_list[dst].append(src)

    def find_loops(self, max_loop_length=30, sample_size=100):
        """
        Find loops in the graph.

        Args:
            max_loop_length: Maximum loop length to search for
            sample_size: Number of nodes to sample for loop detection

        Returns:
            List of loop lengths found
        """
        loop_lengths = []

        # Sample nodes to search from (checking all is expensive)
        if sample_size >= self.n_nodes:
            sampled_nodes = list(range(self.n_nodes))
        else:
            sampled_nodes = np.random.choice(self.n_nodes, sample_size, replace=False)

        for start_node in sampled_nodes:
            loops = self._find_loops_from_node(start_node, max_loop_length)
            loop_lengths.extend(loops)

        return loop_lengths

    def _find_loops_from_node(self, start_node, max_depth):
        """
        Find all loops starting from a given node using BFS.

        Args:
            start_node: Node to start from
            max_depth: Maximum depth to search

        Returns:
            List of loop lengths
        """
        loops = []
        visited = {start_node: 0}
        queue = deque([(start_node, 0)])

        while queue:
            node, depth = queue.popleft()

            if depth >= max_depth:
                continue

            for neighbor in self.adj_list[node]:
                if neighbor == start_node and depth > 0:
                    # Found a loop back to start
                    loops.append(depth + 1)
                elif neighbor not in visited or visited[neighbor] > depth + 1:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1))

        return loops

    def compute_loop_statistics(self, max_loop_length=30, sample_size=100):
        """
        Compute statistics about loops in the graph.

        Returns:
            Dictionary with loop statistics
        """
        loop_lengths = self.find_loops(max_loop_length, sample_size)

        if len(loop_lengths) == 0:
            return {
                'num_loops': 0,
                'min_length': None,
                'max_length': None,
                'mean_length': None,
                'median_length': None,
                'length_distribution': {},
            }

        # Compute distribution
        length_dist = defaultdict(int)
        for length in loop_lengths:
            length_dist[length] += 1

        return {
            'num_loops': len(loop_lengths),
            'min_length': min(loop_lengths),
            'max_length': max(loop_lengths),
            'mean_length': np.mean(loop_lengths),
            'median_length': np.median(loop_lengths),
            'length_distribution': dict(length_dist),
        }

    def compute_degree_statistics(self):
        """
        Compute in-degree and out-degree statistics.

        Returns:
            Dictionary with degree statistics
        """
        in_degrees = [len(self.reverse_adj_list[i]) for i in range(self.n_nodes)]
        out_degrees = [len(self.adj_list[i]) for i in range(self.n_nodes)]

        return {
            'in_degree': {
                'mean': np.mean(in_degrees),
                'std': np.std(in_degrees),
                'min': np.min(in_degrees),
                'max': np.max(in_degrees),
            },
            'out_degree': {
                'mean': np.mean(out_degrees),
                'std': np.std(out_degrees),
                'min': np.min(out_degrees),
                'max': np.max(out_degrees),
            }
        }

    def find_strongly_connected_components(self):
        """
        Find strongly connected components using Tarjan's algorithm.

        Returns:
            List of strongly connected components (each is a list of node IDs)
        """
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = defaultdict(bool)
        components = []

        def strongconnect(node):
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            on_stack[node] = True
            stack.append(node)

            for neighbor in self.adj_list[node]:
                if neighbor not in index:
                    strongconnect(neighbor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
                elif on_stack[neighbor]:
                    lowlinks[node] = min(lowlinks[node], index[neighbor])

            if lowlinks[node] == index[node]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    component.append(w)
                    if w == node:
                        break
                components.append(component)

        for node in range(self.n_nodes):
            if node not in index:
                strongconnect(node)

        return components

    def analyze_connectivity(self):
        """
        Analyze overall connectivity of the graph.

        Returns:
            Dictionary with connectivity statistics
        """
        # Find strongly connected components
        components = self.find_strongly_connected_components()

        # Sort by size
        components.sort(key=len, reverse=True)

        return {
            'num_components': len(components),
            'largest_component_size': len(components[0]) if components else 0,
            'largest_component_fraction': len(components[0]) / self.n_nodes if components else 0,
            'is_strongly_connected': len(components) == 1,
        }

    def compute_shortest_paths(self, source, max_distance=10):
        """
        Compute shortest paths from a source node using BFS.

        Args:
            source: Source node
            max_distance: Maximum distance to compute

        Returns:
            Dictionary mapping node -> distance
        """
        distances = {source: 0}
        queue = deque([(source, 0)])

        while queue:
            node, dist = queue.popleft()

            if dist >= max_distance:
                continue

            for neighbor in self.adj_list[node]:
                if neighbor not in distances:
                    distances[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))

        return distances

    def analyze_node_centrality(self, sample_size=50):
        """
        Analyze node centrality using average path lengths.

        Args:
            sample_size: Number of source nodes to sample

        Returns:
            Dictionary with centrality statistics
        """
        if sample_size >= self.n_nodes:
            sampled_nodes = list(range(self.n_nodes))
        else:
            sampled_nodes = np.random.choice(self.n_nodes, sample_size, replace=False)

        avg_distances = []
        for node in sampled_nodes:
            distances = self.compute_shortest_paths(node, max_distance=20)
            if len(distances) > 1:
                avg_dist = np.mean(list(distances.values()))
                avg_distances.append(avg_dist)

        return {
            'mean_avg_distance': np.mean(avg_distances) if avg_distances else None,
            'network_diameter_estimate': np.max(avg_distances) if avg_distances else None,
        }

    def get_full_analysis(self):
        """
        Get complete graph analysis.

        Returns:
            Dictionary with all analysis results
        """
        print("Analyzing graph structure...")

        analysis = {}

        print("  Computing degree statistics...")
        analysis['degree_stats'] = self.compute_degree_statistics()

        print("  Finding loops...")
        analysis['loop_stats'] = self.compute_loop_statistics()

        print("  Analyzing connectivity...")
        analysis['connectivity'] = self.analyze_connectivity()

        print("  Computing centrality...")
        analysis['centrality'] = self.analyze_node_centrality()

        print("Graph analysis complete.")

        return analysis
