from tqdm import tqdm
import time
import networkx as nx
from functools import lru_cache

class ExactAlgorithm:
    """Exact algorithm for the 2-Terminal Problem"""
    
    def __init__(self, graph):
        """Initialize with the graph"""
        self.graph = graph
    
    def solve(self, agent1_start, agent2_start, agent1_dest, agent2_dest, verbose=True):
        """Find the optimal solution by checking all possible meeting and dropping points"""
        min_sum = float('inf')
        best_m = None
        best_k = None
        
        G = self.graph.G
        
        start_time = time.time()
        
        if verbose:
            node_iterator = tqdm(G.nodes())
        else:
            node_iterator = G.nodes()
            
        for m in node_iterator:
            for k in G.nodes():
                if m != k:
                    try:
                        # Calculate path from terminals to first point m
                        sum_m = nx.shortest_path_length(G, agent1_start, m, weight='weight') + \
                                nx.shortest_path_length(G, agent2_start, m, weight='weight')
                        
                        # Calculate path between intermediate points
                        sum_k = nx.shortest_path_length(G, m, k, weight='weight')
                        
                        # Calculate path from second point k to terminals
                        sum_e = nx.shortest_path_length(G, k, agent1_dest, weight='weight') + \
                                nx.shortest_path_length(G, k, agent2_dest, weight='weight')
                        
                        total = sum_m + sum_k + sum_e
                        
                    except (nx.NodeNotFound, nx.NetworkXNoPath):
                        # Skip if nodes not found or no path exists
                        continue
                    
                    if total < min_sum:
                        min_sum = total
                        best_m = m
                        best_k = k
        
        runtime = time.time() - start_time
                        
        return {
            'solution': (best_m, best_k),
            'total_distance': min_sum,
            'runtime': runtime
        }
        
    def solve_precomp(self, agent1_start, agent2_start, agent1_dest, agent2_dest, verbose=True):
        """Find the optimal solution by checking all possible meeting and dropping points"""
        min_sum = float('inf')
        best_m = None
        best_k = None
        
        G = self.graph.G
        
        # Precompute all shortest paths
        shortest_paths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
        
        start_time = time.time()
        
        if verbose:
            node_iterator = tqdm(G.nodes())
        else:
            node_iterator = G.nodes()
            
        for m in node_iterator:
            for k in G.nodes():
                if m != k:
                    try:
                        # Use precomputed shortest paths
                        sum_m = shortest_paths[agent1_start][m] + shortest_paths[agent2_start][m]
                        sum_k = shortest_paths[m][k]
                        sum_e = shortest_paths[k][agent1_dest] + shortest_paths[k][agent2_dest]
                        
                        total = sum_m + sum_k + sum_e
                        
                    except KeyError:
                        # Skip if nodes not found in precomputed paths
                        continue
                    
                    if total < min_sum:
                        min_sum = total
                        best_m = m
                        best_k = k
        
        runtime = time.time() - start_time
                        
        return {
            'solution': (best_m, best_k),
            'total_distance': min_sum,
            'runtime': runtime
        }