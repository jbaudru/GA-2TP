import networkx as nx
import random

class TransportationGraph:
    """Class representing the graph for the 2-Terminal Problem"""
    
    def __init__(self, num_nodes=100):
        """Initialize a complete graph with random weights"""
        self.N = num_nodes
        self.G = nx.complete_graph(self.N)
        self._assign_random_weights()
        
    def _assign_random_weights(self):
        """Assign random weights to edges"""
        for (u, v) in self.G.edges():
            self.G[u][v]['weight'] = random.uniform(1, 10)
            self.G[u][v]['length'] = self.G[u][v]['weight']  # For compatibility
    
    def get_node_count(self):
        """Get the number of nodes in the graph"""
        return self.G.number_of_nodes()
    
    def get_path_distance(self, start, end):
        """Calculate shortest path distance between two nodes"""
        if start == end:
            return 0
        return nx.shortest_path_length(self.G, start, end, weight='weight')
    
    def load_from_json(self, filepath):
        """Load graph from JSON file"""
        import json
        with open(filepath, 'r') as f:
            graph_data = json.load(f)
        self.G = nx.node_link_graph(graph_data)
        self.N = self.G.number_of_nodes()
        return self
    
    def generate_random_terminals(self, rng=None):
        """Generate random start and end positions for two agents
        
        Args:
            rng: Optional random number generator to use (default: None)
        
        Returns:
            Tuple of (agent1_start, agent2_start, agent1_dest, agent2_dest)
        """
        nodes = list(self.G.nodes())
        # Use provided RNG or fall back to standard random module
        rand = rng if rng is not None else random
        
        agent1_start = rand.choice(nodes)
        
        agent2_start = rand.choice(nodes)
        while agent2_start == agent1_start:
            agent2_start = rand.choice(nodes)
            
        agent1_dest = rand.choice(nodes)
        while agent1_dest == agent1_start:
            agent1_dest = rand.choice(nodes)
            
        agent2_dest = rand.choice(nodes)
        while agent2_dest in [agent2_start, agent1_dest]:
            agent2_dest = rand.choice(nodes)
            
        return agent1_start, agent2_start, agent1_dest, agent2_dest