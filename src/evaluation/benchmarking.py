import random
import time
import numpy as np
from tqdm import tqdm

class Benchmarker:
    """Class for benchmarking and comparing algorithms"""
    
    def __init__(self, graph_generator, ga_class, exact_algo_class, seed=None):
        """Initialize with algorithm classes"""
        self.graph_generator = graph_generator
        self.GA = ga_class
        self.ExactAlgo = exact_algo_class
        self.seed = seed
        self.verbose = False
    
    def compare_algorithms(self, num_instances=50, graph_size=100, pop_size=20, steps=250):
        """Compare GA and exact algorithm on random instances"""
        ga_results = []
        exact_results = []
        
        # Create a separate random generator for terminals with a derived but deterministic seed
        terminal_rng = random.Random()
        
        # If main seed is provided, derive terminal seed from it (adding a large prime number)
        terminal_seed = (self.seed + 104729) 
        terminal_rng.seed(terminal_seed)
        
        for i in tqdm(range(num_instances)):
            # Create new graph instance
            graph = self.graph_generator(graph_size)
            
            # Generate random terminals
            terminals = graph.generate_random_terminals(rng=terminal_rng)
            #agent1_start, agent2_start, agent1_dest, agent2_dest = terminals
            
            # Run exact algorithm
            exact_solver = self.ExactAlgo(graph)
            exact_result = exact_solver.solve_precomp(*terminals, verbose=False)
            exact_dist = exact_result['total_distance']
            
            # Run GA
            ga = self.GA(graph, pop_size=pop_size)
            ga.set_terminals(*terminals)
            ga_result = ga.run(steps=500, verbose=False)
            ga_dist = ga_result['total_distance']
            
            if self.verbose:
                if ga_dist < exact_dist:
                    print(f"WARNING: GA found better solution than exact algorithm!")
                    print(f"GA distance: {ga_dist}")
                    print(f"Exact distance: {exact_dist}")
                    print(f"Instance details: terminals={terminals}")
            
            ga_results.append(ga_dist)
            exact_results.append(exact_dist)
            
            if self.verbose:
                print(f"Instance {i+1}: Exact={exact_dist:.2f}, GA={ga_dist:.2f}")
            
        return ga_results, exact_results
    
    def compare_with_time_budget(self, num_instances=50, graph_size=100, pop_size=20):
        """Compare algorithms with equal time budgets"""
        ga_results = []
        exact_results = []
        exact_times = []
        
        for i in tqdm(range(num_instances)):
            # Create new graph instance
            graph = self.graph_generator(graph_size)
            
            # Generate random terminals
            terminals = graph.generate_random_terminals()
            
            # Time the exact algorithm
            exact_solver = self.ExactAlgo(graph)
            start_time = time.time()
            exact_result = exact_solver.solve(*terminals, verbose=False)
            exact_time = time.time() - start_time
            exact_dist = exact_result['total_distance']
            exact_times.append(exact_time)
            
            # Run GA with same time budget
            ga = self.GA(graph, pop_size=pop_size)
            ga.set_terminals(*terminals)
            ga_result = ga.run_with_time_budget(exact_time, verbose=False)
            ga_dist = ga_result['total_distance']
            
            if ga_dist < exact_dist:
                print(f"WARNING: GA found better solution than exact algorithm!")
                print(f"GA distance: {ga_dist}")
                print(f"Exact distance: {exact_dist}")
                print(f"Instance details: terminals={terminals}")
            
            ga_results.append(ga_dist)
            exact_results.append(exact_dist)
            
            print(f"Instance {i+1}: Exact={exact_dist:.2f} ({exact_time:.3f}s), GA={ga_dist:.2f}")
            
        print(f"Average Exact Algorithm Time: {np.mean(exact_times):.3f}s")
        
        return ga_results, exact_results, exact_times