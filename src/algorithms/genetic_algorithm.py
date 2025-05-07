import random
import time
import numpy as np
import networkx as nx

class GeneticAlgorithm:
    """Genetic algorithm implementation for the 2-Terminal Problem"""
    
    def __init__(self, graph, pop_size=20, seed=42, mutation_rate=0.1, encode_len=8):
        """Initialize the GA with the problem parameters"""
        self.graph = graph
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.encode_len = encode_len
        self.bit_count = 2 * encode_len  # For meeting and dropping points
        self.terminals = None
        self.fitness_cache = {}  # Cache for fitness evaluations
        
        # Set random seed for reproducibility
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        
    def set_terminals(self, agent1_start, agent2_start, agent1_dest, agent2_dest):
        """Set the terminal nodes for the problem"""
        self.terminals = (agent1_start, agent2_start, agent1_dest, agent2_dest)
    
    def generate_individual(self):
        """Generate a random individual"""
        bits = 0
        for _ in range(self.bit_count):
            bits = (bits << 1) | random.getrandbits(1)
            
        return bits
    
    def byte_to_number(self, bit_data):
        """Convert bit representation to actual node IDs in the graph."""
        numbers = []
        bits_per_var = self.bit_count // 2
        max_int = (1 << bits_per_var) - 1

        # Get the list of actual node IDs from the graph
        node_ids = list(self.graph.G.nodes)
        num_nodes = len(node_ids)

        for i in range(2):  # Two variables: meeting point and dropping point
            number = 0
            for _ in range(bits_per_var):
                number = (number << 1) | (bit_data & 1)
                bit_data >>= 1
            node_index = number % num_nodes  # Map to a valid index in the node list
            numbers.append(node_ids[node_index])  # Get the actual node ID

        return numbers[::-1]
        
    def fitness(self, individual):
        """Calculate fitness of an individual with caching"""
        if individual in self.fitness_cache:
            return self.fitness_cache[individual]
            
        if not self.terminals:
            raise ValueError("Terminal nodes not set")
            
        agent1_start, agent2_start, agent1_dest, agent2_dest = self.terminals
        points = self.byte_to_number(individual)
        meeting_point, dropping_point = points
        
        #print("Meeting point:", meeting_point, "Dropping point:", dropping_point)
        
        if meeting_point == dropping_point:
            self.fitness_cache[individual] = float('-inf')
            return float('-inf')  # Penalize same points
        
        # Calculate using path distances
        try:
            sum_m = self.graph.get_path_distance(agent1_start, meeting_point) + \
                    self.graph.get_path_distance(agent2_start, meeting_point)
            
            sum_k = self.graph.get_path_distance(meeting_point, dropping_point)
            
            sum_e = self.graph.get_path_distance(dropping_point, agent1_dest) + \
                    self.graph.get_path_distance(dropping_point, agent2_dest)
            
            total = sum_m + sum_k + sum_e
        except:
            self.fitness_cache[individual] = float('-inf')
            return float('-inf')
        
        result = -total  # Negative since GA maximizes fitness
        self.fitness_cache[individual] = result
        return result
    
    def populate(self):
        """Generate initial population"""
        return [self.generate_individual() for _ in range(self.pop_size)]
    
    def tournament_selection(self, population, k=None):
        """Tournament selection for parents"""
        if k is None:
            k = max(2, self.pop_size // 10)
        contenders = random.sample(population, k)
        return max(contenders, key=self.fitness)
    
    def crossover(self, parent1, parent2):
        """Perform crossover between parents"""
        child = 0
        for i in range(self.bit_count):
            pick = (parent1 if random.random() < 0.5 else parent2)
            bit = (pick >> i) & 1
            child |= (bit << i)
        return child
    
    def mutation(self, individual):
        """Mutate an individual"""
        for i in range(self.bit_count):
            if random.random() < self.mutation_rate:
                individual ^= (1 << i)
        return individual
    
    def evolve(self, population, tabu_list=None):
        """Create a new individual through evolution with tabu list"""
        if tabu_list is None:
            tabu_list = set()
            
        max_attempts = 10  # Prevent infinite loops
        attempts = 0
        
        while attempts < max_attempts:
            p1 = self.tournament_selection(population)
            p2 = self.tournament_selection(population)
            c = self.crossover(p1, p2)
            c = self.mutation(c)
            
            if c not in tabu_list:
                return c
                
            attempts += 1
        
        # If we couldn't find a non-tabu solution, generate random one
        return self.generate_individual()
    
    def run(self, steps=500, verbose=True):
        """Run the genetic algorithm for a specified number of steps"""
        if not self.terminals:
            raise ValueError("Terminal nodes not set")
            
        population = self.populate()
        population_history = []
        fitness_history = []
        best_fitness = float('-inf')
        best_solution = None
        
        evaluated_solutions = set()  # Track unique solutions evaluated
        
        for i in range(steps):
            population_history.append(population[:])
            
            # Track unique solutions
            for ind in population:
                evaluated_solutions.add(ind)
                
            current_fitness = [self.fitness(ind) for ind in population]
            fitness_history.append(current_fitness)
            
            # Generate new unique individual
            new_individual = self.evolve(population)
            while new_individual in evaluated_solutions:
                new_individual = self.evolve(population)
            
            population.append(new_individual)
            population = sorted(population, key=self.fitness, reverse=True)[:self.pop_size]
            
            top_fit = self.fitness(population[0])
            if top_fit > best_fitness:
                best_fitness = top_fit
                best_solution = population[0]
                if verbose:
                    points = self.byte_to_number(population[0])
                    print(f"Step: {i} Fitness: {top_fit:.4f}")
                    print(f"Meeting point: {points[0]}, Dropping point: {points[1]}")
        
        # Make sure we have at least one solution
        if best_solution is None and population:
            best_solution = population[0]
            best_fitness = self.fitness(best_solution)
        
        # Handle case where no solution was found (should be extremely rare)
        if best_solution is None:
            best_solution = self.generate_individual()
            best_fitness = self.fitness(best_solution)
        
        final_solution = self.byte_to_number(best_solution)
        if verbose:
            print("Final Solution:")
            print(f"Meeting point: {final_solution[0]}")
            print(f"Dropping point: {final_solution[1]}")
            print(f"Final fitness: {best_fitness:.4f} \n\n")
        
        return {
            'solution': final_solution,
            'fitness': best_fitness,
            'total_distance': -best_fitness,
            'population_history': population_history,
            'fitness_history': fitness_history
        }
    
    def run_with_time_budget(self, time_budget, verbose=True):
        """Run the GA with a time budget while tracking the same metrics as run()"""
        if not self.terminals:
            raise ValueError("Terminal nodes not set")
                
        population = self.populate()
        population_history = []
        fitness_history = []
        best_fitness = float('-inf')
        best_solution = None
        
        evaluated_solutions = set()  # Track unique solutions evaluated
        
        start_time = time.time()
        iterations = 0
        
        # Ensure we run at least one iteration
        end_time = time.time()
        while end_time - start_time < time_budget:
            iterations += 1
            population_history.append(population[:])
            
            # Track unique solutions
            for ind in population:
                evaluated_solutions.add(ind)
                    
            current_fitness = [self.fitness(ind) for ind in population]
            fitness_history.append(current_fitness)
            
            # Generate new unique individual
            new_individual = self.evolve(population)
            attempt_count = 0
            max_attempts = 100  # Prevent too many attempts
            while new_individual in evaluated_solutions and time.time() - start_time < time_budget:
                new_individual = self.evolve(population)
                attempt_count += 1
                if attempt_count >= max_attempts:
                    break
            
            population.append(new_individual)
            population = sorted(population, key=self.fitness, reverse=True)[:self.pop_size]
            
            top_fit = self.fitness(population[0])
            if top_fit > best_fitness:
                best_fitness = top_fit
                best_solution = population[0]
                if verbose:
                    points = self.byte_to_number(best_solution)
                    print(f"Iteration: {iterations} Fitness: {top_fit:.4f}")
                    print(f"Meeting point: {points[0]}, Dropping point: {points[1]}")
            
            end_time = time.time()
        
        # Make sure we have at least one solution
        if best_solution is None and population:
            best_solution = population[0]
            best_fitness = self.fitness(best_solution)
        
        # Handle case where no solution was found (should be extremely rare)
        if best_solution is None:
            best_solution = self.generate_individual()
            best_fitness = self.fitness(best_solution)
        
        final_solution = self.byte_to_number(best_solution)
        
        if verbose:
            elapsed_time = time.time() - start_time
            print(f"Completed {iterations} iterations in {elapsed_time:.2f}s")
            print("Final Solution:")
            print(f"Meeting point: {final_solution[0]}")
            print(f"Dropping point: {final_solution[1]}")
            print(f"Final fitness: {best_fitness:.4f} \n\n")
        
        return {
            'solution': final_solution,
            'fitness': best_fitness,
            'total_distance': -best_fitness,
            'population_history': population_history,
            'fitness_history': fitness_history,
            'iterations': iterations
        }
        
    def run_on_real_network(self, steps=500, verbose=False, budget=False, time_budget=None):
        """
        Run the genetic algorithm on a real network using the graph data already loaded in self.graph.

        Args:
            steps (int): Number of steps to run the algorithm.
            verbose (bool): Whether to print detailed output.
        """
        if not self.terminals:
            raise ValueError("Terminal nodes not set")

        # Ensure the graph has the required node data
        if not hasattr(self.graph, 'G') or not isinstance(self.graph.G, nx.Graph):
            raise ValueError("Graph does not contain valid node data")

        population = self.populate()
        population_history = []
        fitness_history = []
        best_fitness = float('-inf')
        best_solution = None

        evaluated_solutions = set()  # Track unique solutions evaluated

        if not budget:
            # Run for a fixed number of steps
            for i in range(steps):
                population_history.append(population[:])

                # Track unique solutions
                for ind in population:
                    evaluated_solutions.add(ind)

                current_fitness = [self.fitness(ind) for ind in population]
                fitness_history.append(current_fitness)

                # Generate new unique individual
                new_individual = self.evolve(population)
                while new_individual in evaluated_solutions:
                    new_individual = self.evolve(population)

                population.append(new_individual)
                population = sorted(population, key=self.fitness, reverse=True)[:self.pop_size]

                top_fit = self.fitness(population[0])
                if top_fit > best_fitness:
                    best_fitness = top_fit
                    best_solution = population[0]
                    if verbose:
                        points = self.byte_to_number(population[0])
                        meeting_point = self.graph.G.nodes.get(points[0], {"x": "Unknown", "y": "Unknown"})
                        dropping_point = self.graph.G.nodes.get(points[1], {"x": "Unknown", "y": "Unknown"})
                        print(f"Step: {i} Fitness: {top_fit:.4f}")
                        print(f"Meeting point: ({meeting_point['x']}, {meeting_point['y']}), "
                            f"Dropping point: ({dropping_point['x']}, {dropping_point['y']})")

        else:
            if verbose:
                print("[DEBUG] Running with time budget...")
            # Run with a time budget
            start_time = time.time()
            iterations = 0

            # Ensure we run at least one iteration
            end_time = time.time()
            while end_time - start_time < time_budget:
                iterations += 1
                population_history.append(population[:])

                # Track unique solutions
                for ind in population:
                    evaluated_solutions.add(ind)

                current_fitness = [self.fitness(ind) for ind in population]
                fitness_history.append(current_fitness)

                # Generate new unique individual
                new_individual = self.evolve(population)
                attempt_count = 0
                max_attempts = 100
                while new_individual in evaluated_solutions and time.time() - start_time < time_budget:
                    new_individual = self.evolve(population)
                    attempt_count += 1
                    if attempt_count >= max_attempts:
                        break
                
                population.append(new_individual)
                population = sorted(population, key=self.fitness, reverse=True)[:self.pop_size]
                
                top_fit = self.fitness(population[0])
                if top_fit > best_fitness:
                    best_fitness = top_fit
                    best_solution = population[0]
                    if verbose:
                        points = self.byte_to_number(population[0])
                        meeting_point = self.graph.G.nodes.get(points[0], {"x": "Unknown", "y": "Unknown"})
                        dropping_point = self.graph.G.nodes.get(points[1], {"x": "Unknown", "y": "Unknown"})
                        print(f"Iteration: {iterations} Fitness: {top_fit:.4f}")
                        print(f"Meeting point: ({meeting_point['x']}, {meeting_point['y']}), "
                            f"Dropping point: ({dropping_point['x']}, {dropping_point['y']})")
                
                end_time = time.time()

        # Make sure we have at least one solution
        if best_solution is None and population:
            best_solution = population[0]
            best_fitness = self.fitness(best_solution)

        # Handle case where no solution was found (should be extremely rare)
        if best_solution is None:
            best_solution = self.generate_individual()
            best_fitness = self.fitness(best_solution)

        final_solution = self.byte_to_number(best_solution)
        if verbose:
            meeting_point = self.graph.G.nodes.get(final_solution[0], {"x": "Unknown", "y": "Unknown"})
            dropping_point = self.graph.G.nodes.get(final_solution[1], {"x": "Unknown", "y": "Unknown"})
            print("Final Solution:")
            print(f"Meeting point: ({meeting_point['x']}, {meeting_point['y']})")
            print(f"Dropping point: ({dropping_point['x']}, {dropping_point['y']})")
            print(f"Final fitness: {best_fitness:.4f} \n\n")

        return {
            'solution': final_solution,
            'fitness': best_fitness,
            'total_distance': -best_fitness,
            'population_history': population_history,
            'fitness_history': fitness_history
        }