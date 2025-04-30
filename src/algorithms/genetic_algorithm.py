import random
import time
from tqdm import tqdm
import numpy as np

class GeneticAlgorithm:
    """Genetic algorithm implementation for the 2-Terminal Problem"""
    
    def __init__(self, graph, pop_size=20, mutation_rate=0.1, encode_len=8):
        """Initialize the GA with the problem parameters"""
        self.graph = graph
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.encode_len = encode_len
        self.bit_count = 2 * encode_len  # For meeting and dropping points
        self.terminals = None
    
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
        """Convert bit representation to node IDs"""
        numbers = []
        bits_per_var = self.bit_count // 2
        max_int = (1 << bits_per_var) - 1
        
        for i in range(2):  # Two variables: meeting point and dropping point
            number = 0
            for _ in range(bits_per_var):
                number = (number << 1) | (bit_data & 1)
                bit_data >>= 1
            node_id = number % self.graph.N
            numbers.append(node_id)
        
        return numbers[::-1]
    
    def fitness(self, individual):
        """Calculate fitness of an individual"""
        if not self.terminals:
            raise ValueError("Terminal nodes not set")
            
        agent1_start, agent2_start, agent1_dest, agent2_dest = self.terminals
        points = self.byte_to_number(individual)
        meeting_point, dropping_point = points
        
        if meeting_point == dropping_point:
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
            return float('-inf')
        
        return -total  # Negative since GA maximizes fitness
    
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
    
    def evolve(self, population):
        """Create a new individual through evolution"""
        p1 = self.tournament_selection(population)
        p2 = self.tournament_selection(population)
        c = self.crossover(p1, p2)
        c = self.mutation(c)
        return c
    
    def run(self, steps=500, verbose=True):
        """Run the genetic algorithm for a specified number of steps"""
        if not self.terminals:
            raise ValueError("Terminal nodes not set")
            
        population = self.populate()
        population_history = []
        fitness_history = []
        best_fitness = float('-inf')
        best_solution = None
        
        for i in range(steps):
            population_history.append(population[:])
            current_fitness = [self.fitness(ind) for ind in population]
            fitness_history.append(current_fitness)
            
            population.append(self.evolve(population))
            population = sorted(population, key=self.fitness, reverse=True)[:self.pop_size]
            
            top_fit = self.fitness(population[0])
            if top_fit > best_fitness:
                best_fitness = top_fit
                best_solution = population[0]
                if verbose:
                    points = self.byte_to_number(population[0])
                    print(f"Step: {i} Fitness: {top_fit:.4f}")
                    print(f"Meeting point: {points[0]}, Dropping point: {points[1]}")
        
        final_solution = self.byte_to_number(best_solution)
        if verbose:
            print("\nFinal Solution:")
            print(f"Meeting point: {final_solution[0]}")
            print(f"Dropping point: {final_solution[1]}")
            print(f"Final fitness: {best_fitness:.4f}")
        
        return {
            'solution': final_solution,
            'fitness': best_fitness,
            'total_distance': -best_fitness,
            'population_history': population_history,
            'fitness_history': fitness_history
        }
    
    def run_with_time_budget(self, time_budget, verbose=False):
        """Run the GA with a time budget"""
        if not self.terminals:
            raise ValueError("Terminal nodes not set")
            
        population = self.populate()
        best_fitness = float('-inf')
        best_solution = None
        
        start_time = time.time()
        iterations = 0
        
        while time.time() - start_time < time_budget:
            iterations += 1
            population.append(self.evolve(population))
            population = sorted(population, key=self.fitness, reverse=True)[:self.pop_size]
            
            top_fit = self.fitness(population[0])
            if top_fit > best_fitness:
                best_fitness = top_fit
                best_solution = population[0]
        
        if verbose:
            print(f"Completed {iterations} iterations in {time_budget:.2f}s")
            
        final_solution = self.byte_to_number(best_solution)
        return {
            'solution': final_solution,
            'fitness': best_fitness,
            'total_distance': -best_fitness,
            'iterations': iterations
        }