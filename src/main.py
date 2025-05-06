import argparse
import time
import matplotlib.pyplot as plt
import random
import numpy as np
from tqdm import tqdm

from models.graph import TransportationGraph
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.exact_algorithm import ExactAlgorithm
from utils.visualization import plot_fitness_progress, plot_algorithm_comparison, plot_mse_comparison
from evaluation.benchmarking import Benchmarker

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)

def run_ga_experiment(graph_size=100, pop_size=20, steps=500, seed=None, num_experiments=50):
    """Run multiple GA experiments and return average results"""
    # Set seed if provided
    if seed is not None:
        set_seed(seed)
        
    all_fitness_history = []
    all_total_distances = []
    all_execution_times = []
    
    print(f"Running {num_experiments} experiments with graph size {graph_size}, population {pop_size}, steps {steps}")
    
    # Create a separate random generator for terminals with a derived but deterministic seed
    terminal_rng = random.Random()
    
    # If main seed is provided, derive terminal seed from it (adding a large prime number)
    terminal_seed = (seed + 104729) 
    terminal_rng.seed(terminal_seed)
    
    
    for exp_idx in tqdm(range(num_experiments)):
        # Create graph
        graph = TransportationGraph(graph_size)
        
        # Generate random terminals
        terminals = graph.generate_random_terminals(rng=terminal_rng)
        #agent1_start, agent2_start, agent1_dest, agent2_dest = terminals
        
        # Run GA
        ga = GeneticAlgorithm(graph, pop_size=pop_size, seed=seed)
        ga.set_terminals(*terminals)
        
        start_time = time.time()
        result = ga.run(steps=steps, verbose=False)
        execution_time = time.time() - start_time
        
        all_fitness_history.append(result['fitness_history'])
        all_total_distances.append(result['total_distance'])
        all_execution_times.append(execution_time)
    
    # Calculate average fitness history
    # Find minimum length across all histories
    min_len = min(len(history) for history in all_fitness_history)
    
    # Create a consolidated fitness history where each generation has all experiments' data
    consolidated_fitness_history = []
    for gen_idx in range(min_len):
        # Flatten all population fitness values for this generation across all experiments
        gen_fitness = []
        for exp_idx in range(num_experiments):
            gen_fitness.extend(all_fitness_history[exp_idx][gen_idx])
        consolidated_fitness_history.append(gen_fitness)
    
    # Calculate average total distance
    avg_total_distance = np.mean(all_total_distances)
    std_total_distance = np.std(all_total_distances)
    avg_execution_time = np.mean(all_execution_times)
    
    print(f"\nResults averaged over {num_experiments} experiments:")
    print(f"Average total distance: {avg_total_distance:.2f} ± {std_total_distance:.2f}")
    print(f"Average execution time: {avg_execution_time:.3f}s")
    
    # Plot average results
    plot_fitness_progress(consolidated_fitness_history, pop_size, steps)
    
    # Return aggregated results
    return {
        'fitness_history': consolidated_fitness_history,
        'total_distance': avg_total_distance,
        'std_total_distance': std_total_distance,
        'execution_time': avg_execution_time,
        'all_distances': all_total_distances
    }

def benchmark_algorithms(num_instances=50, graph_size=50, seed=None, pop_size=None, steps=None):
    """Benchmark GA vs exact algorithm"""
    # Set seed if provided
    if seed is not None:
        set_seed(seed)
        
    benchmarker = Benchmarker(
        TransportationGraph,
        GeneticAlgorithm,
        ExactAlgorithm,
        seed=seed
    )
    
    print(f"Running comparison on {num_instances} instances with graph size {graph_size}")
    
    # num_instances=50, graph_size=100, pop_size=20, steps=250
    
    ga_results, exact_results = benchmarker.compare_algorithms(
        num_instances=num_instances,
        graph_size=graph_size,
        pop_size=pop_size,
        steps=steps,
    )
    
    # Plot comparison
    #plot_algorithm_comparison(ga_results, exact_results, graph_size, num_instances)
    str_para_file = f"V{graph_size}_P{pop_size}_T{steps}"
    plot_mse_comparison(ga_results, exact_results, graph_size, num_instances, str_para_file)


def time_budget_comparison(num_instances=50, graph_size=50, seed=None, pop_size=None):
    """Compare algorithms with equal time budgets"""
    # Set seed if provided
    if seed is not None:
        set_seed(seed)
        
    benchmarker = Benchmarker(
        TransportationGraph,
        GeneticAlgorithm,
        ExactAlgorithm,
        seed=seed
    )
    
    print(f"Running time-budget comparison on {num_instances} instances with graph size {graph_size}")
    
    ga_results, exact_results, exact_times = benchmarker.compare_with_time_budget(
        num_instances=num_instances,
        graph_size=graph_size,
        pop_size=pop_size,
    )
    
    print(f"Exact algorithm average time: {np.mean(exact_times):.3f}s")
    
    str_para_file = f"timebudget_V{graph_size}_P{pop_size}"
    plot_mse_comparison(ga_results, exact_results, graph_size, num_instances, str_para_file)
    

def road_network_experiment(filename, seed=None):
    """Run experiment on real road network"""
    # Set seed if provided
    if seed is not None:
        set_seed(seed)
        
    import os
    filepath = os.path.join("data", "benchmark", filename)
    
    # Load graph
    graph = TransportationGraph(0).load_from_json(filepath)
    
    print(f"Road network loaded with {graph.G.number_of_nodes()} nodes and {graph.G.number_of_edges()} edges")
    
    # Generate random terminals
    terminals = graph.generate_random_terminals()
    agent1_start, agent2_start, agent1_dest, agent2_dest = terminals
    
    print(f"Agent 1: start={agent1_start}, destination={agent1_dest}")
    print(f"Agent 2: start={agent2_start}, destination={agent2_dest}")
    
    # Run GA with step limit
    ga = GeneticAlgorithm(graph, pop_size=10, seed=seed)
    ga.set_terminals(*terminals)
    
    start_time = time.time()
    ga_result = ga.run(steps=500)
    ga_time = time.time() - start_time
    
    # Run exact algorithm
    exact_solver = ExactAlgorithm(graph)
    start_time = time.time()
    exact_result = exact_solver.solve(*terminals)
    exact_time = time.time() - start_time
    
    print(f"Exact Algorithm: {exact_result['total_distance']:.2f} ({exact_time:.3f}s)")
    print(f"Genetic Algorithm: {ga_result['total_distance']:.2f} ({ga_time:.3f}s)")
    
    return ga_result, exact_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-Terminal Problem Solver")
    parser.add_argument("-m", "--mode", choices=["ga", "benchmark", "timebudget", "roadnetwork"], 
                        default="ga", help="Mode to run")
    parser.add_argument("-s", "--size", type=int, default=400, help="Graph size")
    parser.add_argument("-p", "--population", type=int, default=50, help="Population size")
    parser.add_argument("-g", "--generations", type=int, default=1000, help="Number of generations")
    parser.add_argument("-i", "--instances", type=int, default=50, help="Number of benchmark instances")
    parser.add_argument("-f", "--file", type=str, default="MON.json", help="Road network file")
    parser.add_argument("--seed", type=int, default=66, help="Random seed for reproducibility")
    
    args = parser.parse_args()

    
    if args.mode == "ga":
        run_ga_experiment(args.size, args.population, args.generations, args.seed, 50)
    elif args.mode == "benchmark":
        benchmark_algorithms(args.instances, args.size, args.seed, args.population, args.generations)
    elif args.mode == "timebudget":
        time_budget_comparison(args.instances, args.size, args.seed, args.population)
    elif args.mode == "roadnetwork":
        road_network_experiment(args.file, args.seed)