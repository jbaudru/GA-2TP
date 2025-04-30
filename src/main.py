import argparse
import time
import matplotlib.pyplot as plt

from models.graph import TransportationGraph
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.exact_algorithm import ExactAlgorithm
from utils.visualization import plot_fitness_progress, plot_algorithm_comparison
from evaluation.benchmarking import Benchmarker

def run_ga_experiment(graph_size=100, pop_size=20, steps=500):
    """Run a basic GA experiment"""
    # Create graph
    graph = TransportationGraph(graph_size)
    
    # Generate random terminals
    terminals = graph.generate_random_terminals()
    agent1_start, agent2_start, agent1_dest, agent2_dest = terminals
    
    print(f"Graph generated with {graph.N} nodes")
    print(f"Agent 1: start={agent1_start}, destination={agent1_dest}")
    print(f"Agent 2: start={agent2_start}, destination={agent2_dest}")
    
    # Run GA
    ga = GeneticAlgorithm(graph, pop_size=pop_size)
    ga.set_terminals(*terminals)
    
    result = ga.run(steps=steps)
    
    # Plot results
    plot_fitness_progress(result['fitness_history'], pop_size, steps)
    
    return result

def benchmark_algorithms(num_instances=10, graph_size=50):
    """Benchmark GA vs exact algorithm"""
    benchmarker = Benchmarker(
        TransportationGraph,
        GeneticAlgorithm,
        ExactAlgorithm
    )
    
    print(f"Running comparison on {num_instances} instances with graph size {graph_size}")
    
    ga_results, exact_results = benchmarker.compare_algorithms(
        num_instances=num_instances,
        graph_size=graph_size
    )
    
    # Plot comparison
    plot_algorithm_comparison(ga_results, exact_results, graph_size, num_instances)

def time_budget_comparison(num_instances=10, graph_size=50):
    """Compare algorithms with equal time budgets"""
    benchmarker = Benchmarker(
        TransportationGraph,
        GeneticAlgorithm,
        ExactAlgorithm
    )
    
    print(f"Running time-budget comparison on {num_instances} instances with graph size {graph_size}")
    
    ga_results, exact_results, exact_times = benchmarker.compare_with_time_budget(
        num_instances=num_instances,
        graph_size=graph_size
    )
    
    # Plot comparison
    plot_algorithm_comparison(ga_results, exact_results, graph_size, num_instances)

def road_network_experiment(filename):
    """Run experiment on real road network"""
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
    ga = GeneticAlgorithm(graph)
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
    parser.add_argument("-s", "--size", type=int, default=100, help="Graph size")
    parser.add_argument("-p", "--population", type=int, default=20, help="Population size")
    parser.add_argument("-g", "--generations", type=int, default=500, help="Number of generations")
    parser.add_argument("-i", "--instances", type=int, default=10, help="Number of benchmark instances")
    parser.add_argument("-f", "--file", type=str, default="MON.json", help="Road network file")
    
    args = parser.parse_args()
    
    if args.mode == "ga":
        run_ga_experiment(args.size, args.population, args.generations)
    elif args.mode == "benchmark":
        benchmark_algorithms(args.instances, args.size)
    elif args.mode == "timebudget":
        time_budget_comparison(args.instances, args.size)
    elif args.mode == "roadnetwork":
        road_network_experiment(args.file)