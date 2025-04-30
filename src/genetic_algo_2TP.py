import random
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import seaborn as sns
from scipy import stats
import pandas as pd
import time
import json
from tqdm import tqdm
from functools import lru_cache

# Graph parameters
N = 100  # number of nodes
number_of_variables = 2  # meeting point and dropping point
G = nx.complete_graph(N)
# Assign random weights
for (u, v) in G.edges():
    G[u][v]['weight'] = random.uniform(1, 10)


# Add random start positions for agents
agent1_start = random.randint(0, N-1)
agent2_start = random.randint(0, N-1)
while agent2_start == agent1_start:  # ensure different start positions
    agent2_start = random.randint(0, N-1)
agent1_dest = random.randint(0, N-1)
# After agent start positions, add destinations
while agent1_dest == agent1_start:
    agent1_dest = random.randint(0, N-1)

agent2_dest = random.randint(0, N-1)
while agent2_dest == agent2_start or agent2_dest == agent1_dest:
    agent2_dest = random.randint(0, N-1)


pop_size = 20 #min(40, max(10, N))  # 1 individuals per node, capped at 50
steps = 500 #min(2000, max(500, N * 20))  # 100 generations per node, capped at 10000


encode_len = 8  # reduced since we only need to encode node IDs
bit_count = number_of_variables * encode_len

def generate_individual(count=bit_count):
    bits = 0
    for _ in range(count):
        bits = (bits << 1) | random.getrandbits(1)
    return bits

def byte_to_number(bit_data, count=bit_count):
    numbers = []
    bits_per_var = count // number_of_variables
    max_int = (1 << bits_per_var) - 1
    
    for i in range(number_of_variables):
        number = 0
        for _ in range(bits_per_var):
            number = (number << 1) | (bit_data & 1)
            bit_data >>= 1
        # Map to valid node ID
        node_id = number % N
        numbers.append(node_id)
    
    return numbers[::-1]

def get_path_distance(start, end):
    if start == end:
        return 0
    return nx.shortest_path_length(G, start, end, weight='weight')


def fitness(individual):
    points = byte_to_number(individual)
    meeting_point, dropping_point = points
    
    if meeting_point == dropping_point:
        return float('-inf')  # penalize same points
    
    # Calculate using same method as exact algorithm
    try:
        sum_m = nx.shortest_path_length(G, agent1_start, meeting_point, weight='weight') + \
                nx.shortest_path_length(G, agent2_start, meeting_point, weight='weight')
        
        sum_k = nx.shortest_path_length(G, meeting_point, dropping_point, weight='weight')
        
        sum_e = nx.shortest_path_length(G, dropping_point, agent1_dest, weight='weight') + \
                nx.shortest_path_length(G, dropping_point, agent2_dest, weight='weight')
        
        total = sum_m + sum_k + sum_e
    except:
        total = float('-inf')
    
    return -total  # Negative since GA maximizes fitness


def populate(population_size):
    return [generate_individual() for _ in range(population_size)]

def selection(population):
    return sorted(population, key=fitness, reverse=True)[:2]

def custom_selection(population):
    return sorted(population, key=fitness, reverse=True)[0], random.choice(population[1:])

def tournament_selection(population, k=pop_size // 10):
    contenders = random.sample(population, k)
    return max(contenders, key=fitness)


def crossover(parent1, parent2):
    child = 0
    for i in range(bit_count):
        pick = (parent1 if random.random() < 0.5 else parent2)
        bit = (pick >> i) & 1
        child |= (bit << i)
    return child

def mutation(individual):
    for i in range(bit_count):
        if random.random() < 0.1:
            individual ^= (1 << i)
    return individual

def evolve(population):
    #p1, p2 = selection(population)
    #p1, p2 = custom_selection(population)
    p1, p2 = tournament_selection(population), tournament_selection(population)
    #p1, p2 = tournament_selection(population, pop_size//2), tournament_selection(population, pop_size//2)
    c = crossover(p1, p2)
    c = mutation(c)
    return c

def plot_fitness_progress(population_history, fitness_history, pop_size=None, steps=None):
    plt.figure(figsize=(8,6))
    
    # Convert negative fitness to positive distances
    best_fitness = [-min(gen_fitness) for gen_fitness in fitness_history]
    avg_fitness = [-np.mean(gen_fitness) for gen_fitness in fitness_history]
    worst_fitness = [-max(gen_fitness) for gen_fitness in fitness_history]
    std_fitness = [np.std(gen_fitness) for gen_fitness in fitness_history]
    
    generations = range(len(fitness_history))
    
    plt.fill_between(generations, 
                     np.array(avg_fitness) - np.array(std_fitness),
                     np.array(avg_fitness) + np.array(std_fitness),
                     alpha=0.2, color='gray', label='Standard Deviation')
    
    plt.plot(worst_fitness, label='Best Distance', color='green', linewidth=1)
    plt.plot(avg_fitness, label='Average Distance', color='blue', linewidth=1)
    plt.plot(best_fitness, label='Worst Distance', color='red', linewidth=1)
    
    plt.xlabel('Generation')
    plt.ylabel('Total Distance')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    name = "images/fitness_progress_N" + str(N) + "_s" + str(steps) + "_P" + str(pop_size) + ".png"
    plt.savefig(name, dpi=300)
    plt.show()


@lru_cache
def exact_algo(a, b, c, d):
    global G
    min_sum = float('inf')
    best_m = None
    best_k = None
    
    for m in tqdm(G.nodes()):
        for k in G.nodes():
            if m != k:
                try:
                    # Calculate path from terminals to first point m
                    sum_m = nx.shortest_path_length(G, a, m, weight='length') + \
                        nx.shortest_path_length(G, b, m, weight='length')
                    
                    # Calculate path between intermediate points
                    sum_k = nx.shortest_path_length(G, m, k, weight='length')
                    
                    # Calculate path from second point k to terminals
                    sum_e = nx.shortest_path_length(G, k, c, weight='length') + \
                        nx.shortest_path_length(G, k, d, weight='length')
                    
                    total = sum_m + sum_k + sum_e
                    
                except nx.NodeNotFound:
                    # Skip this combination if any node is not found
                    continue
                except nx.NetworkXNoPath:
                    # Skip if no path exists between nodes
                    continue
                
                if total < min_sum:
                    min_sum = total
                    best_m = m
                    best_k = k
                    
    return min_sum, best_m, best_k


def run_ga_instance(a, b, c, d):
    global agent1_start, agent2_start, agent1_dest, agent2_dest, G
    agent1_start, agent2_start = a, b
    agent1_dest, agent2_dest = c, d
    
    population = populate(pop_size)
    best_fitness = float('-inf')
    best_solution = None
    
    for _ in tqdm(range(steps)):
        population.append(evolve(population))
        population = sorted(population, key=fitness, reverse=True)[:pop_size]
        top_fit = fitness(population[0])
        if top_fit > best_fitness:
            best_fitness = top_fit
            best_solution = population[0]
    
    # Validate solution
    points = byte_to_number(best_solution)
    m, k = points
    
    # Calculate distance using same path computation as exact algorithm
    try:
        sum_m = nx.shortest_path_length(G, a, m, weight='length') + \
                nx.shortest_path_length(G, b, m, weight='length')
        sum_k = nx.shortest_path_length(G, m, k, weight='length')
        sum_e = nx.shortest_path_length(G, k, c, weight='length') + \
                nx.shortest_path_length(G, k, d, weight='length')
        total = sum_m + sum_k + sum_e
    except:
        total = float('-inf')
    
    return total

def run_ga_instance_time(a, b, c, d, time_budget):
    global agent1_start, agent2_start, agent1_dest, agent2_dest, G
    agent1_start, agent2_start = a, b
    agent1_dest, agent2_dest = c, d
    
    population = populate(pop_size)
    best_fitness = float('-inf')
    best_solution = None
    
    start_time = time.time()
    while time.time() - start_time < time_budget:
        population.append(evolve(population))
        population = sorted(population, key=fitness, reverse=True)[:pop_size]
        top_fit = fitness(population[0])
        if top_fit > best_fitness:
            best_fitness = top_fit
            best_solution = population[0]
    
    # Validate solution
    points = byte_to_number(best_solution)
    m, k = points
    
    # Calculate distance using same path computation as exact algorithm
    try:
        sum_m = nx.shortest_path_length(G, a, m, weight='length') + \
                nx.shortest_path_length(G, b, m, weight='length')
        sum_k = nx.shortest_path_length(G, m, k, weight='length')
        sum_e = nx.shortest_path_length(G, k, c, weight='length') + \
                nx.shortest_path_length(G, k, d, weight='length')
        total = sum_m + sum_k + sum_e
    except:
        total = float('-inf')
    return total


def raincloud_plot(data, labels, ax):
    # Define colors matching seaborn default palette
    colors = sns.color_palette()[:len(labels)]
    
    # Convert data to DataFrame
    df = pd.DataFrame()
    for i, (d, label) in enumerate(zip(data, labels)):
        temp_df = pd.DataFrame({
            'Distance': d,
            'Algorithm': [label] * len(d)
        })
        df = pd.concat([df, temp_df])
    
    # Kernel Density Estimate plot
    sns.kdeplot(data=df, x='Distance', hue='Algorithm', fill=True, alpha=0.5, ax=ax)
    
    # Box plot with light grey color
    sns.boxplot(data=df, x='Algorithm', y='Distance', width=0.4, 
                showfliers=False, ax=ax, color='#E8E8E8', saturation=0.5, linewidth=1.5)
    
    # Strip plot with matching colors
    for i, label in enumerate(labels):
        mask = df['Algorithm'] == label
        sns.stripplot(data=df[mask], x='Algorithm', y='Distance', 
                     size=7, alpha=0.5, color=colors[i], ax=ax)
    
    return ax


def eval_error(num_instances=50):
    global G
    ga_results = []
    exact_results = []
    
    for i in range(num_instances):
        
        G = nx.complete_graph(N)
        for (u, v) in G.edges():
            G[u][v]['length'] = random.uniform(1, 10)
            
        # Generate random instance
        a = random.randint(0, N-1)
        b = random.randint(0, N-1)
        while b == a:
            b = random.randint(0, N-1)
            
        c = random.randint(0, N-1)
        while c == a:
            c = random.randint(0, N-1)
            
        d = random.randint(0, N-1)
        while d in [a, b, c]:
            d = random.randint(0, N-1)
            
        # Run both algorithms
        exact_dist, _, _ = exact_algo(a, b, c, d)
        ga_dist = run_ga_instance(a, b, c, d)
        
        if ga_dist < exact_dist:
            print(f"WARNING: GA found better solution than exact algorithm!")
            print(f"GA distance: {ga_dist}")
            print(f"Exact distance: {exact_dist}")
            print(f"Instance details: a={a}, b={b}, c={c}, d={d}")
            # This should never happen - indicates a bug
        
        ga_results.append(ga_dist)
        exact_results.append(exact_dist)
        
        print(f"Instance {i+1}: Exact={exact_dist:.2f}, GA={ga_dist:.2f}")
    
    
    # Plot results
    plt.figure(figsize=(8,6))
    ax = plt.gca()
    raincloud_plot([ga_results, exact_results], ['GA', 'Exact'], ax)
    
    plt.ylabel('Total Distance')
    plt.xlabel('Algorithm')
    #plt.title(f'Distance Comparison (N={N}, {num_instances} instances)')
    plt.grid(True)
    
    # Calculate and display error statistics
    errors = np.array(ga_results) - np.array(exact_results)
    rel_errors = errors / np.array(exact_results) * 100
    
    print("\nError Statistics:")
    print(f"Mean Relative Error: {np.mean(rel_errors):.2f}%")
    print(f"Std Relative Error: {np.std(rel_errors):.2f}%")
    print(f"Max Relative Error: {np.max(rel_errors):.2f}%")
    
    plt.savefig(f'images/comparison_N{N}_I{num_instances}.png', dpi=300)
    plt.show()
    
    return ga_results, exact_results


def eval_error_with_time_budget(num_instances=50):
    global G
    ga_results = []
    exact_results = []
    exact_times = []
    
    for i in range(num_instances):
        # Generate problem instance
        G = nx.complete_graph(N)
        for (u, v) in G.edges():
            G[u][v]['length'] = random.uniform(1, 10)
            
        # Generate random vertices
        a = random.randint(0, N-1)
        b = random.randint(0, N-1)
        while b == a:
            b = random.randint(0, N-1)
            
        c = random.randint(0, N-1)
        while c == a:
            c = random.randint(0, N-1)
            
        d = random.randint(0, N-1)
        while d in [a, b, c]:
            d = random.randint(0, N-1)
            
        # Time the exact algorithm
        start_time = time.time()
        exact_dist, _, _ = exact_algo(a, b, c, d)
        exact_time = time.time() - start_time
        exact_times.append(exact_time)
        
        best_ga_dist = run_ga_instance_time(a, b, c, d, exact_time)
            
        if best_ga_dist < exact_dist:
            print(f"WARNING: GA found better solution than exact algorithm!")
            print(f"GA distance: {best_ga_dist}")
            print(f"Exact distance: {exact_dist}")
            print(f"Instance details: a={a}, b={b}, c={c}, d={d}")
        
        ga_results.append(best_ga_dist)
        exact_results.append(exact_dist)
        
        print(f"Instance {i+1}: Exact={exact_dist:.2f} ({exact_time:.3f}s), GA={best_ga_dist:.2f}")
    
    # Plot results
    plt.figure(figsize=(8,6))
    ax = plt.gca()
    raincloud_plot([ga_results, exact_results], ['GA', 'Exact'], ax)
    
    plt.ylabel('Total Distance')
    plt.xlabel('Algorithm')
    plt.grid(True)
    
    # Calculate and display error statistics
    errors = np.array(ga_results) - np.array(exact_results)
    rel_errors = errors / np.array(exact_results) * 100
    
    print("\nError Statistics:")
    print(f"Mean Relative Error: {np.mean(rel_errors):.2f}%")
    print(f"Std Relative Error: {np.std(rel_errors):.2f}%")
    print(f"Max Relative Error: {np.max(rel_errors):.2f}%")
    print(f"Average Exact Algorithm Time: {np.mean(exact_times):.3f}s")
    
    plt.savefig(f'images/time_budget_comparison_N{N}_I{num_instances}.png', dpi=300)
    plt.show()
    
    return ga_results, exact_results


def main(pop_size=None, steps=None):
    global G
    
    print("Simulation parameters:")
    print(f"Population size: {pop_size}")
    print(f"Number of steps: {steps}")
    print("=====================================")
    
    population = populate(pop_size)
    population_history = []
    fitness_history = []
    best_fitness = fitness(population[0])
    
    
    print(f"Graph generated with {N} nodes")
    print(f"Agent 1: start={agent1_start}, destination={agent1_dest}")
    print(f"Agent 2: start={agent2_start}, destination={agent2_dest}")
    
    #stagnation_limit = min(500, max(50, N * 10))  # Scale with graph size  # Stop if no improvement for 100 generations
    #stagnation_counter = 0
    
    for i in range(steps):
        population_history.append(population[:])
        current_fitness = [fitness(ind) for ind in population]
        fitness_history.append(current_fitness)
        
        population.append(evolve(population))
        population = sorted(population, key=fitness, reverse=True)[:pop_size]
        top_fit = fitness(population[0])
        
        if top_fit > best_fitness:
            points = byte_to_number(population[0])
            print(f"Step: {i} Fitness: {top_fit:.4f}")
            print(f"Meeting point: {points[0]}, Dropping point: {points[1]}")
            best_fitness = top_fit
            #stagnation_counter = 0
        """
        else:
            stagnation_counter += 1
            if stagnation_counter >= stagnation_limit:
                print(f"Early stopping at generation {i} due to stagnation")
                break
        """
        
    # Final result
    best_solution = byte_to_number(population[0])
    print("\nFinal Solution:")
    print(f"Meeting point: {best_solution[0]}")
    print(f"Dropping point: {best_solution[1]}")
    print(f"Final fitness: {fitness(population[0]):.4f}")
    print("-------------------------------------")
    plot_fitness_progress(population_history, fitness_history, pop_size, steps)


def eval_runtime_on_real_road_network():
    global G
    # Load road network data from JSON
    with open('../data/benchmark/MON.json', 'r') as f:
        graph_data = json.load(f)
    G = nx.node_link_graph(graph_data)
    
    # Generate random start and destination points
    a = random.choice(list(G.nodes()))
    b = random.choice(list(G.nodes()))
    while b == a:
        b = random.choice(list(G.nodes()))
    
    c = random.choice(list(G.nodes()))
    while c in [a, b]:
        c = random.choice(list(G.nodes()))
    
    d = random.choice(list(G.nodes()))
    while d in [a, b, c]:
        d = random.choice(list(G.nodes()))
    
    print(f"Road network loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"Agent 1: start={a}, destination={b}")
    print(f"Agent 2: start={c}, destination={d}")
    
    # Run GA with step limit
    start_time = time.time()
    ga_dist = run_ga_instance(a, b, c, d)
    ga_time = time.time() - start_time
    
    # Time the exact algorithm
    start_time = time.time()
    exact_dist, _, _ = exact_algo(a, b, c, d)
    exact_time = time.time() - start_time
    
    
    
    print(f"Exact Algorithm: {exact_dist:.2f} ({exact_time:.3f}s)")
    print(f"Genetic Algorithm: {ga_dist:.2f} ({ga_time:.3f}s)")
    
    
    return exact_dist, ga_dist


if __name__ == "__main__":
    """
    main(10, 250)
    main(20, 250)
    main(10, 500)
    main(20, 500)
    eval_error()
    """
    #eval_error_with_time_budget()
    eval_runtime_on_real_road_network()

# P=10

# V=10, Average Exact Algorithm Time: 0.007s, Mean Relative Error: 22.80%
# V=25, Average Exact Algorithm Time: 0.234s, Mean Relative Error: 18.39%
# V=50, Average Exact Algorithm Time: 3.886s, Mean Relative Error: 8.01%
# V=100, Average Exact Algorithm Time: 62.584s, Mean Relative Error: 7.20%

# P=20
# V=10, Average Exact Algorithm Time: 0.006s, Mean Relative Error:  17.08%
# V=25, Average Exact Algorithm Time: 0.235s, Mean Relative Error: 25.67%
# V=50, Average Exact Algorithm Time: 3.795s , Mean Relative Error: 15.87%
# V=100, Average Exact Algorithm Time: 61.904s, Mean Relative Error: 9.28%


# Real road network
#V = 2770, E = 6407 (TOU.json)
# GA: 01min:03sec
# Exact: 28h:33min:41sec

# (FRA.json)
# V = 2390, E = 5296
# GA: 00:52sec
# Exact: 11:08:53sec

# (MON.json)
# V = 3002, E = 6620
# GA : 01:07sec
# Exact : 38:48:09