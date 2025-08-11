import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
import random
import sys
import os
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.graph import TransportationGraph
from algorithms.genetic_algorithm import GeneticAlgorithm

class VideoGenerator:
    def __init__(self, output_folder="../video_output"):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        
        # Set up visual style
        plt.style.use('dark_background')
        self.colors = {
            'primary': '#00ff88',
            'secondary': '#ff6b6b',
            'accent': '#4ecdc4',
            'background': '#1a1a1a',
            'text': '#ffffff',
            'meeting': '#ffd700',
            'dropping': '#ff8c00'
        }
    
    def create_problem_intro_animation(self, num_nodes=20):
        """Create animation showing the 2TP problem definition"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.patch.set_facecolor(self.colors['background'])
        fig.suptitle('The 2-Terminal Problem (2TP)', fontsize=20, color=self.colors['text'])
        
        # Create a simple graph for demonstration
        graph = TransportationGraph(num_nodes)
        G = graph.G
        pos = nx.spring_layout(G, seed=123)  # Different seed for different layout
        
        # Generate terminals (ensure all 4 are different) - use different seed for different terminals
        terminals = graph.generate_random_terminals(rng=random.Random(123))
        agent1_start, agent2_start, agent1_dest, agent2_dest = terminals
        
        # Generate example meeting and dropping points (different from terminals)
        # Choose nodes that create good visible paths
        available_nodes = [n for n in G.nodes() if n not in terminals]
        
        # Try to find meeting and dropping points that create connected paths
        best_meeting = None
        best_dropping = None
        min_total_distance = float('inf')
        
        # Test several combinations to find paths with no overlapping nodes
        for _ in range(min(50, len(available_nodes) * (len(available_nodes) - 1))):
            if len(available_nodes) < 2:
                break
            test_nodes = random.Random(123 + _).sample(available_nodes, 2)  # Use seed 123 for consistency
            test_meeting, test_dropping = test_nodes
            
            try:
                # Get all paths for this combination
                path1_to_m = nx.shortest_path(G, agent1_start, test_meeting, weight='weight')
                path2_to_m = nx.shortest_path(G, agent2_start, test_meeting, weight='weight')
                path_m_to_d = nx.shortest_path(G, test_meeting, test_dropping, weight='weight')
                path_d_to_1 = nx.shortest_path(G, test_dropping, agent1_dest, weight='weight')
                path_d_to_2 = nx.shortest_path(G, test_dropping, agent2_dest, weight='weight')
                
                # Check for overlapping nodes (excluding terminals and meeting/dropping points)
                all_paths = [path1_to_m, path2_to_m, path_m_to_d, path_d_to_1, path_d_to_2]
                path_nodes = [set(path[1:-1]) for path in all_paths]  # Exclude start/end nodes
                
                # Check if any intermediate nodes overlap between different agent paths
                overlap_found = False
                
                # Check A1 vs A2 paths to meeting point
                if path_nodes[0] & path_nodes[1]:  # A1→M vs A2→M
                    overlap_found = True
                
                # Check A1 vs A2 paths from dropping point
                if path_nodes[3] & path_nodes[4]:  # D→A1 vs D→A2
                    overlap_found = True
                
                # If no overlaps in critical paths, this is a good candidate
                if not overlap_found:
                    # Calculate total distance
                    total_dist = sum(len(path) - 1 for path in all_paths)  # Use path length as distance
                    
                    if total_dist < min_total_distance:
                        min_total_distance = total_dist
                        best_meeting = test_meeting
                        best_dropping = test_dropping
            except:
                continue
        
        # Fallback if no good combination found (try to ensure visual clarity)
        if best_meeting is None or best_dropping is None:
            # Use nodes that are well-separated for better visualization
            if len(available_nodes) >= 2:
                example_meeting = available_nodes[len(available_nodes)//4]
                example_dropping = available_nodes[3*len(available_nodes)//4]
            else:
                example_meeting = available_nodes[0] if available_nodes else list(G.nodes())[0]
                example_dropping = available_nodes[1] if len(available_nodes) > 1 else list(G.nodes())[1]
        else:
            example_meeting = best_meeting
            example_dropping = best_dropping
        
        def animate_intro(frame):
            ax1.clear()
            ax2.clear()
            
            # Set background color for axes
            ax1.set_facecolor(self.colors['background'])
            ax2.set_facecolor(self.colors['background'])
            
            # Left panel: Graph visualization
            ax1.set_title('Transportation Network', fontsize=16, color=self.colors['text'])
            
            # Draw base network
            nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=self.colors['accent'], 
                                 node_size=200, alpha=0.6)
            
            if frame > 30:
                nx.draw_networkx_edges(G, pos, ax=ax1, edge_color=self.colors['text'], 
                                     alpha=0.3, width=1)
            
            # Highlight terminals progressively
            if frame > 60:
                # Agent 1 start and destination
                nx.draw_networkx_nodes(G, pos, nodelist=[agent1_start], ax=ax1,
                                     node_color=self.colors['primary'], 
                                     node_size=400, alpha=1.0, label='Agent 1 Start')
                nx.draw_networkx_nodes(G, pos, nodelist=[agent1_dest], ax=ax1,
                                     node_color=self.colors['primary'], 
                                     node_size=400, alpha=0.7, label='Agent 1 Dest')
            
            if frame > 90:
                # Agent 2 start and destination
                nx.draw_networkx_nodes(G, pos, nodelist=[agent2_start], ax=ax1,
                                     node_color=self.colors['secondary'], 
                                     node_size=400, alpha=1.0, label='Agent 2 Start')
                nx.draw_networkx_nodes(G, pos, nodelist=[agent2_dest], ax=ax1,
                                     node_color=self.colors['secondary'], 
                                     node_size=400, alpha=0.7, label='Agent 2 Dest')
            
            # Show example meeting and dropping points
            if frame > 120:
                # Meeting point in yellow
                nx.draw_networkx_nodes(G, pos, nodelist=[example_meeting], ax=ax1,
                                     node_color=self.colors['meeting'], 
                                     node_size=500, alpha=0.9)
                # Dropping point in orange
                nx.draw_networkx_nodes(G, pos, nodelist=[example_dropping], ax=ax1,
                                     node_color=self.colors['dropping'], 
                                     node_size=500, alpha=0.9)
            
            # Add labels to terminals
            if frame > 150:
                ax1.text(pos[agent1_start][0], pos[agent1_start][1]+0.1, 'A1 Start', 
                        ha='center', color=self.colors['primary'], fontweight='bold')
                ax1.text(pos[agent1_dest][0], pos[agent1_dest][1]+0.1, 'A1 End', 
                        ha='center', color=self.colors['primary'], fontweight='bold')
                ax1.text(pos[agent2_start][0], pos[agent2_start][1]+0.1, 'A2 Start', 
                        ha='center', color=self.colors['secondary'], fontweight='bold')
                ax1.text(pos[agent2_dest][0], pos[agent2_dest][1]+0.1, 'A2 End', 
                        ha='center', color=self.colors['secondary'], fontweight='bold')
                ax1.text(pos[example_meeting][0], pos[example_meeting][1]+0.1, 'Meeting (M)', 
                        ha='center', color=self.colors['meeting'], fontweight='bold')
                ax1.text(pos[example_dropping][0], pos[example_dropping][1]+0.1, 'Dropping (D)', 
                        ha='center', color=self.colors['dropping'], fontweight='bold')
            
            # Draw example solution paths with correct colors
            if frame > 180:
                try:
                    # Paths to meeting point
                    path1_to_m = nx.shortest_path(G, agent1_start, example_meeting, weight='weight')
                    path2_to_m = nx.shortest_path(G, agent2_start, example_meeting, weight='weight')
                    
                    # Path from meeting to dropping
                    path_m_to_d = nx.shortest_path(G, example_meeting, example_dropping, weight='weight')
                    
                    # Paths from dropping point
                    path_d_to_1 = nx.shortest_path(G, example_dropping, agent1_dest, weight='weight')
                    path_d_to_2 = nx.shortest_path(G, example_dropping, agent2_dest, weight='weight')
                    
                    # Draw path edges with correct colors
                    all_paths = [path1_to_m, path2_to_m, path_m_to_d, path_d_to_1, path_d_to_2]
                    path_colors = [self.colors['primary'], self.colors['secondary'], 
                                  self.colors['meeting'], self.colors['primary'], self.colors['secondary']]
                    
                    for path, color in zip(all_paths, path_colors):
                        if len(path) > 1:
                            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                            nx.draw_networkx_edges(G, pos, edgelist=path_edges, ax=ax1,
                                                 edge_color=color, width=4, alpha=0.8)
                except:
                    pass  # Skip if paths don't exist
            
            ax1.set_aspect('equal')
            ax1.axis('off')
            
            # Right panel: Problem explanation
            ax2.text(0.5, 0.9, '2-Terminal Problem', 
                    ha='center', va='center', fontsize=18, color=self.colors['text'], fontweight='bold')
            
            if frame > 30:
                ax2.text(0.5, 0.8, 'Two agents need to travel from', 
                        ha='center', va='center', fontsize=14, color=self.colors['text'])
                ax2.text(0.5, 0.75, 'their starting points to destinations', 
                        ha='center', va='center', fontsize=14, color=self.colors['text'])
            
            if frame > 60:
                ax2.text(0.5, 0.65, 'Solution: Find optimal meeting (M) & dropping (D) points', 
                        ha='center', va='center', fontsize=14, color=self.colors['meeting'])
            
            if frame > 90:
                ax2.text(0.5, 0.55, 'Minimize total travel distance:', 
                        ha='center', va='center', fontsize=14, color=self.colors['text'])
                ax2.text(0.5, 0.5, '(A1start→M + A2start→M) + (M→D) + (D→A1end + D→A2end)', 
                        ha='center', va='center', fontsize=12, color=self.colors['accent'])
            
            if frame > 120:
                ax2.text(0.5, 0.35, 'Using Genetic Algorithm:', 
                        ha='center', va='center', fontsize=14, color=self.colors['secondary'])
                ax2.text(0.5, 0.28, '• Encode meeting & dropping points as chromosomes', 
                        ha='center', va='center', fontsize=11, color=self.colors['text'])
                ax2.text(0.5, 0.23, '• Evolve population through selection & mutation', 
                        ha='center', va='center', fontsize=11, color=self.colors['text'])
                ax2.text(0.5, 0.18, '• Fitness = -total_distance (maximize)', 
                        ha='center', va='center', fontsize=11, color=self.colors['text'])
            
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')
        
        print("Creating problem introduction animation...")
        anim = animation.FuncAnimation(fig, animate_intro, frames=210, interval=80)  # More frames for path drawing
        output_path = self.output_folder / 'problem_introduction.mp4'
        anim.save(str(output_path), writer='ffmpeg', fps=20, dpi=120)  # Higher FPS and DPI
        plt.close()
        print(f"Saved: {output_path}")
        return anim
    
    def create_ga_evolution_animation(self, num_nodes=50, generations=2000, pop_size=20):
        """Animate the genetic algorithm evolution process"""
        print(f"Creating GA evolution animation with {generations} generations on {num_nodes} nodes...")
        
        # Create graph and set up GA
        graph = TransportationGraph(num_nodes)
        terminals = graph.generate_random_terminals()
        
        ga = GeneticAlgorithm(graph, pop_size=pop_size, seed=42)
        ga.set_terminals(*terminals)
        
        # Collect evolution data using EXACT same process as ga.run()
        population = ga.populate()
        population_history = []
        fitness_history = []
        evaluated_solutions = set()
        
        for i in tqdm(range(generations), desc="Running GA"):
            population_history.append(population[:])
            
            # Track unique solutions (exactly like in ga.run())
            for ind in population:
                evaluated_solutions.add(ind)
                
            current_fitness = [ga.fitness(ind) for ind in population]
            fitness_history.append(current_fitness)
            
            # Evolve population (exactly like in ga.run())
            if i < generations - 1:  # Don't evolve on last iteration
                # Generate new unique individual
                new_individual = ga.evolve(population)
                while new_individual in evaluated_solutions:
                    new_individual = ga.evolve(population)
                
                population.append(new_individual)
                population = sorted(population, key=ga.fitness, reverse=True)[:pop_size]
        
        # Convert evolution data for animation (convert negative fitness to positive distance)
        evolution_data = []
        for gen in range(len(fitness_history)):
            current_fitness = fitness_history[gen]
            best_idx = np.argmax(current_fitness)
            best_solution = population_history[gen][best_idx]
            
            # Convert negative fitness to positive distance for display
            positive_fitness = [-f for f in current_fitness if f != float('-inf')]
            best_positive_fitness = -current_fitness[best_idx] if current_fitness[best_idx] != float('-inf') else float('inf')
            avg_positive_fitness = np.mean(positive_fitness) if positive_fitness else float('inf')
            
            evolution_data.append({
                'generation': gen,
                'fitness_values': positive_fitness,  # Store positive values for display
                'best_fitness': best_positive_fitness,  # Store positive for display
                'avg_fitness': avg_positive_fitness,  # Store average positive fitness
                'best_solution': best_solution,
                'population_size': len(current_fitness)
            })
        
        # Create animation
        fig = plt.figure(figsize=(20, 12))
        fig.patch.set_facecolor(self.colors['background'])
        fig.suptitle('Genetic Algorithm Evolution for 2TP', fontsize=20, color=self.colors['text'])
        
        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        ax_graph = fig.add_subplot(gs[:2, :2])
        ax_fitness_dist = fig.add_subplot(gs[0, 2])
        ax_fitness_evolution = fig.add_subplot(gs[1, 2])
        ax_solution_info = fig.add_subplot(gs[2, :])
        
        # Get graph layout
        G = graph.G
        pos = nx.spring_layout(G, seed=42)
        
        # Sample frames for smoother animation - we'll show every 5th generation for performance
        frame_skip = max(1, generations // 100)  # Show max 100 frames
        sampled_data = [evolution_data[i] for i in range(0, len(evolution_data), frame_skip)]
        
        def animate_evolution(frame):
            if frame >= len(sampled_data):
                return
            
            gen_data = sampled_data[frame]
            actual_generation = gen_data['generation']
            
            # Clear all axes
            for ax in [ax_graph, ax_fitness_dist, ax_fitness_evolution, ax_solution_info]:
                ax.clear()
                ax.set_facecolor(self.colors['background'])
            
            # Panel 1: Graph with best solution
            ax_graph.set_title(f'Best Solution - Generation {actual_generation}', fontsize=14, color=self.colors['text'])
            self._draw_solution_on_graph(ax_graph, graph, pos, terminals, gen_data['best_solution'])
            
            # Panel 2: Population fitness distribution
            ax_fitness_dist.set_title('Fitness Distribution', fontsize=12, color=self.colors['text'])
            if gen_data['fitness_values']:
                valid_fitness = [f for f in gen_data['fitness_values'] if f != float('-inf')]
                if valid_fitness:
                    ax_fitness_dist.hist(valid_fitness, bins=10, color=self.colors['accent'], alpha=0.7)
                    ax_fitness_dist.axvline(gen_data['best_fitness'], color=self.colors['primary'], 
                                           linestyle='--', linewidth=2, label='Best')
            ax_fitness_dist.set_xlabel('Fitness', color=self.colors['text'])
            ax_fitness_dist.set_ylabel('Count', color=self.colors['text'])
            ax_fitness_dist.tick_params(colors=self.colors['text'])
            
            # Panel 3: Fitness evolution over time
            ax_fitness_evolution.set_title('Fitness Evolution', fontsize=12, color=self.colors['text'])
            if frame > 0:
                # Show fitness evolution up to current frame using sampled data
                generations_shown = [sampled_data[i]['generation'] for i in range(frame + 1)]
                best_fitness_history = [sampled_data[i]['best_fitness'] for i in range(frame + 1)]
                avg_fitness_history = [sampled_data[i]['avg_fitness'] for i in range(frame + 1)]
                
                # Plot best fitness
                ax_fitness_evolution.plot(generations_shown, best_fitness_history, 
                                        color=self.colors['primary'], linewidth=2, label='Best Fitness')
                # Plot average fitness with lower alpha
                ax_fitness_evolution.plot(generations_shown, avg_fitness_history, 
                                        color=self.colors['secondary'], linewidth=2, alpha=0.6, label='Avg Fitness')
                ax_fitness_evolution.legend(loc='upper right')
            ax_fitness_evolution.set_xlabel('Generation', color=self.colors['text'])
            ax_fitness_evolution.set_ylabel('Fitness', color=self.colors['text'])
            ax_fitness_evolution.tick_params(colors=self.colors['text'])
            ax_fitness_evolution.grid(True, alpha=0.3)
            
            # Panel 4: Solution information
            ax_solution_info.set_title(f'Generation {actual_generation} Statistics', fontsize=12, color=self.colors['text'])
            
            # Decode best solution
            best_points = ga.byte_to_number(gen_data['best_solution'])
            meeting_point, dropping_point = best_points
            
            info_text = f"Best Fitness: {gen_data['best_fitness']:.2f}\n"
            info_text += f"Avg Fitness: {gen_data['avg_fitness']:.2f}\n"
            info_text += f"Meeting Point: {meeting_point}\n"
            info_text += f"Dropping Point: {dropping_point}\n"
            info_text += f"Population Size: {gen_data['population_size']}"
            
            ax_solution_info.text(0.1, 0.5, info_text, fontsize=12, color=self.colors['text'],
                                transform=ax_solution_info.transAxes, verticalalignment='center')
            
            ax_solution_info.set_xlim(0, 1)
            ax_solution_info.set_ylim(0, 1)
            ax_solution_info.axis('off')
        
        print("Generating animation frames...")
        anim = animation.FuncAnimation(fig, animate_evolution, frames=len(sampled_data), 
                                     interval=100, repeat=True)  # Faster frame rate
        output_path = self.output_folder / f'ga_evolution_{num_nodes}nodes_{generations}gen.mp4'
        anim.save(str(output_path), writer='ffmpeg', fps=10, dpi=120)  # Higher FPS and DPI
        plt.close()
        print(f"Saved: {output_path}")
        return anim
    
    def _draw_solution_on_graph(self, ax, graph, pos, terminals, solution):
        """Helper to draw a solution on the graph"""
        G = graph.G
        agent1_start, agent2_start, agent1_dest, agent2_dest = terminals
        
        # Draw base network
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=self.colors['text'], 
                             node_size=100, alpha=0.3)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=self.colors['text'], 
                             alpha=0.1, width=0.5)
        
        # Decode solution
        ga = GeneticAlgorithm(graph, pop_size=10)  # Dummy GA for decoding
        ga.set_terminals(*terminals)
        points = ga.byte_to_number(solution)
        meeting_point, dropping_point = points
        
        # Draw terminals
        terminal_nodes = [agent1_start, agent2_start, agent1_dest, agent2_dest]
        terminal_colors = [self.colors['primary'], self.colors['secondary'], 
                          self.colors['primary'], self.colors['secondary']]
        
        for node, color in zip(terminal_nodes, terminal_colors):
            nx.draw_networkx_nodes(G, pos, nodelist=[node], ax=ax,
                                 node_color=color, node_size=300, alpha=0.8)
        
        # Draw meeting and dropping points
        if meeting_point in G.nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=[meeting_point], ax=ax,
                                 node_color=self.colors['meeting'], node_size=400, alpha=0.9)
        
        if dropping_point in G.nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=[dropping_point], ax=ax,
                                 node_color=self.colors['dropping'], node_size=400, alpha=0.9)
        
        # Draw solution paths
        try:
            # Paths to meeting point
            path1_to_m = nx.shortest_path(G, agent1_start, meeting_point, weight='weight')
            path2_to_m = nx.shortest_path(G, agent2_start, meeting_point, weight='weight')
            
            # Path from meeting to dropping
            path_m_to_d = nx.shortest_path(G, meeting_point, dropping_point, weight='weight')
            
            # Paths from dropping point
            path_d_to_1 = nx.shortest_path(G, dropping_point, agent1_dest, weight='weight')
            path_d_to_2 = nx.shortest_path(G, dropping_point, agent2_dest, weight='weight')
            
            # Draw path edges
            all_paths = [path1_to_m, path2_to_m, path_m_to_d, path_d_to_1, path_d_to_2]
            path_colors = [self.colors['primary'], self.colors['secondary'], 
                          self.colors['meeting'], self.colors['primary'], self.colors['secondary']]
            
            for path, color in zip(all_paths, path_colors):
                if len(path) > 1:
                    path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                    nx.draw_networkx_edges(G, pos, edgelist=path_edges, ax=ax,
                                         edge_color=color, width=3, alpha=0.7)
        except:
            pass  # Skip if paths don't exist
        
        ax.set_aspect('equal')
        ax.axis('off')
    
    def create_comparison_animation(self):
        """Create animation comparing different graph sizes"""
        print("Creating comparison animation...")
        
        graph_sizes = [25, 50, 100]  # Larger graphs
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor(self.colors['background'])
        fig.suptitle('GA Performance on Different Graph Sizes', fontsize=18, color=self.colors['text'])
        
        results = {}
        
        for size in graph_sizes:
            print(f"Running GA on graph size {size}...")
            graph = TransportationGraph(size)
            terminals = graph.generate_random_terminals(rng=random.Random(42))
            
            # Verify terminals are different
            agent1_start, agent2_start, agent1_dest, agent2_dest = terminals
            print(f"  Graph {size} terminals: A1({agent1_start}→{agent1_dest}), A2({agent2_start}→{agent2_dest})")
            
            ga = GeneticAlgorithm(graph, pop_size=40, seed=42)
            ga.set_terminals(*terminals)
            
            # Run for 4000 generations using EXACT same process as ga.run()
            population = ga.populate()
            evaluated_solutions = set()
            
            best_fitness_over_time = []
            avg_fitness_over_time = []
            
            generations_to_run = 4000  # 4000 generations for comparison
            for gen in tqdm(range(generations_to_run), desc=f"Graph size {size}"):
                # Track unique solutions (exactly like in ga.run())
                for ind in population:
                    evaluated_solutions.add(ind)
                    
                current_fitness = [ga.fitness(ind) for ind in population]
                valid_fitness = [f for f in current_fitness if f != float('-inf')]
                
                # Convert to positive distance values for display
                best_positive = -max(current_fitness) if current_fitness else float('inf')
                avg_positive = -np.mean(valid_fitness) if valid_fitness else float('inf')
                
                best_fitness_over_time.append(best_positive)
                avg_fitness_over_time.append(avg_positive)
                
                # Evolve population (exactly like in ga.run())
                if gen < generations_to_run - 1:
                    # Generate new unique individual
                    new_individual = ga.evolve(population)
                    while new_individual in evaluated_solutions:
                        new_individual = ga.evolve(population)
                    
                    population.append(new_individual)
                    population = sorted(population, key=ga.fitness, reverse=True)[:40]
            
            results[size] = {
                'best': best_fitness_over_time,
                'avg': avg_fitness_over_time
            }
        
        # Sample data for animation (every 20th generation due to more generations)
        sampled_results = {}
        for size in graph_sizes:
            sampled_results[size] = {
                'best': [results[size]['best'][i] for i in range(0, len(results[size]['best']), 20)],
                'avg': [results[size]['avg'][i] for i in range(0, len(results[size]['avg']), 20)]
            }
        
        def animate_comparison(frame):
            for i, (size, ax) in enumerate(zip(graph_sizes, axes)):
                ax.clear()
                ax.set_facecolor(self.colors['background'])
                ax.set_title(f'Graph Size: {size} nodes', fontsize=14, color=self.colors['text'])
                
                if frame < len(sampled_results[size]['best']):
                    generations = list(range(0, (frame + 1) * 20, 20))  # Adjust x-axis for sampling every 20th
                    best_fitness_values = sampled_results[size]['best'][:frame + 1]
                    avg_fitness_values = sampled_results[size]['avg'][:frame + 1]
                    
                    # Plot best fitness
                    ax.plot(generations, best_fitness_values, color=self.colors['primary'], 
                           linewidth=2, label='Best Fitness')
                    # Plot average fitness with lower alpha
                    ax.plot(generations, avg_fitness_values, color=self.colors['secondary'], 
                           linewidth=2, alpha=0.6, label='Avg Fitness')
                    
                    if i == 0:  # Show legend only on first subplot
                        ax.legend(loc='upper right')
                
                ax.set_xlabel('Generation', color=self.colors['text'])
                ax.set_ylabel('Distance', color=self.colors['text'])
                ax.tick_params(colors=self.colors['text'])
                ax.grid(True, alpha=0.3)
        
        max_frames = max(len(sampled_results[size]['best']) for size in graph_sizes)
        anim = animation.FuncAnimation(fig, animate_comparison, frames=max_frames, interval=50)  # Faster frame rate
        output_path = self.output_folder / 'algorithm_comparison.mp4'
        anim.save(str(output_path), writer='ffmpeg', fps=20, dpi=120)  # Higher FPS
        plt.close()
        print(f"Saved: {output_path}")
        return anim
    
    def create_multiple_ga_evolution_videos(self, generations=4000, pop_size=40, num_videos=3):
        """Create multiple GA evolution videos with different graph sizes and terminals"""
        graph_sizes = [50, 200, 500]  # Different sizes for each video
        print(f"Creating {num_videos} GA evolution videos with different graph sizes and terminals...")
        
        for video_idx in range(num_videos):
            num_nodes = graph_sizes[video_idx]
            print(f"\nGenerating video {video_idx + 1}/{num_videos} - {num_nodes} nodes...")
            
            # Create graph with different seed for terminals (larger differences)
            graph = TransportationGraph(num_nodes)
            # Use much larger seed differences to ensure very different terminal combinations
            terminals = graph.generate_random_terminals(rng=random.Random(42 + video_idx * 1000))
            
            # Verify all 4 terminals are different (they should be due to our fix)
            agent1_start, agent2_start, agent1_dest, agent2_dest = terminals
            print(f"  Video {video_idx + 1} terminals: A1({agent1_start}→{agent1_dest}), A2({agent2_start}→{agent2_dest})")
            
            ga = GeneticAlgorithm(graph, pop_size=pop_size, seed=42 + video_idx * 1000)
            ga.set_terminals(*terminals)
            
            # Collect evolution data using exact same process as ga.run()
            evolution_data = []
            population = ga.populate()
            evaluated_solutions = set()
            
            for gen in tqdm(range(generations), desc=f"Video {video_idx + 1} - Running GA"):
                # Track unique solutions (exactly like in ga.run())
                for ind in population:
                    evaluated_solutions.add(ind)
                    
                current_fitness = [ga.fitness(individual) for individual in population]
                valid_fitness = [f for f in current_fitness if f != float('-inf')]
                best_idx = np.argmax(current_fitness)
                best_solution = population[best_idx]
                
                # Convert to positive distance values for display
                best_positive = -current_fitness[best_idx]
                avg_positive = -np.mean(valid_fitness) if valid_fitness else float('inf')
                
                # Store generation data
                evolution_data.append({
                    'generation': gen,
                    'fitness_values': [-f for f in current_fitness],  # Convert to positive
                    'best_fitness': best_positive,
                    'avg_fitness': avg_positive,
                    'best_solution': best_solution,
                    'population_size': len(population)
                })
                
                # Evolve population (exactly like in ga.run())
                if gen < generations - 1:
                    # Generate new unique individual
                    new_individual = ga.evolve(population)
                    while new_individual in evaluated_solutions:
                        new_individual = ga.evolve(population)
                    
                    population.append(new_individual)
                    population = sorted(population, key=ga.fitness, reverse=True)[:pop_size]
            
            # Sample data for animation (every 20th generation for performance)
            sample_every = 20
            sampled_data = [evolution_data[i] for i in range(0, len(evolution_data), sample_every)]
            if evolution_data[-1] not in sampled_data:  # Ensure last generation is included
                sampled_data.append(evolution_data[-1])
            
            # Create animation for this video
            fig = plt.figure(figsize=(20, 12))
            fig.patch.set_facecolor(self.colors['background'])
            fig.suptitle(f'GA Evolution for 2TP - Example {video_idx + 1}', fontsize=20, color=self.colors['text'])
            
            # Create subplots
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            ax_graph = fig.add_subplot(gs[:2, :2])
            ax_fitness_dist = fig.add_subplot(gs[0, 2])
            ax_fitness_evolution = fig.add_subplot(gs[1, 2])
            ax_solution_info = fig.add_subplot(gs[2, :])
            
            # Get graph layout
            G = graph.G
            pos = nx.spring_layout(G, seed=42 + video_idx * 1000)
            
            def animate_evolution(frame):
                if frame >= len(sampled_data):
                    return
                
                gen_data = sampled_data[frame]
                actual_generation = gen_data['generation']
                
                # Clear all axes
                for ax in [ax_graph, ax_fitness_dist, ax_fitness_evolution, ax_solution_info]:
                    ax.clear()
                    ax.set_facecolor(self.colors['background'])
                
                # Panel 1: Graph with best solution
                ax_graph.set_title(f'Best Solution - Generation {actual_generation}', fontsize=14, color=self.colors['text'])
                self._draw_solution_on_graph(ax_graph, graph, pos, terminals, gen_data['best_solution'])
                
                # Panel 2: Population fitness distribution
                ax_fitness_dist.set_title('Fitness Distribution', fontsize=12, color=self.colors['text'])
                if gen_data['fitness_values']:
                    valid_fitness = [f for f in gen_data['fitness_values'] if f != float('inf')]
                    if valid_fitness:
                        ax_fitness_dist.hist(valid_fitness, bins=10, color=self.colors['accent'], alpha=0.7)
                        ax_fitness_dist.axvline(gen_data['best_fitness'], color=self.colors['primary'], 
                                               linestyle='--', linewidth=2, label='Best')
                ax_fitness_dist.set_xlabel('Distance', color=self.colors['text'])
                ax_fitness_dist.set_ylabel('Count', color=self.colors['text'])
                ax_fitness_dist.tick_params(colors=self.colors['text'])
                
                # Panel 3: Fitness evolution over time
                ax_fitness_evolution.set_title('Distance Evolution', fontsize=12, color=self.colors['text'])
                if frame > 0:
                    # Show fitness evolution up to current frame using sampled data
                    generations_shown = [sampled_data[i]['generation'] for i in range(frame + 1)]
                    best_fitness_history = [sampled_data[i]['best_fitness'] for i in range(frame + 1)]
                    avg_fitness_history = [sampled_data[i]['avg_fitness'] for i in range(frame + 1)]
                    
                    # Plot best fitness
                    ax_fitness_evolution.plot(generations_shown, best_fitness_history, 
                                            color=self.colors['primary'], linewidth=2, label='Best Distance')
                    # Plot average fitness with lower alpha
                    ax_fitness_evolution.plot(generations_shown, avg_fitness_history, 
                                            color=self.colors['secondary'], linewidth=2, alpha=0.6, label='Avg Distance')
                    ax_fitness_evolution.legend(loc='upper right')
                ax_fitness_evolution.set_xlabel('Generation', color=self.colors['text'])
                ax_fitness_evolution.set_ylabel('Distance', color=self.colors['text'])
                ax_fitness_evolution.tick_params(colors=self.colors['text'])
                ax_fitness_evolution.grid(True, alpha=0.3)
                
                # Panel 4: Solution information
                ax_solution_info.set_title(f'Generation {actual_generation} Statistics', fontsize=12, color=self.colors['text'])
                
                # Decode best solution
                best_points = ga.byte_to_number(gen_data['best_solution'])
                meeting_point, dropping_point = best_points
                
                info_text = f"Best Distance: {gen_data['best_fitness']:.2f}\n"
                info_text += f"Avg Distance: {gen_data['avg_fitness']:.2f}\n"
                info_text += f"Meeting Point: {meeting_point}\n"
                info_text += f"Dropping Point: {dropping_point}\n"
                info_text += f"Population Size: {gen_data['population_size']}"
                
                ax_solution_info.text(0.1, 0.5, info_text, fontsize=12, color=self.colors['text'],
                                    transform=ax_solution_info.transAxes, verticalalignment='center')
                
                ax_solution_info.set_xlim(0, 1)
                ax_solution_info.set_ylim(0, 1)
                ax_solution_info.axis('off')
            
            print(f"Generating animation frames for video {video_idx + 1}...")
            anim = animation.FuncAnimation(fig, animate_evolution, frames=len(sampled_data), 
                                         interval=100, repeat=True)  # Faster frame rate
            output_path = self.output_folder / f'ga_evolution_{num_nodes}nodes_4000gen_40pop_example{video_idx + 1}.mp4'
            anim.save(str(output_path), writer='ffmpeg', fps=10, dpi=120)  # Higher FPS and DPI
            plt.close()
            print(f"Saved: {output_path}")
        
        print(f"✅ All {num_videos} GA evolution videos completed!")

def main():
    """Generate all video components"""
    print("Starting video generation for GA-2TP project...")
    generator = VideoGenerator()
    
    try:
        print("\n1. Generating problem introduction...")
        generator.create_problem_intro_animation(num_nodes=20)
        
        print("\n2. Generating multiple GA evolution videos (50, 200, 500 nodes, 4000 gen, 40 pop)...")
        generator.create_multiple_ga_evolution_videos()
        
        print("\n3. Generating comparison animation...")
        generator.create_comparison_animation()
        
        print(f"\n✅ Video generation complete! Check the {generator.output_folder} folder.")
        print("Generated files:")
        for file in generator.output_folder.glob("*.mp4"):
            print(f"  - {file.name}")
            
    except Exception as e:
        print(f"❌ Error during video generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
