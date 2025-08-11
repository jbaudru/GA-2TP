import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
import random
import sys
import os
import json
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.graph import TransportationGraph
from algorithms.genetic_algorithm import GeneticAlgorithm

class RealRoadNetworkVideoGenerator:
    def __init__(self, output_folder="../video_output"):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        
        # Set up visual style with enhanced colors
        plt.style.use('dark_background')
        self.colors = {
            'primary': '#00ff88',      # Bright green for Agent 1
            'secondary': '#ff6b6b',    # Bright red for Agent 2  
            'accent': '#4ecdc4',       # Teal for shared paths
            'background': '#1a1a1a',
            'text': '#ffffff',
            'meeting': '#ffd700',      # Gold for meeting points
            'dropping': '#ff8c00',     # Orange for dropping points
            'major_road': '#888888',   # Light gray for major roads
            'minor_road': '#444444'    # Dark gray for minor roads
        }
    
    def load_real_road_network(self, filename="BRU.json"):
        """Load real road network from JSON file following road_network_experiment pattern"""
        print(f"Loading real road network from {filename}...")
        
        # Load from data directory (same as in road_network_experiment)
        filepath = os.path.join("..", "data", filename)
        
        # Create empty graph and load from JSON (same pattern as road_network_experiment)
        graph = TransportationGraph(0).load_from_json(filepath)
        
        print(f"Loaded real road network: {graph.G.number_of_nodes()} nodes, {graph.G.number_of_edges()} edges")
        
        return graph
    
    def create_real_road_network_video(self, filename="BRU.json", generations=8000, pop_size=100):
        """Create video showing GA evolution on real Brussels road network"""
        print(f"Creating real road network video with {filename} data...")
        
        # Load real road network (following road_network_experiment pattern)
        graph = self.load_real_road_network(filename)
        
        # Generate terminals with starts clustered in one area, destinations clustered in another area
        def get_clustered_terminals_for_visualization(graph, min_cluster_separation=0.8, max_within_cluster=0.4):
            """
            Generate terminals optimized for visualization:
            - Start points: clustered together in one area of the network
            - Destination points: clustered together in a different area of the network
            - Areas should be well separated for clear visualization
            """
            nodes = list(graph.G.nodes())
            max_attempts = 200
            
            # Use current time for truly random seed each run
            import time
            random_seed = int(time.time() * 1000) % 1000000
            print(f"Using random seed: {random_seed}")
            
            # Pre-calculate network diameter estimate for distance thresholds
            sample_nodes = random.Random(random_seed).sample(nodes, min(50, len(nodes)))
            max_network_dist = 0
            for i in range(len(sample_nodes)):
                for j in range(i+1, min(i+10, len(sample_nodes))):
                    try:
                        dist = nx.shortest_path_length(graph.G, sample_nodes[i], sample_nodes[j], weight='weight')
                        max_network_dist = max(max_network_dist, dist)
                    except:
                        continue
            
            min_area_separation = max_network_dist * min_cluster_separation
            max_intra_cluster = max_network_dist * max_within_cluster
            
            print(f"Target distances - Area separation: ≥{min_area_separation:.0f}m, Within cluster: ≤{max_intra_cluster:.0f}m")
            
            for attempt in range(max_attempts):
                rng = random.Random(random_seed + attempt)
                all_nodes = list(nodes)
                rng.shuffle(all_nodes)
                
                # Strategy: Find two well-separated areas, then pick nodes within each area
                for i in range(len(all_nodes) - 3):
                    # Pick first start point as anchor for start cluster
                    A1_start = all_nodes[i]
                    
                    # Find nodes close to A1_start for start cluster
                    start_cluster_candidates = []
                    for node in all_nodes:
                        if node != A1_start:
                            try:
                                dist = nx.shortest_path_length(graph.G, A1_start, node, weight='weight')
                                if dist <= max_intra_cluster:
                                    start_cluster_candidates.append(node)
                            except:
                                continue
                    
                    if len(start_cluster_candidates) < 1:
                        continue
                        
                    # Pick A2_start from start cluster
                    A2_start = rng.choice(start_cluster_candidates)
                    
                    # Now find destination area that's far from start area
                    for j in range(len(all_nodes)):
                        if all_nodes[j] in [A1_start, A2_start]:
                            continue
                            
                        # Check if this could be a good destination area anchor
                        potential_dest_anchor = all_nodes[j]
                        
                        try:
                            # Check distance from start area to potential destination area
                            dist_to_start_area = min(
                                nx.shortest_path_length(graph.G, A1_start, potential_dest_anchor, weight='weight'),
                                nx.shortest_path_length(graph.G, A2_start, potential_dest_anchor, weight='weight')
                            )
                            
                            if dist_to_start_area < min_area_separation:
                                continue  # Too close to start area
                                
                            # Find nodes close to this destination anchor
                            dest_cluster_candidates = []
                            for node in all_nodes:
                                if node not in [A1_start, A2_start, potential_dest_anchor]:
                                    try:
                                        dist = nx.shortest_path_length(graph.G, potential_dest_anchor, node, weight='weight')
                                        if dist <= max_intra_cluster:
                                            dest_cluster_candidates.append(node)
                                    except:
                                        continue
                            
                            # Add the anchor itself as a candidate
                            dest_cluster_candidates.append(potential_dest_anchor)
                            
                            if len(dest_cluster_candidates) < 2:
                                continue
                                
                            # Pick two destinations from destination cluster
                            selected_dests = rng.sample(dest_cluster_candidates, 2)
                            A1_dest, A2_dest = selected_dests
                            
                            # Verify all paths exist
                            try:
                                test_distances = [
                                    nx.shortest_path_length(graph.G, A1_start, A1_dest, weight='weight'),
                                    nx.shortest_path_length(graph.G, A2_start, A2_dest, weight='weight'),
                                    nx.shortest_path_length(graph.G, A1_start, A2_dest, weight='weight'),
                                    nx.shortest_path_length(graph.G, A2_start, A1_dest, weight='weight')
                                ]
                                
                                # Verify cluster distances
                                start_cluster_dist = nx.shortest_path_length(graph.G, A1_start, A2_start, weight='weight')
                                dest_cluster_dist = nx.shortest_path_length(graph.G, A1_dest, A2_dest, weight='weight')
                                
                                if (start_cluster_dist <= max_intra_cluster and 
                                    dest_cluster_dist <= max_intra_cluster and
                                    all(d > max_network_dist * 0.1 for d in test_distances)):
                                    
                                    print(f"✅ Found clustered terminals (attempt {attempt + 1}):")
                                    print(f"   Start cluster distance: {start_cluster_dist:.0f}m")
                                    print(f"   Destination cluster distance: {dest_cluster_dist:.0f}m")
                                    print(f"   Area separation: {dist_to_start_area:.0f}m")
                                    return (A1_start, A2_start, A1_dest, A2_dest)
                                    
                            except:
                                continue
                                
                        except:
                            continue
            
            # Fallback: return any valid terminals if clustering fails
            print("⚠️  Using fallback terminal selection...")
            rng = random.Random(random_seed + 999)
            return graph.generate_random_terminals(rng)
        
        terminals = get_clustered_terminals_for_visualization(graph)
        A1_start, A2_start, A1_dest, A2_dest = terminals
        
        print(f"Selected clustered terminals for visualization:")
        print(f"  Agent 1: start={A1_start}, destination={A1_dest}")
        print(f"  Agent 2: start={A2_start}, destination={A2_dest}")
        
        # Calculate and display cluster distances for verification
        try:
            start_cluster_dist = nx.shortest_path_length(graph.G, A1_start, A2_start, weight='weight')
            dest_cluster_dist = nx.shortest_path_length(graph.G, A1_dest, A2_dest, weight='weight')
            
            # Calculate area separation (minimum distance between start and destination areas)
            area_separation = min(
                nx.shortest_path_length(graph.G, A1_start, A1_dest, weight='weight'),
                nx.shortest_path_length(graph.G, A1_start, A2_dest, weight='weight'),
                nx.shortest_path_length(graph.G, A2_start, A1_dest, weight='weight'),
                nx.shortest_path_length(graph.G, A2_start, A2_dest, weight='weight')
            )
            
            print(f"  ✅ Start area cohesion: {start_cluster_dist:.0f}m (both agents start nearby)")
            print(f"  ✅ Destination area cohesion: {dest_cluster_dist:.0f}m (both agents end nearby)")
            print(f"  ✅ Area separation: {area_separation:.0f}m (start area ↔ destination area)")
        except:
            print("  (Distance calculation unavailable)")
        
        # Set up genetic algorithm with truly random seed for different behavior each run
        import time
        ga_seed = int(time.time() * 1000) % 1000000
        ga = GeneticAlgorithm(graph, pop_size=pop_size, seed=ga_seed)
        print(f"GA using random seed: {ga_seed}")
        ga.set_terminals(A1_start, A2_start, A1_dest, A2_dest)
        
        print("Running GA evolution...")
        
        # Collect evolution data using exact same process as other videos
        evolution_data = []
        population = ga.populate()
        evaluated_solutions = set()
        
        for gen in tqdm(range(generations), desc="Running GA on real road network"):
            # Track unique solutions (exactly like in other videos)
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
            
            # Evolve population (exactly like in other videos)
            if gen < generations - 1:
                # Generate new unique individual
                new_individual = ga.evolve(population)
                while new_individual in evaluated_solutions:
                    new_individual = ga.evolve(population)
                
                population.append(new_individual)
                population = sorted(population, key=ga.fitness, reverse=True)[:pop_size]
        
        print("Creating animation...")
        
        # Get node positions for visualization (needed for pre-computation)
        print("Preparing visualization data...")
        node_positions = {}
        for node in graph.G.nodes():
            node_data = graph.G.nodes[node]
            if 'x' in node_data and 'y' in node_data:
                node_positions[node] = (node_data['x'], node_data['y'])
            else:
                # Fallback for nodes without coordinates
                node_positions[node] = (0, 0)
        
        # Pre-compute road network background with uniform styling
        print("Pre-computing road network background...")
        
        # Simple edge sampling for uniform visualization
        edges = list(graph.G.edges())
        #max_edges = 1000  # Reasonable number for performance
        #if len(edges) > max_edges:
        #    sampled_edges = random.Random(42).sample(edges, max_edges)
        #else:
        
        sampled_edges = edges
        
        # Pre-compute all edge coordinates with uniform styling
        background_x_coords = []
        background_y_coords = []
        
        for edge in sampled_edges:
            if edge[0] in node_positions and edge[1] in node_positions:
                pos1 = node_positions[edge[0]]
                pos2 = node_positions[edge[1]]
                background_x_coords.extend([pos1[0], pos2[0], None])
                background_y_coords.extend([pos1[1], pos2[1], None])
        
        print(f"Road network: {len(sampled_edges)} edges prepared")
        
        # Calculate map bounds once
        all_x = [pos[0] for pos in node_positions.values() if pos[0] is not None]
        all_y = [pos[1] for pos in node_positions.values() if pos[1] is not None]
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        # Add padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        padding = 0.05
        x_min -= x_range * padding
        x_max += x_range * padding
        y_min -= y_range * padding
        y_max += y_range * padding
        
        # Sample data for animation (more aggressive sampling for better performance)
        sample_every = 50  # Increased from 40 for better performance
        sampled_data = [evolution_data[i] for i in range(0, len(evolution_data), sample_every)]
        if evolution_data[-1] not in sampled_data:  # Ensure last generation is included
            sampled_data.append(evolution_data[-1])
        
        print(f"Animation will show {len(sampled_data)} frames (sampled from {len(evolution_data)} generations)")
        
        # Create animation
        fig = plt.figure(figsize=(20, 12))
        fig.patch.set_facecolor(self.colors['background'])
        fig.suptitle('GA Evolution on Real Brussels Road Network', fontsize=20, color=self.colors['text'])
        
        # Create subplots with larger map
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
        ax_map = fig.add_subplot(gs[:, :3])  # Larger map view (3 out of 4 columns)
        ax_fitness_evolution = fig.add_subplot(gs[0, 3])
        ax_solution_info = fig.add_subplot(gs[1, 3])
        
        def animate_evolution(frame):
            if frame >= len(sampled_data):
                return
            
            gen_data = sampled_data[frame]
            actual_generation = gen_data['generation']
            
            # Clear all axes
            for ax in [ax_map, ax_fitness_evolution, ax_solution_info]:
                ax.clear()
                ax.set_facecolor(self.colors['background'])
            
            # Panel 1: Road network map with solution
            ax_map.set_title(f'Brussels Road Network - Generation {actual_generation}', 
                            fontsize=14, color=self.colors['text'])
            
            # Set map bounds (pre-computed)
            ax_map.set_xlim(x_min, x_max)
            ax_map.set_ylim(y_min, y_max)
            
            # Draw uniform road network background - all edges same color and opacity
            if background_x_coords:
                ax_map.plot(background_x_coords, background_y_coords, color='#666666', 
                           alpha=0.6, linewidth=0.5, rasterized=True, solid_capstyle='round')

            # Draw terminals
            terminal_positions = {
                'A1_start': node_positions[A1_start],
                'A2_start': node_positions[A2_start],
                'A1_dest': node_positions[A1_dest],
                'A2_dest': node_positions[A2_dest]
            }
            
            # Draw terminals with enhanced visibility and better styling
            ax_map.scatter(*terminal_positions['A1_start'], s=150, c=self.colors['primary'], 
                          marker='s', edgecolors='white', linewidths=3, label='Agent 1 Start', zorder=12)
            ax_map.scatter(*terminal_positions['A2_start'], s=150, c=self.colors['secondary'], 
                          marker='s', edgecolors='white', linewidths=3, label='Agent 2 Start', zorder=12)
            ax_map.scatter(*terminal_positions['A1_dest'], s=150, c=self.colors['primary'], 
                          marker='o', edgecolors='white', linewidths=3, label='Agent 1 Dest', zorder=12)
            ax_map.scatter(*terminal_positions['A2_dest'], s=150, c=self.colors['secondary'], 
                          marker='o', edgecolors='white', linewidths=3, label='Agent 2 Dest', zorder=12)
            
            # Draw best solution path if available
            if gen_data['best_solution'] is not None:
                try:
                    # Decode best solution
                    best_points = ga.byte_to_number(gen_data['best_solution'])
                    meeting_point, dropping_point = best_points
                    
                    if meeting_point in node_positions and dropping_point in node_positions:
                        # Draw meeting and dropping points with enhanced visibility
                        ax_map.scatter(*node_positions[meeting_point], s=200, c=self.colors['meeting'], 
                                      marker='*', edgecolors='white', linewidths=3, label='Meeting Point', zorder=15)
                        ax_map.scatter(*node_positions[dropping_point], s=180, c=self.colors['dropping'], 
                                      marker='D', edgecolors='white', linewidths=3, label='Dropping Point', zorder=15)
                        
                        # Draw solution paths
                        try:
                            # Paths to meeting point
                            path1_to_m = nx.shortest_path(graph.G, A1_start, meeting_point, weight='weight')
                            path2_to_m = nx.shortest_path(graph.G, A2_start, meeting_point, weight='weight')
                            
                            # Path from meeting to dropping
                            path_m_to_d = nx.shortest_path(graph.G, meeting_point, dropping_point, weight='weight')
                            
                            # Paths from dropping point
                            path_d_to_1 = nx.shortest_path(graph.G, dropping_point, A1_dest, weight='weight')
                            path_d_to_2 = nx.shortest_path(graph.G, dropping_point, A2_dest, weight='weight')
                            
                            # Draw enhanced path edges with better visual hierarchy
                            all_paths = [path1_to_m, path2_to_m, path_m_to_d, path_d_to_1, path_d_to_2]
                            path_colors = [self.colors['primary'], self.colors['secondary'], 
                                          self.colors['accent'], self.colors['primary'], self.colors['secondary']]
                            path_widths = [4, 4, 3.5, 2.5, 2.5]  # Enhanced widths for better visibility
                            path_alphas = [0.95, 0.95, 0.85, 0.75, 0.75]  # Higher alpha for better contrast
                            path_styles = ['-', '-', '-', '--', '--']  # Different styles for different segments
                            
                            for path, color, width, alpha, style in zip(all_paths, path_colors, path_widths, path_alphas, path_styles):
                                if len(path) > 1:
                                    # Batch coordinates for this path
                                    path_x = []
                                    path_y = []
                                    
                                    for i in range(len(path) - 1):
                                        if path[i] in node_positions and path[i+1] in node_positions:
                                            pos1 = node_positions[path[i]]
                                            pos2 = node_positions[path[i+1]]
                                            path_x.extend([pos1[0], pos2[0], None])
                                            path_y.extend([pos1[1], pos2[1], None])
                                    
                                    # Draw entire path with enhanced styling
                                    if path_x:
                                        ax_map.plot(path_x, path_y, color=color, linewidth=width, 
                                                   alpha=alpha, linestyle=style, solid_capstyle='round', 
                                                   solid_joinstyle='round', zorder=8)
                        except:
                            pass  # Skip if paths don't exist
                except:
                    pass  # Skip if solution can't be decoded
            
            ax_map.set_aspect('equal')
            ax_map.axis('off')
            
            # Create a comprehensive legend with better organization
            legend_elements = []
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=self.colors['primary'], 
                                            markersize=10, markeredgecolor='white', markeredgewidth=1, label='Agent 1 Start'))
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['primary'], 
                                            markersize=10, markeredgecolor='white', markeredgewidth=1, label='Agent 1 Dest'))
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=self.colors['secondary'], 
                                            markersize=10, markeredgecolor='white', markeredgewidth=1, label='Agent 2 Start'))
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['secondary'], 
                                            markersize=10, markeredgecolor='white', markeredgewidth=1, label='Agent 2 Dest'))
            legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=self.colors['meeting'], 
                                            markersize=12, markeredgecolor='white', markeredgewidth=1, label='Meeting Point'))
            legend_elements.append(plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=self.colors['dropping'], 
                                            markersize=8, markeredgecolor='white', markeredgewidth=1, label='Dropping Point'))
            
            ax_map.legend(handles=legend_elements, loc='upper right', fontsize=9, 
                         fancybox=True, shadow=True, framealpha=0.9)
            
            # Panel 2: Fitness evolution over time
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
            
            # Panel 3: Solution information
            ax_solution_info.set_title(f'Generation {actual_generation} Statistics', fontsize=12, color=self.colors['text'])
            
            # Decode best solution
            try:
                best_points = ga.byte_to_number(gen_data['best_solution'])
                meeting_point, dropping_point = best_points
                
                info_text = f"Best Distance: {gen_data['best_fitness']:.2f} km\n"
                info_text += f"Avg Distance: {gen_data['avg_fitness']:.2f} km\n"
                info_text += f"Meeting Point: {meeting_point}\n"
                info_text += f"Dropping Point: {dropping_point}\n"
                info_text += f"Population Size: {gen_data['population_size']}\n"
                info_text += f"Network: {graph.G.number_of_nodes()} nodes, {graph.G.number_of_edges()} edges"
            except:
                info_text = f"Best Distance: {gen_data['best_fitness']:.2f} km\n"
                info_text += f"Avg Distance: {gen_data['avg_fitness']:.2f} km\n"
                info_text += f"Population Size: {gen_data['population_size']}\n"
                info_text += f"Network: {graph.G.number_of_nodes()} nodes, {graph.G.number_of_edges()} edges"
            
            ax_solution_info.text(0.1, 0.5, info_text, fontsize=10, color=self.colors['text'],
                                transform=ax_solution_info.transAxes, verticalalignment='center')
            
            ax_solution_info.set_xlim(0, 1)
            ax_solution_info.set_ylim(0, 1)
            ax_solution_info.axis('off')
        
        print("Generating optimized animation frames...")
        anim = animation.FuncAnimation(fig, animate_evolution, frames=len(sampled_data), 
                                     interval=200, repeat=True)  # Slightly slower for better viewing
        
        output_path = self.output_folder / f'ga_evolution_real_brussels_road_network.mp4'
        print(f"Saving video to {output_path}...")
        
        # Use progress callback for better user feedback
        def progress_callback(frame, total):
            percent = (frame + 1) / total * 100
            print(f"\rRendering video: {frame + 1}/{total} frames ({percent:.1f}%)", end="", flush=True)
        
        anim.save(str(output_path), writer='ffmpeg', fps=6, dpi=100,  # Reduced FPS and DPI for performance
                  progress_callback=progress_callback)
        
        print(f"\n✅ Real road network video saved: {output_path}")
        plt.close()
        return anim

def main():
    """Generate real road network video"""
    print("Starting real Brussels road network video generation...")
    print("⚠️  This will take a significant amount of time due to the large network size!")
    
    generator = RealRoadNetworkVideoGenerator()
    
    try:
        generator.create_real_road_network_video("BRU.json", generations=4000, pop_size=100)
        print("✅ Real road network video generation complete!")
        
    except Exception as e:
        print(f"❌ Error during video generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
