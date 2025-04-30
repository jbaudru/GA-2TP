import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def plot_fitness_progress(fitness_history, pop_size=None, steps=None):
    """Plot the fitness progress over generations"""
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
    name = f"../images/fitness_progress_N{pop_size}_s{steps}_P{pop_size}.png"
    plt.savefig(name, dpi=300)
    plt.show()

def raincloud_plot(data, labels, ax):
    """Create a raincloud plot for comparing distributions"""
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

def plot_algorithm_comparison(ga_results, exact_results, graph_size, num_instances):
    """Plot comparison between GA and exact algorithm results"""
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
    
    plt.savefig(f'../images/comparison_N{graph_size}_I{num_instances}.png', dpi=300)
    plt.show()
    
    return {
        'mean_relative_error': np.mean(rel_errors),
        'std_relative_error': np.std(rel_errors),
        'max_relative_error': np.max(rel_errors)
    }