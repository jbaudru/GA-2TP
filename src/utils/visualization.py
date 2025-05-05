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
    plt.ylim(0, 20)  # Set y-axis limits from 0 to 20
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    base_name = f"output/cache_fitness_progress_N{pop_size}_s{steps}_P{pop_size}"
    img_name = f"{base_name}.png"
    txt_name = f"{base_name}.txt"
    
    # Save image
    plt.savefig(img_name, dpi=300)
    
    # Save results to text file
    with open(txt_name, 'w') as f:
        f.write(f"Fitness Progress Results\n")
        f.write(f"Population Size: {pop_size}\n")
        f.write(f"Steps: {steps}\n\n")
        f.write(f"{'Generation':<12}{'Best':<12}{'Average':<12}{'Worst':<12}{'StdDev':<12}\n")
        
        for gen, (best, avg, worst, std) in enumerate(zip(best_fitness, avg_fitness, worst_fitness, std_fitness)):
            f.write(f"{gen:<12}{best:<12.2f}{avg:<12.2f}{worst:<12.2f}{std:<12.2f}\n")
    
    plt.show()

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
    
    base_name = f'output/comparison_N{graph_size}_I{num_instances}'
    img_name = f"{base_name}.png"
    txt_name = f"{base_name}.txt"
    
    # Save image
    plt.savefig(img_name, dpi=300)
    
    # Save results to text file
    with open(txt_name, 'w') as f:
        f.write(f"Algorithm Comparison Results\n")
        f.write(f"Graph Size: {graph_size}\n")
        f.write(f"Number of Instances: {num_instances}\n\n")
        f.write(f"Error Statistics:\n")
        f.write(f"Mean Relative Error: {np.mean(rel_errors):.2f}%\n")
        f.write(f"Std Relative Error: {np.std(rel_errors):.2f}%\n")
        f.write(f"Max Relative Error: {np.max(rel_errors):.2f}%\n\n")
        
        f.write(f"{'Instance':<10}{'GA':<15}{'Exact':<15}{'Error':<15}{'Relative Error %':<15}\n")
        for i, (ga, exact, err, rel_err) in enumerate(zip(ga_results, exact_results, errors, rel_errors)):
            f.write(f"{i:<10}{ga:<15.2f}{exact:<15.2f}{err:<15.2f}{rel_err:<15.2f}\n")
    
    plt.show()
    
    return {
        'mean_relative_error': np.mean(rel_errors),
        'std_relative_error': np.std(rel_errors),
        'max_relative_error': np.max(rel_errors)
    }


def plot_mse_comparison(ga_results, exact_results, graph_size, num_instances, name):
    """
    Plot error distribution between GA and exact algorithm results
    
    Args:
        ga_results (list): Results from genetic algorithm
        exact_results (list): Results from exact algorithm
        graph_size (int): Size of the graph
        num_instances (int): Number of instances tested
        
    Returns:
        dict: Error statistics
    """
    # Calculate errors and statistics
    errors = np.array(ga_results) - np.array(exact_results)
    rel_errors = errors / np.array(exact_results) * 100
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    txt_name = f'output/error/error_distribution_' + name + '.txt'

    # Save results to text file
    with open(txt_name, 'w') as f:
        f.write(f"Error Distribution Results\n")
        f.write(f"Graph Size: {graph_size}\n")
        f.write(f"Number of Instances: {num_instances}\n\n")
        f.write(f"Error Statistics:\n")
        f.write(f"Mean Error: {mean_error:.4f}\n")
        f.write(f"Std Error: {std_error:.4f}\n")
        f.write(f"Mean Relative Error: {np.mean(rel_errors):.2f}%\n")
        f.write(f"Max Relative Error: {np.max(rel_errors):.2f}%\n\n")
        
        f.write(f"{'Instance':<10}{'GA':<15}{'Exact':<15}{'Error':<15}{'Relative Error %':<15}\n")
        for i, (ga, exact, err, rel_err) in enumerate(zip(ga_results, exact_results, errors, rel_errors)):
            f.write(f"{i:<10}{ga:<15.2f}{exact:<15.2f}{err:<15.2f}{rel_err:<15.2f}\n")
    
    return {
        'mean_error': mean_error,
        'std_error': std_error,
        'mean_relative_error': np.mean(rel_errors),
        'max_relative_error': np.max(rel_errors)
    }

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

