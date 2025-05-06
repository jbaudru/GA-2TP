import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns

def parse_error_distribution_files(directory, parameter, prefix="error_distribution"):
    """Parse error distribution files and extract relevant data."""
    data = []
    for filename in os.listdir(directory):
        if filename.startswith(prefix):
            with open(os.path.join(directory, filename), 'r') as file:
                lines = file.readlines()
                param_value = None
                mre_values = []
                
                # Extract the specified parameter
                for line in lines:
                    if line.startswith(f"{parameter}:"):
                        param_value = line.split(":")[1].strip()
                        break
                
                # Extract relative errors
                for line in lines[12:]:
                    try:
                        parts = line.split()
                        if len(parts) >= 4:
                            ga_distance = float(parts[1])
                            exact_distance = float(parts[2])
                            relative_error = float(parts[4])
                            mre_values.append(relative_error)
                    except (IndexError, ValueError):
                        continue
                
                if param_value is not None and mre_values:
                    data.append({"parameter": param_value, "mre_values": mre_values})
    return data

def plot_mre_distribution(data, parameter):
    """Create elegant distribution plots for MRE values."""
    plt.figure(figsize=(8,6))
    
    # Sort data by parameter value
    data = sorted(data, key=lambda x: float(x["parameter"].split(":")[0].strip()))
    
    # Create color palette
    palette = sns.color_palette("viridis", len(data))
    
    # Setup consistent x-grid
    x_grid = np.linspace(0, 100, 500)  # Focus on 0 to 0.5 range
    
    # Plot smooth distributions
    for idx, entry in enumerate(data):
        param_value = entry["parameter"]
        mre_values = np.array(entry["mre_values"])
        
        if len(mre_values) < 2:
            continue
            
        try:
            # Apply kernel density estimation
            kde = gaussian_kde(mre_values, bw_method='scott')
            density = kde(x_grid)
            
            # Normalize for consistent scaling
            if np.max(density) > 0:
                density = density / np.max(density)
                
                # Plot smooth curve with filled area
                plt.plot(x_grid, density, color=palette[idx], linewidth=2, 
                         label=f"{parameter}: {param_value}")
                plt.fill_between(x_grid, density, alpha=0.15, color=palette[idx])
        except:
            # Fallback to histogram if KDE fails
            hist, edges = np.histogram(mre_values, bins=20, range=(0, 100), density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            if np.max(hist) > 0:
                hist = hist / np.max(hist)
                plt.plot(centers, hist, color=palette[idx], linewidth=2,
                         label=f"{parameter}: {param_value}")
    
    # Style the plot
    plt.xlabel("Mean Relative Error (MRE)", fontsize=12)
    plt.ylabel("Normalized Kernel Density Estimation (KDE)", fontsize=12)
    plt.xlim(-4, 100)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3, linestyle='--')
    #plt.title(f"Distribution of Mean Relative Error by {parameter}", fontsize=14)
    
    # Add legend with clean formatting
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Save and show
    plt.savefig(f"../output/city{parameter}_T_20_P20.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Directory containing the error distribution files
    directory = "../output/error/realworld_P20_T_test/"
    
    # User-specified parameter for comparison
    parameter = "Graph Size"  # Change to "Population Size" if needed
    
    # Parse files and plot
    data = parse_error_distribution_files(directory, parameter)
    
    if data:
        plot_mre_distribution(data, parameter)
    else:
        print(f"No valid error distribution files found for parameter: {parameter}")