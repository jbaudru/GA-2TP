import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os

def parse_error_distribution_files(directory, parameter, prefix="error_distribution"):
    """
    Parse error distribution files and extract relevant data.
    
    Args:
        directory (str): Directory containing the error distribution files.
        parameter (str): Parameter to compare (e.g., "Graph Size", "Population Size").
        prefix (str): Prefix of the files to parse.
    
    Returns:
        list: A list of dictionaries containing the specified parameter and MSE values.
    """
    data = []
    for filename in os.listdir(directory):
        if filename.startswith(prefix):
            with open(os.path.join(directory, filename), 'r') as file:
                lines = file.readlines()
                param_value = None
                mse_values = []
                
                # Extract the specified parameter
                for line in lines:
                    if line.startswith(f"{parameter}:"):
                        param_value = line.split(":")[1].strip()
                        break
                
                # Extract MSE values
                for line in lines[12:]:
                    try:
                        parts = line.split()
                        if len(parts) >= 4:  # Ensure there are enough columns
                            error = float(parts[3])  # Extract the "Error" column
                            mse_values.append(error ** 2)  # Square the error to get MSE
                    except (IndexError, ValueError):
                        continue
                
                if param_value is not None and mse_values:
                    data.append({"parameter": param_value, "mse_values": mse_values})
    return data

def plot_mse_comparison(data, parameter):
    """
    Plot KDE plot for MSE distributions based on the specified parameter.
    
    Args:
        data (list): List of dictionaries containing the parameter and MSE values.
        parameter (str): Parameter to compare (e.g., "Graph Size", "Population Size").
    """
    plt.figure(figsize=(8,6))
    
    # Sort the data by the parameter value (assuming numeric values)
    data = sorted(data, key=lambda x: float(x["parameter"].split(":")[0].strip()))

    
    palette = sns.color_palette("tab10", 10)
    
    # Plot each file's KDE curve
    for idx, entry in enumerate(data):
        param_value = entry["parameter"]
        mse_values = entry["mse_values"]
        print(mse_values)
        
        if all(mse == 0 for mse in mse_values):
            # Plot a vertical line at 0 if all MSE values are 0
            plt.axvline(x=0, color=palette[idx], linestyle='--', label=f"{parameter}: {param_value}")
        else:
            sns.kdeplot(
                mse_values, label=f"{parameter}: {param_value}",
                fill=True, alpha=0.1, linewidth=1, color=palette[idx]
            )
            
    # Customize plot
    #plt.title(f"MSE Distribution KDE Plot by {parameter}")
    plt.xlabel("Mean Squared Error (MSE)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(-0.5, 5)  # Set x-axis limits from 0 to 0.5
    plt.ylim(0, 5)  # Set y-axis limits from 0 to 5
    plt.savefig(f"../output/mse_distribution_{parameter}_V100.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Directory containing the error distribution files
    directory = "../output/error/pop_size_v100/"
    
    # User-specified parameter for comparison
    parameter = "Population Size"  # Change to "Population Size" if needed
    
    # Parse files and extract data
    data = parse_error_distribution_files(directory, parameter)
    print(len(data), "files parsed")
    print(len(data[0]))
    
    # Plot KDE
    if data:
        plot_mse_comparison(data, parameter)
    else:
        print(f"No valid error distribution files found for parameter: {parameter}")