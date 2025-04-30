# Two Transfer Point (2TP) Problem Solver

This repository contains the code for the paper **"Solving the Two Transfer Point (2TP) Problem using Genetic Algorithms"**, which addresses an important optimization challenge in vehicle routing and ridesharing applications. The 2TP problem seeks to find the optimal **meeting** and **drop-off** points on a graph to **minimize the total travel distance** for two users with distinct origins and destinations.

## 🔍 Overview

To efficiently solve the 2TP problem, this project implements a **Genetic Algorithm (GA)** that uses:
- Binary encoding of candidate solutions
- Fitness evaluation based on total travel distance
- Genetic operators such as tournament selection, crossover, and mutation

The approach is benchmarked against an **exact algorithm** to analyze trade-offs between solution quality and computational performance. Experiments are conducted on both **synthetic random graphs** and a **real-world road network**.

## 📌 Features

- Solve the 2TP problem using either a **heuristic (GA)** or an **exact** method
- Support for various experimental modes:
  - `ga`: run the genetic algorithm
  - `benchmark`: evaluate GA vs exact algorithm on multiple instances
  - `timebudget`: compare both methods under equal time constraints
  - `roadnetwork`: run the GA on a real-world road network graph
- Configurable graph size, population size, number of generations, and random seed
- Easily extensible for additional experiments or network models

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Install required dependencies (if any are used; otherwise remove this)

```bash
pip install -r requirements.txt
```

### Usage
You can run the main script with various modes:
```bash
python main.py -m <mode> [options]
```

Available Modes and Options

| Argument           | Description                                         | Default     |
|--------------------|-----------------------------------------------------|-------------|
| `-m`, `--mode`     | Mode: `ga`, `benchmark`, `timebudget`, `roadnetwork` | `ga`        |
| `-s`, `--size`     | Graph size (number of nodes)                       | `100`       |
| `-p`, `--population` | Population size for the GA                        | `10`        |
| `-g`, `--generations` | Number of generations for GA                    | `250`       |
| `-i`, `--instances` | Number of instances for benchmarking               | `10`        |
| `-f`, `--file`     | JSON file for the road network                      | `MON.json`  |
| `--seed`           | Random seed for reproducibility                     | `66`        |

### Examples
Run the GA on a 100-node random graph:
```bash
python main.py -m ga -s 100 -p 20 -g 300
```
Run benchmarking over 10 instances:
```bash
python main.py -m benchmark -s 80 -i 10
```
Run the GA on a real-world network:
```bash
python main.py -m roadnetwork -f MON.json
```

## 📊 Results
Our experiments show that:

- The GA can find near-optimal solutions for the 2TP problem on random graphs.
- It scales well compared to the exact algorithm, particularly for large instances.
- In real-world networks, the GA provides practical and efficient solutions.