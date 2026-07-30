# Traveling Salesman Problem (TSP) Optimization Algorithms
Author: Georgia Kalapotharakou

This repository contains Python implementations of several algorithms for solving or approximating the TSP. The project compares greedy methods, exact search, local search, randomized search, and evolutionary search on distance matrices.

## Algorithms Included

- Nearest Neighbor
- Nearest Neighbor with 2-Opt
- Random-Restart Nearest Neighbor
- A* Search with a Prim/MST-based heuristic
- Hill Climbing
- Simulated Annealing
- Genetic Algorithm

## Files

- `greedy_tsp_algorithms.py` - nearest neighbor, nearest neighbor with 2-opt, and random-restart nearest neighbor
- `astar_tsp.py` - A* search using a Prim/MST-based heuristic
- `local_search_tsp_algorithms.py` - hill climbing, simulated annealing, and genetic algorithm

## Input Format

Each algorithm expects a matrix file where each row contains space-separated numeric distances. An example of an input matrix is given in example/example_matrix.txt.

## How to Run
### greedy_tsp_algorithms.py
- python greedy_tsp_algorithms.py NN example/example_matrix.txt
- python greedy_tsp_algorithms.py NN2OPT example/example_matrix.txt
- python greedy_tsp_algorithms.py RRNN example/example_matrix.txt 2 100
### astar_tsp.py
- python astar_tsp.py ASTAR example/example_matrix.txt
### local_search_tsp_algorithms.py
- python local_search_tsp_algorithms.py HC example/example_matrix.txt
- python local_search_tsp_algorithms.py SA example/example_matrix.txt
- python local_search_tsp_algorithms.py GA example/example_matrix.txt

## Output
Each script prints timing information and the final tour cost. The A* script also prints the number of expanded nodes.
