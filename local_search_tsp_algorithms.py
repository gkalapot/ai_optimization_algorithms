import os
import numpy as np
import random
import statistics
import time
import heapq
import math
from operator import itemgetter
import zipfile
import os
import sys


def ensure_unzipped(zip_path="matrices.zip", extract_dir="matrices"):
    if os.path.exists(extract_dir):
        return
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)


def run_and_time(
    flag,
    mat,
    num_repeats=1000,
    alpha=0.995,
    initial_temp=50.0,
    max_iterations=5000,
    mutation_chance=0.03,
    population_size=100,
    num_generations=900,
    repeats_if_zero_cpu=50,
):

    t0 = time.time_ns()
    c0 = time.process_time_ns()

    if flag == 0:
        path, cost = hill_climb(mat, num_repeats)
    elif flag == 1:
        path, cost = simulated_annealing(mat, alpha, initial_temp, max_iterations)
    elif flag == 2:
        repeats_if_zero_cpu = 5
        path, cost = geneticAlgorithm(
            mat, mutation_chance, population_size, num_generations, start=0
        )

    c1 = time.process_time_ns()
    t1 = time.time_ns()

    real_ns = t1 - t0
    cpu_ns = c1 - c0

    if cpu_ns == 0:
        c0 = time.process_time_ns()
        for _ in range(repeats_if_zero_cpu):
            if flag == 0:
                path, cost = hill_climb(mat, num_repeats)
            elif flag == 1:
                path, cost = simulated_annealing(
                    mat, alpha, initial_temp, max_iterations
                )
            elif flag == 2:
                path, cost = geneticAlgorithm(
                    mat, mutation_chance, population_size, num_generations, start=0
                )
        c1 = time.process_time_ns()
        cpu_ns = (c1 - c0) / repeats_if_zero_cpu

    return real_ns, cpu_ns, cost


def load_matrix_from_file(filepath):
    matrix = []
    with open(filepath, "r") as f:
        for line in f:
            matrix.append([float(x) for x in line.strip().split()])
    return matrix


def path_cost(matrix, path):
    total = 0.0
    for i in range(len(path) - 1):
        total = total + matrix[path[i]][path[i + 1]]
    return total


# Hill climbing
def swap_two_interior(path):
    arr = list(path)
    i = random.randint(1, len(arr) - 2)
    j = random.randint(1, len(arr) - 2)
    while j == i:
        j = random.randint(1, len(arr) - 2)

    tmp = arr[i]
    arr[i] = arr[j]
    arr[j] = tmp
    return arr


def random_tour(matrix, startVertex=0):
    n = len(matrix)
    cities = list(range(n))
    cities.remove(startVertex)
    random.shuffle(cities)
    return [startVertex] + cities + [startVertex]


def hill_climb(matrix, num_restarts):
    best_overall_soln = None
    best_overall_cost = float("inf")

    for _ in range(num_restarts):
        curr_soln = random_tour(matrix, startVertex=0)
        curr_cost = path_cost(matrix, curr_soln)

        improved = True
        while improved:
            improved = False

            for _ in range(50):
                potential_soln = swap_two_interior(curr_soln)
                potential_cost = path_cost(matrix, potential_soln)

                if potential_cost < curr_cost:
                    curr_soln = potential_soln
                    curr_cost = potential_cost
                    improved = True
                    break

        if curr_cost < best_overall_cost:
            best_overall_soln = curr_soln
            best_overall_cost = curr_cost

    return best_overall_soln, best_overall_cost


# Simulated Annealing
def simulated_annealing(matrix, alpha, initial_temp, max_iterations):
    curr_soln = random_tour(matrix, startVertex=0)
    curr_cost = path_cost(matrix, curr_soln)

    best_soln = list(curr_soln)
    best_cost = curr_cost
    t = initial_temp

    for _ in range(max_iterations):
        new_soln = swap_two_interior(curr_soln)
        new_cost = path_cost(matrix, new_soln)

        if new_cost < curr_cost:
            curr_soln = new_soln
            curr_cost = new_cost
            t = alpha * t
        else:
            if t <= 0:
                prob = 0
            else:
                prob = math.exp((curr_cost - new_cost) / t)

            r = random.random()
            if r < prob:
                curr_soln = new_soln
                curr_cost = new_cost
                t = alpha * t

        if curr_cost < best_cost:
            best_soln = list(curr_soln)
            best_cost = curr_cost

    return best_soln, best_cost


# Genetic Algorithm + helpers
def resolve(gene, seg_set, pos_in_p1_seg, p2):
    while gene in seg_set:
        idx = pos_in_p1_seg[gene]
        gene = p2[idx]
    return gene


def pmx_crossover(parent1, parent2, start=0):
    p1 = parent1.copy()
    p2 = parent2.copy()

    p1.pop(0)
    p1.pop(-1)

    p2.pop(0)
    p2.pop(-1)
    n = len(p1)

    x = random.randint(0, n - 1)
    y = random.randint(0, n - 1)

    while x == y:
        y = random.randint(0, n - 1)

    a = min(x, y)
    b = max(x, y)

    child = [None] * n
    child[a : b + 1] = p1[a : b + 1]

    segment_already_in_child = p1[a : b + 1]
    seg_set = set(segment_already_in_child)

    pos_in_p1_seg = {}

    for i in range(a, b + 1):
        gene = p1[i]
        pos_in_p1_seg[gene] = i

    for i in range(n):
        if a <= i <= b:
            continue
        gene = p2[i]
        child[i] = resolve(gene, seg_set, pos_in_p1_seg, p2)

    return [start] + child + [start]


def mutate(path):
    arr = list(path)
    i = random.randint(1, len(arr) - 2)
    j = random.randint(1, len(arr) - 2)
    while j == i:
        j = random.randint(1, len(arr) - 2)
    arr[i], arr[j] = arr[j], arr[i]
    return arr


def geneticAlgorithm(
    matrix, mutation_chance, population_size, num_generations, start=0
):
    n = len(matrix)
    population = []
    for _ in range(population_size):
        population.append(random_tour(matrix, startVertex=0))

    for _ in range(num_generations):
        scored = []
        for path in population:
            scored.append((path_cost(matrix, path), path))
        scored.sort(key=itemgetter(0))

        if population_size // 2 < 2:
            half = 2
        else:
            half = population_size // 2
        parents = []
        for i in range(half):
            parents.append(scored[i][1])

        children = []
        while len(children) < population_size:
            p1 = random.choice(parents)
            p2 = random.choice(parents)

            child = pmx_crossover(p1, p2, start=0)

            if random.random() < mutation_chance:
                child = mutate(child)

            children.append(child)

        combined = population + children
        combined_scored = []
        for t in combined:
            combined_scored.append((path_cost(matrix, t), t))
        combined_scored.sort(key=itemgetter(0))

        population = []
        for i in range(population_size):
            population.append(combined_scored[i][1])

    best_p = population[0]
    best_cost = path_cost(matrix, best_p)

    for i in population:
        c = path_cost(matrix, i)
        if c < best_cost:
            best_cost = c
            best_p = i

    return best_p, best_cost


if __name__ == "__main__":
    ensure_unzipped()

    if len(sys.argv) < 3:
        print("Usage: python local_search_tsp_algorithms.py <NN|NN2OPT|RRNN> <matrix_file> [k] [repeats]")
        sys.exit(1)

    alg = sys.argv[1].upper()
    matrix_file = sys.argv[2]
    mat = load_matrix_from_file(matrix_file)

    if alg == "HC":
        flag = 0
    elif alg == "SA":
        flag = 1
    elif alg == "GA":
        flag = 2
    else:
        print("Unknown algorithm:", alg)
        sys.exit(1)

    real_ns, cpu_ns, cost = run_and_time(flag, mat)

    print("real_ns:", real_ns)
    print("cpu_ns:", cpu_ns)
    print("cost:", cost)
