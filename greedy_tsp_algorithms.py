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


def run_and_time(flag, mat, repeats_if_zero_cpu=20000):
    if flag == 0:
        t0 = time.time_ns()
        c0 = time.process_time_ns()
        path, cost = nearest_neighbor(mat, startVertex=0)
        c1 = time.process_time_ns()
        t1 = time.time_ns()
    if flag == 1:
        t0 = time.time_ns()
        c0 = time.process_time_ns()
        path, cost = two_opt_adjacent(mat)
        c1 = time.process_time_ns()
        t1 = time.time_ns()
    if flag == 2:
        repeats_if_zero_cpu = 100
        t0 = time.time_ns()
        c0 = time.process_time_ns()
        path, cost = rrnn(mat, k_best, r_best, startVertex=0)
        c1 = time.process_time_ns()
        t1 = time.time_ns()

    real_ns = t1 - t0
    cpu_ns = c1 - c0

    if cpu_ns == 0:
        c0 = time.process_time_ns()
        for _ in range(repeats_if_zero_cpu):
            if flag == 0:
                path, cost = nearest_neighbor(mat, startVertex=0)
            if flag == 1:
                path, cost = two_opt_adjacent(mat)
            if flag == 2:
                path, cost = rrnn(mat, k_best, r_best, startVertex=0)
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


# NN
def nearest_neighbor(matrix, startVertex=0):
    n = len(matrix)
    current = startVertex

    visited = [False] * n
    visited[current] = True

    path = [current]

    for _ in range(n - 1):
        best_city = -1
        best_dist = float("inf")

        for j in range(n):
            if not visited[j]:
                dist = matrix[current][j]
                if dist < best_dist:
                    best_dist = dist
                    best_city = j

        visited[best_city] = True
        current = best_city
        path.append(current)

    path.append(startVertex)

    cost = path_cost(matrix, path)
    return path, cost


# NN + 2-Opt Adjacent
def two_opt_adjacent(matrix, path=None):
    if path is None:
        path, _ = nearest_neighbor(matrix)
    best_path = []
    for city in path:
        best_path.append(city)

    best_cost = path_cost(matrix, best_path)

    improved = True
    while improved:
        improved = False

        for i in range(1, len(best_path) - 2):
            candidate = list(best_path)
            temp = candidate[i]
            candidate[i] = candidate[i + 1]
            candidate[i + 1] = temp

            candidate_cost = path_cost(matrix, candidate)
            if candidate_cost < best_cost:
                best_path = candidate
                best_cost = candidate_cost
                improved = True
                break

    return best_path, best_cost


# RRNN
def rrnn(matrix, k, num_repeats, startVertex=0):
    n = len(matrix)

    best_overall_path = None
    best_overall_cost = float("inf")

    for repeat in range(num_repeats):
        visited = [False] * n
        current = startVertex
        visited[current] = True

        path = [startVertex]

        for p in range(n - 1):
            candidates = []
            candidates_with_cost = []
            for city in range(n):
                if not visited[city]:
                    candidates.append(city)
            for city in candidates:
                candidates_with_cost.append((matrix[current][city], city))

            candidates_with_cost.sort()
            top_k = candidates_with_cost[:k]
            next_city = random.choice(top_k)[1]

            visited[next_city] = True
            path.append(next_city)
            current = next_city

        path.append(startVertex)

        path, cost = two_opt_adjacent(matrix, path)

        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall_path = path[:]

    return best_overall_path, best_overall_cost


if __name__ == "__main__":
    ensure_unzipped()

    if len(sys.argv) < 3:
        print(
            "Usage: python 421_project1_p1.py <NN|NN2OPT|RRNN> <matrix_file> [k] [repeats]"
        )
        sys.exit(1)

    alg = sys.argv[1].upper()
    matrix_file = sys.argv[2]

    mat = load_matrix_from_file(matrix_file)

    if alg == "NN":
        flag = 0
    elif alg == "NN2OPT":
        flag = 1
    elif alg == "RRNN":
        flag = 2
        k_best = int(sys.argv[3]) if len(sys.argv) >= 4 else 2
        r_best = int(sys.argv[4]) if len(sys.argv) >= 5 else 100
    else:
        print("Unknown algorithm:", alg)
        sys.exit(1)

    real_ns, cpu_ns, cost = run_and_time(flag, mat)

    print("real_ns:", real_ns)
    print("cpu_ns:", cpu_ns)
    print("cost:", cost)
