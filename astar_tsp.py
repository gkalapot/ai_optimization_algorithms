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


def run_and_time(flag, mat, repeats_if_zero_cpu=50):
    if flag == 3:
        t0 = time.time_ns()
        c0 = time.process_time_ns()
        path, cost, expanded = a_star_alg(mat, startVertex=0)
        c1 = time.process_time_ns()
        t1 = time.time_ns()

    real_ns = t1 - t0
    cpu_ns = c1 - c0

    if cpu_ns == 0:
        c0 = time.process_time_ns()
        for _ in range(repeats_if_zero_cpu):
            if flag == 3:
                path, cost, expanded = a_star_alg(mat, startVertex=0)
        c1 = time.process_time_ns()
        cpu_ns = (c1 - c0) / repeats_if_zero_cpu

    return real_ns, cpu_ns, cost, expanded


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


# A*
def prim_mst_cost(matrix, nodes):
    if len(nodes) <= 1:
        return 0.0

    in_mst = [False] * len(nodes)
    min_edge = [float("inf")] * len(nodes)

    in_mst[0] = True
    for i in range(1, len(nodes)):
        min_edge[i] = matrix[nodes[0]][nodes[i]]

    total = 0.0

    for _ in range(len(nodes) - 1):
        best_i = -1
        best_w = float("inf")

        for i in range(len(nodes)):
            if (not in_mst[i]) and min_edge[i] < best_w:
                best_w = min_edge[i]
                best_i = i

        in_mst[best_i] = True
        total = total + best_w

        for j in range(len(nodes)):
            if not in_mst[j]:
                w = matrix[nodes[best_i]][nodes[j]]
                if w < min_edge[j]:
                    min_edge[j] = w

    return total


def prims_heuristic(matrix, current, start, path):
    n = len(matrix)
    path_set = set(path)
    unvisited = []
    for city in range(n):
        if city not in path_set:
            unvisited.append(city)

    if len(unvisited) == 0:
        return matrix[current][start]

    mst = prim_mst_cost(matrix, unvisited)

    best_from_current = float("inf")
    for i in unvisited:
        if matrix[current][i] < best_from_current:
            best_from_current = matrix[current][i]

    best_to_start = float("inf")
    for i in unvisited:
        if matrix[i][start] < best_to_start:
            best_to_start = matrix[i][start]

    return mst + best_from_current + best_to_start


def a_star_alg(matrix, startVertex=0):
    n = len(matrix)
    pq = []
    best_g = {}

    start_path = [startVertex]
    start_g = 0.0
    start_h = prims_heuristic(matrix, startVertex, startVertex, start_path)
    start_f = start_g + start_h

    heapq.heappush(pq, (start_f, start_g, startVertex, start_path))
    best_g[(startVertex, tuple(start_path))] = start_g
    nodes_expanded = 0

    while pq:
        f, g, currVertex, path = heapq.heappop(pq)

        state = (currVertex, tuple(path))
        if state in best_g and g > best_g[state]:
            continue

        nodes_expanded = nodes_expanded + 1

        if len(path) == n:
            complete_path = path + [startVertex]
            total_cost = g + matrix[currVertex][startVertex]
            return complete_path, total_cost, nodes_expanded

        for nxt in range(n):
            path_set = set(path)
            if nxt in path_set:
                continue

            new_g = g + matrix[currVertex][nxt]
            new_path = path + [nxt]

            h = prims_heuristic(matrix, nxt, startVertex, new_path)
            new_f = new_g + h

            new_state = (nxt, tuple(new_path))
            if new_state not in best_g or new_g < best_g[new_state]:
                best_g[new_state] = new_g
                heapq.heappush(pq, (new_f, new_g, nxt, new_path))

    return None, float("inf"), nodes_expanded


if __name__ == "__main__":
    ensure_unzipped()

    if len(sys.argv) < 3:
        print("Usage: python astar_tsp.py <NN|NN2OPT|RRNN> <matrix_file> [k] [repeats]")
        sys.exit(1)

    alg = sys.argv[1].upper()
    matrix_file = sys.argv[2]
    mat = load_matrix_from_file(matrix_file)

    if alg == "ASTAR":
        flag = 3
    else:
        print("Unknown algorithm:", alg)
        sys.exit(1)

    real_ns, cpu_ns, cost, expanded = run_and_time(flag, mat)

    print("real_ns:", real_ns)
    print("cpu_ns:", cpu_ns)
    print("cost:", cost)
    print("nodes_expanded:", expanded)
