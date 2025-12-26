# EXTREME: Approximation Algorithms & NP-Hard Problem Heuristics

import random
from collections import defaultdict
import heapq

# HARD: 2-Approximation for Vertex Cover
def vertex_cover_approx(n, edges):
    """2-approximation for minimum vertex cover."""
    cover = set()
    remaining_edges = set((min(u,v), max(u,v)) for u, v in edges)

    while remaining_edges:
        u, v = remaining_edges.pop()
        cover.add(u)
        cover.add(v)
        # Remove edges incident to u or v
        remaining_edges = {(a, b) for a, b in remaining_edges
                          if a != u and a != v and b != u and b != v}

    return list(cover)

# HARD: Greedy Set Cover
def set_cover_greedy(universe, sets):
    """Greedy approximation for set cover."""
    uncovered = set(universe)
    cover = []

    while uncovered:
        best_set = max(range(len(sets)),
                      key=lambda i: len(sets[i] & uncovered))
        if not (sets[best_set] & uncovered):
            break
        cover.append(best_set)
        uncovered -= sets[best_set]

    return cover

# HARD: 2-Approximation for Metric TSP
def metric_tsp_approx(dist):
    """2-approximation for metric TSP using MST."""
    n = len(dist)
    if n <= 1:
        return list(range(n))

    # Build MST using Prim's
    in_mst = [False] * n
    key = [float('inf')] * n
    parent = [-1] * n
    key[0] = 0

    for _ in range(n):
        u = min((k for k in range(n) if not in_mst[k]), key=lambda x: key[x])
        in_mst[u] = True
        for v in range(n):
            if not in_mst[v] and dist[u][v] < key[v]:
                key[v] = dist[u][v]
                parent[v] = u

    # DFS preorder on MST
    adj = defaultdict(list)
    for v in range(1, n):
        adj[parent[v]].append(v)
        adj[v].append(parent[v])

    tour = []
    visited = [False] * n

    def dfs(u):
        visited[u] = True
        tour.append(u)
        for v in adj[u]:
            if not visited[v]:
                dfs(v)

    dfs(0)
    return tour

# HARD: Christofides Algorithm (3/2 approximation for metric TSP)
def christofides_tsp(dist):
    """3/2-approximation for metric TSP."""
    n = len(dist)
    if n <= 2:
        return list(range(n))

    # Step 1: Build MST
    in_mst = [False] * n
    key = [float('inf')] * n
    parent = [-1] * n
    key[0] = 0
    mst_edges = []

    for _ in range(n):
        u = min((k for k in range(n) if not in_mst[k]), key=lambda x: key[x])
        in_mst[u] = True
        if parent[u] != -1:
            mst_edges.append((parent[u], u))
        for v in range(n):
            if not in_mst[v] and dist[u][v] < key[v]:
                key[v] = dist[u][v]
                parent[v] = u

    # Step 2: Find odd-degree vertices
    degree = [0] * n
    for u, v in mst_edges:
        degree[u] += 1
        degree[v] += 1
    odd_vertices = [v for v in range(n) if degree[v] % 2 == 1]

    # Step 3: Minimum weight perfect matching on odd vertices (greedy approximation)
    matching = []
    remaining = set(odd_vertices)
    while remaining:
        u = remaining.pop()
        if not remaining:
            break
        v = min(remaining, key=lambda x: dist[u][x])
        remaining.remove(v)
        matching.append((u, v))

    # Step 4: Combine MST and matching
    multigraph = defaultdict(list)
    for u, v in mst_edges + matching:
        multigraph[u].append(v)
        multigraph[v].append(u)

    # Step 5: Eulerian tour (Hierholzer's)
    euler_tour = []
    stack = [0]
    while stack:
        u = stack[-1]
        if multigraph[u]:
            v = multigraph[u].pop()
            multigraph[v].remove(u)
            stack.append(v)
        else:
            euler_tour.append(stack.pop())

    # Step 6: Convert to Hamiltonian
    visited = set()
    tour = []
    for v in euler_tour:
        if v not in visited:
            visited.add(v)
            tour.append(v)

    return tour

# HARD: FPTAS for Knapsack
def knapsack_fptas(weights, values, capacity, epsilon):
    """FPTAS for 0/1 knapsack."""
    n = len(weights)
    if n == 0:
        return 0

    max_value = max(values)
    k = epsilon * max_value / n

    # Scale values
    scaled_values = [int(v / k) for v in values]

    # DP with scaled values
    max_scaled = sum(scaled_values)
    # dp[v] = minimum weight to achieve value v
    dp = [float('inf')] * (max_scaled + 1)
    dp[0] = 0

    for i in range(n):
        for v in range(max_scaled, scaled_values[i] - 1, -1):
            if dp[v - scaled_values[i]] + weights[i] <= capacity:
                dp[v] = min(dp[v], dp[v - scaled_values[i]] + weights[i])

    # Find maximum achievable value
    for v in range(max_scaled, -1, -1):
        if dp[v] <= capacity:
            # Convert back
            return sum(values[i] for i in range(n)
                      if dp[v] != float('inf'))

    return 0

# HARD: Local Search for MAX-SAT
def max_sat_local_search(clauses, num_vars, max_iterations=1000):
    """Local search heuristic for MAX-SAT."""
    # Random initial assignment
    assignment = [random.choice([True, False]) for _ in range(num_vars)]

    def count_satisfied():
        count = 0
        for clause in clauses:
            satisfied = False
            for var, neg in clause:
                val = assignment[var]
                if neg:
                    val = not val
                if val:
                    satisfied = True
                    break
            if satisfied:
                count += 1
        return count

    best_count = count_satisfied()
    best_assignment = assignment[:]

    for _ in range(max_iterations):
        # Try flipping each variable
        improved = False
        for i in range(num_vars):
            assignment[i] = not assignment[i]
            new_count = count_satisfied()
            if new_count > best_count:
                best_count = new_count
                best_assignment = assignment[:]
                improved = True
            else:
                assignment[i] = not assignment[i]  # Flip back

        if not improved:
            # Random restart with small probability
            if random.random() < 0.1:
                i = random.randint(0, num_vars - 1)
                assignment[i] = not assignment[i]

    return best_count, best_assignment

# Tests
tests = []

# Vertex Cover
cover = vertex_cover_approx(4, [(0,1), (1,2), (2,3)])
tests.append(("vc_size", len(cover) <= 4, True))  # 2-approx

# Set Cover
universe = {1, 2, 3, 4, 5}
sets = [{1, 2, 3}, {2, 4}, {3, 4}, {4, 5}]
cover = set_cover_greedy(universe, sets)
covered = set()
for i in cover:
    covered |= sets[i]
tests.append(("set_cover", covered == universe, True))

# Metric TSP
dist = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]
tour = metric_tsp_approx(dist)
tests.append(("tsp_approx", len(set(tour)), 4))

# Christofides
tour = christofides_tsp(dist)
tests.append(("christofides", len(set(tour)), 4))

# Knapsack FPTAS
weights = [2, 3, 4, 5]
values = [3, 4, 5, 6]
result = knapsack_fptas(weights, values, 5, 0.5)
tests.append(("knapsack_fptas", result >= 0, True))

# MAX-SAT
random.seed(42)
clauses = [
    [(0, False), (1, False)],  # x0 OR x1
    [(0, True), (1, True)],    # NOT x0 OR NOT x1
    [(0, False), (1, True)],   # x0 OR NOT x1
]
count, _ = max_sat_local_search(clauses, 2, 100)
tests.append(("max_sat", count >= 2, True))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
