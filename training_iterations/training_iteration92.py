# EXTREME: Network Flow & Matching Algorithms

from collections import defaultdict, deque

# HARD: Hopcroft-Karp Maximum Bipartite Matching
def hopcroft_karp(graph, n, m):
    """Maximum bipartite matching using Hopcroft-Karp O(E*sqrt(V))."""
    INF = float('inf')
    match_l = [-1] * n  # Match for left vertices
    match_r = [-1] * m  # Match for right vertices
    dist = [0] * n

    def bfs():
        queue = deque()
        for u in range(n):
            if match_l[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = INF

        found = False
        while queue:
            u = queue.popleft()
            for v in graph.get(u, []):
                if match_r[v] == -1:
                    found = True
                elif dist[match_r[v]] == INF:
                    dist[match_r[v]] = dist[u] + 1
                    queue.append(match_r[v])
        return found

    def dfs(u):
        for v in graph.get(u, []):
            if match_r[v] == -1 or (dist[match_r[v]] == dist[u] + 1 and dfs(match_r[v])):
                match_l[u] = v
                match_r[v] = u
                return True
        dist[u] = INF
        return False

    matching = 0
    while bfs():
        for u in range(n):
            if match_l[u] == -1 and dfs(u):
                matching += 1

    return matching, [(u, match_l[u]) for u in range(n) if match_l[u] != -1]

# HARD: Hungarian Algorithm (Minimum Cost Bipartite Matching)
def hungarian(cost_matrix):
    """Find minimum cost perfect matching."""
    n = len(cost_matrix)
    m = len(cost_matrix[0])
    INF = float('inf')

    # Pad to make square
    size = max(n, m)
    cost = [[0] * size for _ in range(size)]
    for i in range(n):
        for j in range(m):
            cost[i][j] = cost_matrix[i][j]

    u = [0] * (size + 1)
    v = [0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)

    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (size + 1)
        used = [False] * (size + 1)

        while p[j0] != 0:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0

            for j in range(1, size + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1

        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    result = [0] * size
    for j in range(1, size + 1):
        if p[j] != 0:
            result[p[j] - 1] = j - 1

    total_cost = sum(cost_matrix[i][result[i]] for i in range(n) if result[i] < m)
    return total_cost, result[:n]

# HARD: Ford-Fulkerson with DFS
def ford_fulkerson_dfs(capacity, source, sink):
    """Maximum flow using Ford-Fulkerson with DFS."""
    n = len(capacity)
    residual = [row[:] for row in capacity]

    def dfs(s, t, visited, min_cap):
        if s == t:
            return min_cap
        visited.add(s)
        for v in range(n):
            if v not in visited and residual[s][v] > 0:
                flow = dfs(v, t, visited, min(min_cap, residual[s][v]))
                if flow > 0:
                    residual[s][v] -= flow
                    residual[v][s] += flow
                    return flow
        return 0

    max_flow = 0
    while True:
        flow = dfs(source, sink, set(), float('inf'))
        if flow == 0:
            break
        max_flow += flow

    return max_flow

# HARD: Min-Cut from Max-Flow
def min_cut(capacity, source, sink):
    """Find minimum cut edges."""
    n = len(capacity)
    residual = [row[:] for row in capacity]

    # Run max flow first
    def bfs_augment():
        parent = [-1] * n
        visited = [False] * n
        visited[source] = True
        queue = deque([source])

        while queue:
            u = queue.popleft()
            for v in range(n):
                if not visited[v] and residual[u][v] > 0:
                    visited[v] = True
                    parent[v] = u
                    if v == sink:
                        return parent
                    queue.append(v)
        return None

    while True:
        parent = bfs_augment()
        if not parent:
            break
        # Find min capacity
        path_flow = float('inf')
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, residual[u][v])
            v = u
        # Update residual
        v = sink
        while v != source:
            u = parent[v]
            residual[u][v] -= path_flow
            residual[v][u] += path_flow
            v = u

    # Find reachable from source
    visited = [False] * n
    queue = deque([source])
    visited[source] = True
    while queue:
        u = queue.popleft()
        for v in range(n):
            if not visited[v] and residual[u][v] > 0:
                visited[v] = True
                queue.append(v)

    # Min cut edges
    cut_edges = []
    for u in range(n):
        for v in range(n):
            if visited[u] and not visited[v] and capacity[u][v] > 0:
                cut_edges.append((u, v))

    return cut_edges

# HARD: Maximum Weighted Bipartite Matching (Kuhn-Munkres alternative)
def max_weight_matching(weights):
    """Maximum weight bipartite matching."""
    n = len(weights)
    m = len(weights[0]) if n > 0 else 0

    # Negate weights for Hungarian
    max_val = max(max(row) for row in weights) if weights else 0
    neg_weights = [[max_val - weights[i][j] for j in range(m)] for i in range(n)]

    cost, assignment = hungarian(neg_weights)
    total_weight = sum(weights[i][assignment[i]] for i in range(n) if assignment[i] < m)

    return total_weight, assignment

# Tests
tests = []

# Hopcroft-Karp
graph = {0: [0, 1], 1: [0], 2: [1, 2], 3: [2]}
matching, pairs = hopcroft_karp(graph, 4, 3)
tests.append(("hk_matching", matching, 3))

# Hungarian Algorithm
cost = [
    [3, 5, 10, 1],
    [2, 8, 7, 11],
    [4, 6, 1, 8],
    [5, 7, 3, 4]
]
total, assignment = hungarian(cost)
tests.append(("hungarian", total, 11))  # 1+2+1+7=11 or similar

# Ford-Fulkerson
capacity = [
    [0, 16, 13, 0, 0, 0],
    [0, 0, 10, 12, 0, 0],
    [0, 4, 0, 0, 14, 0],
    [0, 0, 9, 0, 0, 20],
    [0, 0, 0, 7, 0, 4],
    [0, 0, 0, 0, 0, 0]
]
tests.append(("ff_flow", ford_fulkerson_dfs(capacity, 0, 5), 23))

# Min Cut
cut = min_cut(capacity, 0, 5)
tests.append(("min_cut", len(cut) >= 1, True))

# Max Weight Matching
weights = [
    [3, 2, 7],
    [8, 6, 4],
    [5, 9, 1]
]
max_w, _ = max_weight_matching(weights)
tests.append(("max_weight", max_w, 21))  # 7+8+9=24 or 3+8+9=20... depends on matching

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
