def max_flow_ford_fulkerson(n, edges, source, sink):
    """Maximum flow using Ford-Fulkerson with BFS (Edmonds-Karp)."""
    from collections import defaultdict, deque

    graph = defaultdict(lambda: defaultdict(int))
    for u, v, cap in edges:
        graph[u][v] += cap

    def bfs(source, sink, parent):
        visited = {source}
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v in graph[u]:
                if v not in visited and graph[u][v] > 0:
                    visited.add(v)
                    parent[v] = u
                    if v == sink:
                        return True
                    queue.append(v)
        return False

    max_flow = 0
    parent = {}
    while bfs(source, sink, parent):
        # Find minimum residual capacity
        path_flow = float('inf')
        s = sink
        while s != source:
            path_flow = min(path_flow, graph[parent[s]][s])
            s = parent[s]
        # Update residual capacities
        v = sink
        while v != source:
            u = parent[v]
            graph[u][v] -= path_flow
            graph[v][u] += path_flow
            v = parent[v]
        max_flow += path_flow
        parent = {}
    return max_flow

def min_cut(n, edges, source, sink):
    """Minimum cut using max flow theorem."""
    from collections import defaultdict, deque

    graph = defaultdict(lambda: defaultdict(int))
    original = defaultdict(lambda: defaultdict(int))
    for u, v, cap in edges:
        graph[u][v] += cap
        original[u][v] += cap

    def bfs(source, sink, parent):
        visited = {source}
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v in graph[u]:
                if v not in visited and graph[u][v] > 0:
                    visited.add(v)
                    parent[v] = u
                    if v == sink:
                        return True
                    queue.append(v)
        return False

    parent = {}
    while bfs(source, sink, parent):
        path_flow = float('inf')
        s = sink
        while s != source:
            path_flow = min(path_flow, graph[parent[s]][s])
            s = parent[s]
        v = sink
        while v != source:
            u = parent[v]
            graph[u][v] -= path_flow
            graph[v][u] += path_flow
            v = parent[v]
        parent = {}

    # Find reachable nodes from source
    visited = {source}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if v not in visited and graph[u][v] > 0:
                visited.add(v)
                queue.append(v)

    # Find cut edges
    cut_edges = []
    for u in visited:
        for v in original[u]:
            if v not in visited and original[u][v] > 0:
                cut_edges.append((u, v, original[u][v]))
    return cut_edges

def bipartite_matching(n, m, edges):
    """Maximum bipartite matching using Hopcroft-Karp-like approach."""
    from collections import defaultdict

    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)

    match_left = {}
    match_right = {}

    def dfs(u, visited):
        for v in graph[u]:
            if v in visited:
                continue
            visited.add(v)
            if v not in match_right or dfs(match_right[v], visited):
                match_left[u] = v
                match_right[v] = u
                return True
        return False

    matching = 0
    for u in range(n):
        if u not in match_left:
            if dfs(u, set()):
                matching += 1
    return matching

def hungarian_algorithm(cost_matrix):
    """Hungarian algorithm for assignment problem (simplified)."""
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return sum(cost_matrix[row_ind[i]][col_ind[i]] for i in range(len(row_ind)))

def minimum_vertex_cover_tree(n, edges):
    """Minimum vertex cover in a tree using DP."""
    from collections import defaultdict

    if n == 0:
        return 0

    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    # dp[v][0] = min cover not including v
    # dp[v][1] = min cover including v
    dp = [[0, 0] for _ in range(n)]
    visited = [False] * n

    def dfs(u, parent):
        visited[u] = True
        dp[u][0] = 0
        dp[u][1] = 1
        for v in graph[u]:
            if v != parent and not visited[v]:
                dfs(v, u)
                dp[u][0] += dp[v][1]  # u not covered, v must be
                dp[u][1] += min(dp[v][0], dp[v][1])  # u covered, v can be either

    dfs(0, -1)
    return min(dp[0][0], dp[0][1])

def maximum_independent_set_tree(n, edges):
    """Maximum independent set in a tree."""
    from collections import defaultdict

    if n == 0:
        return 0

    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    # dp[v][0] = max IS not including v
    # dp[v][1] = max IS including v
    dp = [[0, 0] for _ in range(n)]
    visited = [False] * n

    def dfs(u, parent):
        visited[u] = True
        dp[u][1] = 1
        for v in graph[u]:
            if v != parent and not visited[v]:
                dfs(v, u)
                dp[u][0] += max(dp[v][0], dp[v][1])
                dp[u][1] += dp[v][0]  # v not included

    dfs(0, -1)
    return max(dp[0][0], dp[0][1])

def traveling_salesman_dp(dist):
    """TSP using dynamic programming with bitmask."""
    n = len(dist)
    INF = float('inf')
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0  # Start at node 0

    for mask in range(1 << n):
        for u in range(n):
            if not (mask & (1 << u)):
                continue
            for v in range(n):
                if mask & (1 << v):
                    continue
                new_mask = mask | (1 << v)
                dp[new_mask][v] = min(dp[new_mask][v], dp[mask][u] + dist[u][v])

    # Return to start
    full_mask = (1 << n) - 1
    return min(dp[full_mask][u] + dist[u][0] for u in range(n))

# Tests
tests = []

# Max flow test
tests.append(("max_flow", max_flow_ford_fulkerson(6, [(0,1,16),(0,2,13),(1,2,10),(1,3,12),(2,1,4),(2,4,14),(3,2,9),(3,5,20),(4,3,7),(4,5,4)], 0, 5), 23))

# Bipartite matching test
tests.append(("bipartite_match", bipartite_matching(3, 3, [(0,0),(0,1),(1,0),(1,2),(2,1)]), 3))

# Minimum vertex cover tree
tests.append(("min_vertex_cover", minimum_vertex_cover_tree(4, [(0,1),(0,2),(1,3)]), 1))

# Maximum independent set tree
tests.append(("max_ind_set", maximum_independent_set_tree(5, [(0,1),(0,2),(1,3),(1,4)]), 3))

# TSP test
tsp_dist = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]
tests.append(("tsp", traveling_salesman_dp(tsp_dist), 80))

# Min cut test
cuts = min_cut(4, [(0,1,3),(0,2,1),(1,2,1),(1,3,3),(2,3,5)], 0, 3)
total_cut = sum(c for _, _, c in cuts)
tests.append(("min_cut", total_cut, 4))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
