# EXTREME: Advanced Graph + DP Combinations

from collections import defaultdict, deque
import heapq
import sys
sys.setrecursionlimit(5000)

# HARD: Maximum Flow - Edmonds-Karp (BFS-based)
def max_flow(n, edges, source, sink):
    """Maximum flow using Edmonds-Karp algorithm."""
    # Build adjacency list with capacities
    capacity = [[0] * n for _ in range(n)]
    for u, v, c in edges:
        capacity[u][v] += c

    def bfs():
        parent = [-1] * n
        visited = [False] * n
        visited[source] = True
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v in range(n):
                if not visited[v] and capacity[u][v] > 0:
                    visited[v] = True
                    parent[v] = u
                    if v == sink:
                        return parent
                    queue.append(v)
        return None

    max_flow_value = 0
    while True:
        parent = bfs()
        if parent is None:
            break
        # Find min capacity along path
        path_flow = float('inf')
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, capacity[u][v])
            v = u
        # Update capacities
        v = sink
        while v != source:
            u = parent[v]
            capacity[u][v] -= path_flow
            capacity[v][u] += path_flow
            v = u
        max_flow_value += path_flow

    return max_flow_value

# HARD: Bipartite Matching
def max_bipartite_matching(adj, n, m):
    """Maximum matching in bipartite graph."""
    match_r = [-1] * m

    def dfs(u, visited):
        for v in adj.get(u, []):
            if visited[v]:
                continue
            visited[v] = True
            if match_r[v] == -1 or dfs(match_r[v], visited):
                match_r[v] = u
                return True
        return False

    matching = 0
    for u in range(n):
        visited = [False] * m
        if dfs(u, visited):
            matching += 1

    return matching

# HARD: Strongly Connected Components (iterative Kosaraju's)
def scc_kosaraju(n, edges):
    """Find strongly connected components."""
    graph = [[] for _ in range(n)]
    reverse_graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
        reverse_graph[v].append(u)

    # First pass - get finish order
    visited = [False] * n
    order = []
    for start in range(n):
        if visited[start]:
            continue
        stack = [(start, 0)]
        while stack:
            node, idx = stack.pop()
            if idx == 0:
                if visited[node]:
                    continue
                visited[node] = True
                stack.append((node, 1))
                for neighbor in graph[node]:
                    if not visited[neighbor]:
                        stack.append((neighbor, 0))
            else:
                order.append(node)

    # Second pass on reverse graph
    visited = [False] * n
    components = []
    for start in reversed(order):
        if visited[start]:
            continue
        component = []
        stack = [start]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            component.append(node)
            for neighbor in reverse_graph[node]:
                if not visited[neighbor]:
                    stack.append(neighbor)
        components.append(sorted(component))

    return sorted(components)

# HARD: Articulation Points
def find_articulation_points(n, edges):
    """Find articulation points in graph."""
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    disc = [0] * n
    low = [0] * n
    parent = [-1] * n
    visited = [False] * n
    ap = set()
    timer = [1]

    def dfs(u):
        visited[u] = True
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        children = 0
        for v in graph[u]:
            if not visited[v]:
                children += 1
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])
                if parent[u] == -1 and children > 1:
                    ap.add(u)
                if parent[u] != -1 and low[v] >= disc[u]:
                    ap.add(u)
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    for i in range(n):
        if not visited[i]:
            dfs(i)

    return sorted(ap)

# HARD: Bridges in Graph
def find_bridges(n, edges):
    """Find bridges in graph."""
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    disc = [0] * n
    low = [0] * n
    parent = [-1] * n
    visited = [False] * n
    bridges = []
    timer = [1]

    def dfs(u):
        visited[u] = True
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        for v in graph[u]:
            if not visited[v]:
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.append(tuple(sorted([u, v])))
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    for i in range(n):
        if not visited[i]:
            dfs(i)

    return sorted(bridges)

# HARD: Minimum Spanning Tree with Kruskal
def mst_kruskal(n, edges):
    """Minimum spanning tree using Kruskal's algorithm."""
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        parent[py] = px
        return True

    edges = sorted(edges, key=lambda x: x[2])
    mst_weight = 0

    for u, v, w in edges:
        if union(u, v):
            mst_weight += w

    return mst_weight

# HARD: Dijkstra with negative edge detection
def dijkstra(n, edges, start):
    """Single source shortest paths."""
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))

    dist = [float('inf')] * n
    dist[start] = 0
    heap = [(0, start)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))

    return dist

# HARD: Bellman-Ford with negative cycle detection
def bellman_ford(n, edges, start):
    """Single source shortest paths with negative cycles."""
    dist = [float('inf')] * n
    dist[start] = 0

    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    # Check for negative cycle
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            return None  # Negative cycle

    return dist

# Tests
tests = []

# Max flow
tests.append(("max_flow", max_flow(6, [(0,1,16),(0,2,13),(1,2,10),(1,3,12),(2,1,4),(2,4,14),(3,2,9),(3,5,20),(4,3,7),(4,5,4)], 0, 5), 23))

# Bipartite matching
bip_graph = {0: [0, 1], 1: [0], 2: [1, 2]}
tests.append(("bipartite", max_bipartite_matching(bip_graph, 3, 3), 3))

# SCC
tests.append(("scc", scc_kosaraju(5, [(0,1),(1,2),(2,0),(1,3),(3,4)]),
              [[0,1,2], [3], [4]]))

# Articulation points
tests.append(("ap", find_articulation_points(5, [(0,1),(0,2),(1,2),(1,3),(3,4)]), [1, 3]))

# Bridges
tests.append(("bridges", find_bridges(5, [(0,1),(0,2),(1,2),(1,3),(3,4)]), [(1,3), (3,4)]))

# MST
tests.append(("mst", mst_kruskal(4, [(0,1,10),(0,2,6),(0,3,5),(1,3,15),(2,3,4)]), 19))

# Dijkstra
tests.append(("dijkstra", dijkstra(5, [(0,1,10),(0,2,3),(1,2,1),(1,3,2),(2,1,4),(2,3,8),(2,4,2),(3,4,7),(4,3,9)], 0),
              [0, 7, 3, 9, 5]))

# Bellman-Ford
tests.append(("bellman", bellman_ford(5, [(0,1,6),(0,2,7),(1,2,8),(1,3,5),(1,4,-4),(2,3,-3),(2,4,9),(3,1,-2),(4,0,2),(4,3,7)], 0),
              [0, 2, 7, 4, -2]))

# Bellman-Ford negative cycle
tests.append(("neg_cycle", bellman_ford(3, [(0,1,1),(1,2,-1),(2,0,-1)], 0), None))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
