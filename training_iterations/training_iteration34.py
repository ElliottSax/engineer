def bellman_ford(n, edges, source):
    """Single-source shortest paths with negative edges."""
    dist = [float('inf')] * n
    dist[source] = 0
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    # Check for negative cycles
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            return None  # Negative cycle
    return dist

def floyd_warshall(n, edges):
    """All-pairs shortest paths."""
    dist = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        dist[u][v] = w
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist

def a_star_search(grid, start, end):
    """A* pathfinding on grid."""
    import heapq
    m, n = len(grid), len(grid[0])

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = [(heuristic(start, end), 0, start)]
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current_g, current = heapq.heappop(open_set)
        if current == end:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = current[0] + dr, current[1] + dc
            if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] != 1:
                neighbor = (nr, nc)
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
    return None

def topological_sort_kahn(n, edges):
    """Topological sort using Kahn's algorithm."""
    from collections import defaultdict, deque
    graph = defaultdict(list)
    in_degree = [0] * n
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    queue = deque([i for i in range(n) if in_degree[i] == 0])
    result = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return result if len(result) == n else []

def strongly_connected_components(n, edges):
    """Finds SCCs using Kosaraju's algorithm."""
    from collections import defaultdict
    graph = defaultdict(list)
    reverse_graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        reverse_graph[v].append(u)

    # First DFS to get finish order
    visited = [False] * n
    stack = []

    def dfs1(node):
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                dfs1(neighbor)
        stack.append(node)

    for i in range(n):
        if not visited[i]:
            dfs1(i)

    # Second DFS on reverse graph
    visited = [False] * n
    sccs = []

    def dfs2(node, scc):
        visited[node] = True
        scc.append(node)
        for neighbor in reverse_graph[node]:
            if not visited[neighbor]:
                dfs2(neighbor, scc)

    while stack:
        node = stack.pop()
        if not visited[node]:
            scc = []
            dfs2(node, scc)
            sccs.append(scc)
    return sccs

def articulation_points(n, edges):
    """Finds articulation points in graph."""
    from collections import defaultdict
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    disc = [0] * n
    low = [0] * n
    parent = [-1] * n
    ap = set()
    time = [1]

    def dfs(u):
        children = 0
        disc[u] = low[u] = time[0]
        time[0] += 1
        for v in graph[u]:
            if disc[v] == 0:
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
        if disc[i] == 0:
            dfs(i)
    return ap

def bipartite_check(n, edges):
    """Checks if graph is bipartite."""
    from collections import defaultdict, deque
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    color = [-1] * n
    for start in range(n):
        if color[start] == -1:
            queue = deque([start])
            color[start] = 0
            while queue:
                node = queue.popleft()
                for neighbor in graph[node]:
                    if color[neighbor] == -1:
                        color[neighbor] = 1 - color[node]
                        queue.append(neighbor)
                    elif color[neighbor] == color[node]:
                        return False
    return True

def euler_path(n, edges):
    """Finds Euler path if exists."""
    from collections import defaultdict
    graph = defaultdict(list)
    degree = [0] * n
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
        degree[u] += 1
        degree[v] += 1

    # Check if Euler path exists
    odd_degree = [i for i in range(n) if degree[i] % 2 == 1]
    if len(odd_degree) not in [0, 2]:
        return None

    start = odd_degree[0] if odd_degree else 0
    edge_count = defaultdict(int)
    for u, v in edges:
        edge_count[(min(u, v), max(u, v))] += 1

    path = []
    stack = [start]
    while stack:
        v = stack[-1]
        if graph[v]:
            u = graph[v].pop()
            key = (min(u, v), max(u, v))
            if edge_count[key] > 0:
                edge_count[key] -= 1
                graph[u].remove(v)
                stack.append(u)
        else:
            path.append(stack.pop())
    return path[::-1]

# Tests
tests = []

# Bellman-Ford test
bf_dist = bellman_ford(5, [(0,1,6),(0,2,7),(1,2,8),(1,3,5),(1,4,-4),(2,3,-3),(2,4,9),(3,1,-2),(4,0,2),(4,3,7)], 0)
tests.append(("bellman_ford", bf_dist[4], -2))

# Floyd-Warshall test
fw_dist = floyd_warshall(4, [(0,1,3),(0,3,7),(1,0,8),(1,2,2),(2,0,5),(2,3,1),(3,0,2)])
tests.append(("floyd_warshall", fw_dist[0][2], 5))  # 0->1->2

# A* test
grid = [[0,0,0],[0,1,0],[0,0,0]]
path = a_star_search(grid, (0,0), (2,2))
tests.append(("a_star", len(path), 5))

# Topological sort test
tests.append(("topo_sort", topological_sort_kahn(6, [(5,2),(5,0),(4,0),(4,1),(2,3),(3,1)])[:2], [4, 5]))

# SCC test
sccs = strongly_connected_components(8, [(0,1),(1,2),(2,0),(2,3),(3,4),(4,5),(5,6),(6,4),(6,7)])
tests.append(("scc_count", len(sccs), 4))

# Articulation points test
aps = articulation_points(5, [(0,1),(1,2),(2,0),(1,3),(3,4)])
tests.append(("articulation", 1 in aps and 3 in aps, True))

# Bipartite test
tests.append(("bipartite_yes", bipartite_check(4, [(0,1),(1,2),(2,3),(3,0)]), True))
tests.append(("bipartite_no", bipartite_check(3, [(0,1),(1,2),(2,0)]), False))

# Euler path test
euler = euler_path(5, [(0,1),(1,2),(2,0),(2,3),(3,4),(4,2)])
tests.append(("euler_exists", euler is not None, True))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
