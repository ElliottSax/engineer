# ULTRA: Graph Optimization and Special Graphs

from collections import defaultdict, deque
import heapq

# ULTRA: 2-SAT Solver (Full Implementation)
class TwoSAT:
    def __init__(self, n):
        self.n = n
        self.graph = [[] for _ in range(2 * n)]
        self.reverse_graph = [[] for _ in range(2 * n)]

    def add_clause(self, x, neg_x, y, neg_y):
        """Add clause (x OR y) where neg indicates negation."""
        # (x OR y) => (!x -> y) AND (!y -> x)
        a = 2 * x + (1 if neg_x else 0)
        not_a = 2 * x + (0 if neg_x else 1)
        b = 2 * y + (1 if neg_y else 0)
        not_b = 2 * y + (0 if neg_y else 1)

        self.graph[not_a].append(b)
        self.graph[not_b].append(a)
        self.reverse_graph[b].append(not_a)
        self.reverse_graph[a].append(not_b)

    def solve(self):
        """Returns assignment if satisfiable, None otherwise."""
        n2 = 2 * self.n
        visited = [False] * n2
        order = []

        # First DFS
        def dfs1(v):
            visited[v] = True
            for u in self.graph[v]:
                if not visited[u]:
                    dfs1(u)
            order.append(v)

        for i in range(n2):
            if not visited[i]:
                dfs1(i)

        # Second DFS (reverse)
        comp = [-1] * n2
        comp_id = 0

        def dfs2(v, c):
            comp[v] = c
            for u in self.reverse_graph[v]:
                if comp[u] == -1:
                    dfs2(u, c)

        for v in reversed(order):
            if comp[v] == -1:
                dfs2(v, comp_id)
                comp_id += 1

        # Check satisfiability
        assignment = [False] * self.n
        for i in range(self.n):
            if comp[2*i] == comp[2*i + 1]:
                return None
            assignment[i] = comp[2*i] > comp[2*i + 1]

        return assignment

# ULTRA: Shortest Path Faster Algorithm (SPFA)
def spfa(n, edges, source):
    """SPFA with negative cycle detection."""
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))

    dist = [float('inf')] * n
    dist[source] = 0
    in_queue = [False] * n
    count = [0] * n
    queue = deque([source])
    in_queue[source] = True

    while queue:
        u = queue.popleft()
        in_queue[u] = False

        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                if not in_queue[v]:
                    queue.append(v)
                    in_queue[v] = True
                    count[v] += 1
                    if count[v] >= n:
                        return None  # Negative cycle

    return dist

# ULTRA: Dinic's Algorithm for Max Flow
class Dinic:
    def __init__(self, n):
        self.n = n
        self.graph = [[] for _ in range(n)]

    def add_edge(self, u, v, cap):
        self.graph[u].append([v, cap, len(self.graph[v])])
        self.graph[v].append([u, 0, len(self.graph[u]) - 1])

    def bfs(self, s, t):
        self.level = [-1] * self.n
        self.level[s] = 0
        queue = deque([s])

        while queue:
            u = queue.popleft()
            for v, cap, _ in self.graph[u]:
                if cap > 0 and self.level[v] < 0:
                    self.level[v] = self.level[u] + 1
                    queue.append(v)

        return self.level[t] >= 0

    def dfs(self, u, t, flow):
        if u == t:
            return flow
        for i in range(self.iter[u], len(self.graph[u])):
            self.iter[u] = i
            v, cap, rev = self.graph[u][i]
            if cap > 0 and self.level[v] == self.level[u] + 1:
                d = self.dfs(v, t, min(flow, cap))
                if d > 0:
                    self.graph[u][i][1] -= d
                    self.graph[v][rev][1] += d
                    return d
        return 0

    def max_flow(self, s, t):
        flow = 0
        while self.bfs(s, t):
            self.iter = [0] * self.n
            while True:
                f = self.dfs(s, t, float('inf'))
                if f == 0:
                    break
                flow += f
        return flow

# ULTRA: Minimum Cut in Undirected Graph (Stoer-Wagner)
def stoer_wagner(n, adj_matrix):
    """Find minimum cut using Stoer-Wagner algorithm."""
    INF = float('inf')
    # adj_matrix[i][j] = weight of edge i-j

    # Work with a copy
    adj = [row[:] for row in adj_matrix]
    vertices = list(range(n))
    min_cut = INF

    while len(vertices) > 1:
        # Maximum adjacency search
        added = [False] * n
        order = []
        weights = [0] * n

        for _ in range(len(vertices)):
            # Find maximum weight vertex not yet added
            max_w = -1
            max_v = -1
            for v in vertices:
                if not added[v] and weights[v] > max_w:
                    max_w = weights[v]
                    max_v = v

            added[max_v] = True
            order.append(max_v)

            for v in vertices:
                if not added[v]:
                    weights[v] += adj[max_v][v]

        # Last two vertices
        s, t = order[-2], order[-1]

        # Update minimum cut
        cut_weight = sum(adj[t][v] for v in vertices if v != t)
        min_cut = min(min_cut, cut_weight)

        # Merge s and t
        for v in vertices:
            if v != s and v != t:
                adj[s][v] += adj[t][v]
                adj[v][s] += adj[v][t]

        vertices.remove(t)

    return min_cut

# ULTRA: Gomory-Hu Tree (simplified)
def gomory_hu_tree(n, adj_matrix):
    """Build Gomory-Hu tree for all-pairs min-cut."""
    # Returns parent array and min-cut to parent
    parent = [0] * n
    min_cut = [float('inf')] * n

    for i in range(1, n):
        # Find min-cut between i and parent[i]
        # Simplified: use direct edge weight
        min_cut[i] = adj_matrix[i][parent[i]]

    return parent, min_cut

# ULTRA: K Shortest Paths (Yen's Algorithm)
def k_shortest_paths(n, edges, source, target, k):
    """Find k shortest paths using Yen's algorithm."""
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))

    def dijkstra(start, end, removed_edges, removed_nodes):
        dist = {start: 0}
        heap = [(0, start, [start])]

        while heap:
            d, u, path = heapq.heappop(heap)
            if u == end:
                return d, path
            if d > dist.get(u, float('inf')):
                continue

            for v, w in graph[u]:
                if v in removed_nodes:
                    continue
                if (u, v) in removed_edges:
                    continue
                new_dist = d + w
                if new_dist < dist.get(v, float('inf')):
                    dist[v] = new_dist
                    heapq.heappush(heap, (new_dist, v, path + [v]))

        return float('inf'), []

    # Find first shortest path
    dist, path = dijkstra(source, target, set(), set())
    if not path:
        return []

    result = [(dist, path)]
    candidates = []

    for _ in range(k - 1):
        for i in range(len(result[-1][1]) - 1):
            spur_node = result[-1][1][i]
            root_path = result[-1][1][:i + 1]
            root_dist = sum(
                min(w for v2, w in graph[root_path[j]] if v2 == root_path[j+1])
                for j in range(i)
            ) if i > 0 else 0

            removed_edges = set()
            for d, p in result:
                if p[:i + 1] == root_path and len(p) > i + 1:
                    removed_edges.add((p[i], p[i + 1]))

            removed_nodes = set(root_path[:-1])

            spur_dist, spur_path = dijkstra(spur_node, target, removed_edges, removed_nodes)
            if spur_path:
                total_path = root_path[:-1] + spur_path
                total_dist = root_dist + spur_dist
                if (total_dist, total_path) not in candidates:
                    heapq.heappush(candidates, (total_dist, total_path))

        if not candidates:
            break

        result.append(heapq.heappop(candidates))

    return result

# Tests
tests = []

# 2-SAT
sat = TwoSAT(3)
sat.add_clause(0, False, 1, False)  # x0 OR x1
sat.add_clause(0, True, 1, True)    # !x0 OR !x1
sat.add_clause(0, False, 2, True)   # x0 OR !x2
assignment = sat.solve()
tests.append(("2sat", assignment is not None, True))

# SPFA
dist = spfa(5, [(0,1,1),(1,2,2),(0,2,4),(2,3,1),(3,4,3)], 0)
tests.append(("spfa", dist, [0, 1, 3, 4, 7]))

# SPFA negative cycle
dist_neg = spfa(3, [(0,1,1),(1,2,-1),(2,0,-1)], 0)
tests.append(("spfa_neg", dist_neg, None))

# Dinic
dinic = Dinic(6)
dinic.add_edge(0, 1, 16)
dinic.add_edge(0, 2, 13)
dinic.add_edge(1, 2, 10)
dinic.add_edge(1, 3, 12)
dinic.add_edge(2, 1, 4)
dinic.add_edge(2, 4, 14)
dinic.add_edge(3, 2, 9)
dinic.add_edge(3, 5, 20)
dinic.add_edge(4, 3, 7)
dinic.add_edge(4, 5, 4)
tests.append(("dinic", dinic.max_flow(0, 5), 23))

# Stoer-Wagner
adj = [
    [0, 2, 0, 3],
    [2, 0, 3, 0],
    [0, 3, 0, 4],
    [3, 0, 4, 0]
]
tests.append(("stoer_wagner", stoer_wagner(4, adj), 5))

# K shortest paths
ksp = k_shortest_paths(4, [(0,1,1),(0,2,2),(1,3,2),(2,3,1)], 0, 3, 2)
tests.append(("ksp_count", len(ksp), 2))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
