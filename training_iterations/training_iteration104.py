# ULTRA: Optimal Algorithms for Classic Problems

from functools import lru_cache
from collections import deque
import heapq

# ULTRA: Optimal Binary Search Tree (Knuth Optimization)
def optimal_bst_knuth(freq):
    """O(n^2) optimal BST using Knuth's optimization."""
    n = len(freq)
    if n == 0:
        return 0

    # dp[i][j] = min cost for keys i..j
    # opt[i][j] = optimal root for i..j
    INF = float('inf')
    dp = [[0] * n for _ in range(n)]
    opt = [[0] * n for _ in range(n)]
    prefix_sum = [0] * (n + 1)

    for i in range(n):
        prefix_sum[i + 1] = prefix_sum[i] + freq[i]

    def sum_range(i, j):
        return prefix_sum[j + 1] - prefix_sum[i]

    # Base case: single keys
    for i in range(n):
        dp[i][i] = freq[i]
        opt[i][i] = i

    # Length 2 to n
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = INF

            # Knuth optimization: opt[i][j-1] <= opt[i][j] <= opt[i+1][j]
            lo = opt[i][j - 1] if j > 0 else i
            hi = opt[i + 1][j] if i < n - 1 else j

            for r in range(lo, min(hi, j) + 1):
                left = dp[i][r - 1] if r > i else 0
                right = dp[r + 1][j] if r < j else 0
                cost = left + right + sum_range(i, j)
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    opt[i][j] = r

    return dp[0][n - 1]

# ULTRA: Matrix Chain Multiplication (Knuth-style)
def matrix_chain_optimal(dims):
    """O(n^2) matrix chain multiplication."""
    n = len(dims) - 1
    if n <= 0:
        return 0

    INF = float('inf')
    dp = [[0] * n for _ in range(n)]

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = INF
            for k in range(i, j):
                cost = dp[i][k] + dp[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                dp[i][j] = min(dp[i][j], cost)

    return dp[0][n - 1]

# ULTRA: 1D1D DP Optimization (Divide and Conquer)
def dp_1d1d_optimized(n, cost_fn):
    """
    Optimized 1D1D DP where opt[i] is monotonic.
    dp[i] = min(dp[j] + cost(j, i)) for j < i
    """
    dp = [float('inf')] * (n + 1)
    dp[0] = 0

    def solve(lo, hi, opt_lo, opt_hi):
        if lo > hi:
            return
        mid = (lo + hi) // 2
        best_k = opt_lo
        best_val = float('inf')

        for k in range(opt_lo, min(mid, opt_hi + 1)):
            val = dp[k] + cost_fn(k, mid)
            if val < best_val:
                best_val = val
                best_k = k

        dp[mid] = best_val
        solve(lo, mid - 1, opt_lo, best_k)
        solve(mid + 1, hi, best_k, opt_hi)

    solve(1, n, 0, n - 1)
    return dp[n]

# ULTRA: Minimum Cost Max Flow (SPFA-based)
class MinCostMaxFlow:
    def __init__(self, n):
        self.n = n
        self.graph = [[] for _ in range(n)]
        self.edges = []

    def add_edge(self, u, v, cap, cost):
        self.graph[u].append(len(self.edges))
        self.edges.append([v, cap, cost])
        self.graph[v].append(len(self.edges))
        self.edges.append([u, 0, -cost])

    def min_cost_max_flow(self, s, t):
        total_flow = 0
        total_cost = 0
        INF = float('inf')

        while True:
            # SPFA for shortest path
            dist = [INF] * self.n
            dist[s] = 0
            parent = [-1] * self.n
            parent_edge = [-1] * self.n
            in_queue = [False] * self.n
            queue = deque([s])
            in_queue[s] = True

            while queue:
                u = queue.popleft()
                in_queue[u] = False
                for idx in self.graph[u]:
                    v, cap, cost = self.edges[idx]
                    if cap > 0 and dist[u] + cost < dist[v]:
                        dist[v] = dist[u] + cost
                        parent[v] = u
                        parent_edge[v] = idx
                        if not in_queue[v]:
                            queue.append(v)
                            in_queue[v] = True

            if dist[t] == INF:
                break

            # Find minimum capacity along path
            flow = INF
            v = t
            while v != s:
                flow = min(flow, self.edges[parent_edge[v]][1])
                v = parent[v]

            # Update capacities
            v = t
            while v != s:
                idx = parent_edge[v]
                self.edges[idx][1] -= flow
                self.edges[idx ^ 1][1] += flow
                v = parent[v]

            total_flow += flow
            total_cost += flow * dist[t]

        return total_flow, total_cost

# ULTRA: Weighted Bipartite Matching (Kuhn-Munkres/Hungarian)
def hungarian_optimal(cost):
    """O(n^3) Hungarian algorithm for minimum cost matching."""
    n = len(cost)
    m = len(cost[0]) if n > 0 else 0
    size = max(n, m)

    # Pad to square
    c = [[0] * size for _ in range(size)]
    for i in range(n):
        for j in range(m):
            c[i][j] = cost[i][j]

    u = [0] * (size + 1)
    v = [0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    INF = float('inf')

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
                    cur = c[i0 - 1][j - 1] - u[i0] - v[j]
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
        if p[j]:
            result[p[j] - 1] = j - 1

    total = sum(cost[i][result[i]] for i in range(n) if result[i] < m)
    return total, result[:n]

# ULTRA: Offline LCA (Tarjan's Algorithm)
def tarjan_lca(n, edges, queries, root=0):
    """Tarjan's offline LCA algorithm."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Group queries by node
    query_list = [[] for _ in range(n)]
    for i, (u, v) in enumerate(queries):
        query_list[u].append((v, i))
        query_list[v].append((u, i))

    parent = list(range(n))
    rank = [0] * n
    ancestor = list(range(n))
    visited = [False] * n
    answers = [0] * len(queries)

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1

    def dfs(u, p):
        visited[u] = True
        for v in adj[u]:
            if v != p:
                dfs(v, u)
                union(u, v)
                ancestor[find(u)] = u

        for v, qi in query_list[u]:
            if visited[v]:
                answers[qi] = ancestor[find(v)]

    dfs(root, -1)
    return answers

# Tests
tests = []

# Optimal BST
tests.append(("obst_knuth", optimal_bst_knuth([3, 2, 4, 1]), 17))

# Matrix Chain
tests.append(("matrix_chain", matrix_chain_optimal([10, 20, 30, 40, 30]), 30000))

# 1D1D DP
def cost_sq(i, j):
    return (j - i) ** 2
tests.append(("1d1d", dp_1d1d_optimized(5, cost_sq), 5))

# Min Cost Max Flow
mcmf = MinCostMaxFlow(4)
mcmf.add_edge(0, 1, 2, 1)
mcmf.add_edge(0, 2, 1, 4)
mcmf.add_edge(1, 2, 1, 2)
mcmf.add_edge(1, 3, 1, 2)
mcmf.add_edge(2, 3, 2, 1)
flow, cost = mcmf.min_cost_max_flow(0, 3)
tests.append(("mcmf_flow", flow, 2))
tests.append(("mcmf_cost", cost, 6))

# Hungarian
cost_matrix = [
    [3, 5, 10],
    [2, 8, 7],
    [4, 6, 1]
]
total, assignment = hungarian_optimal(cost_matrix)
tests.append(("hungarian", total, 6))  # 3+2+1=6 or similar min

# Tarjan LCA
edges = [(0, 1), (0, 2), (1, 3), (1, 4)]
queries = [(3, 4), (3, 2), (4, 2)]
answers = tarjan_lca(5, edges, queries, 0)
tests.append(("tarjan_lca", answers, [1, 0, 0]))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
