# ULTRA: Advanced Tree Algorithms II

from collections import defaultdict, deque

# ULTRA: Tree Isomorphism using Hashing
def tree_hash(n, edges, root=0):
    """Compute a hash for a rooted tree to check isomorphism."""
    if n == 0:
        return 0

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Build tree with root
    parent = [-1] * n
    children = [[] for _ in range(n)]
    visited = [False] * n
    order = []

    stack = [root]
    visited[root] = True
    while stack:
        u = stack.pop()
        order.append(u)
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                children[u].append(v)
                stack.append(v)

    # Compute hash bottom-up
    MOD = 10**9 + 7
    hash_val = [0] * n

    for u in reversed(order):
        if not children[u]:
            hash_val[u] = 1
        else:
            child_hashes = sorted(hash_val[c] for c in children[u])
            h = 1
            for ch in child_hashes:
                h = (h * 31 + ch) % MOD
            hash_val[u] = h

    return hash_val[root]

# ULTRA: Centroid of Tree
def tree_centroid(n, edges):
    """Find centroid of tree (node whose removal creates subtrees of size <= n/2)."""
    if n == 0:
        return -1
    if n == 1:
        return 0

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    size = [1] * n
    parent = [-1] * n

    # BFS to get order
    order = []
    visited = [False] * n
    queue = deque([0])
    visited[0] = True

    while queue:
        u = queue.popleft()
        order.append(u)
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                queue.append(v)

    # Compute sizes bottom-up
    for u in reversed(order):
        for v in adj[u]:
            if v != parent[u]:
                size[u] += size[v]

    # Find centroid
    for u in range(n):
        is_centroid = True
        max_subtree = n - size[u]  # Parent's subtree
        for v in adj[u]:
            if v != parent[u]:
                max_subtree = max(max_subtree, size[v])
        if max_subtree <= n // 2:
            return u

    return 0

# ULTRA: All Pairs Distances in Tree (O(n^2))
def tree_all_distances(n, edges):
    """Compute all pairwise distances in a tree."""
    if n == 0:
        return []

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    dist = [[0] * n for _ in range(n)]

    for start in range(n):
        visited = [False] * n
        queue = deque([(start, 0)])
        visited[start] = True

        while queue:
            u, d = queue.popleft()
            dist[start][u] = d
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append((v, d + 1))

    return dist

# ULTRA: Tree Diameter with Path
def tree_diameter_path(n, edges):
    """Find diameter of tree and return the path."""
    if n == 0:
        return 0, []
    if n == 1:
        return 0, [0]

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    def bfs_farthest(start):
        visited = [-1] * n
        visited[start] = 0
        queue = deque([start])
        farthest = start

        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if visited[v] == -1:
                    visited[v] = visited[u] + 1
                    queue.append(v)
                    if visited[v] > visited[farthest]:
                        farthest = v

        return farthest, visited

    # Find one endpoint
    u, _ = bfs_farthest(0)
    # Find other endpoint and distances
    v, dist = bfs_farthest(u)

    diameter = dist[v]

    # Reconstruct path
    path = [v]
    while path[-1] != u:
        curr = path[-1]
        for next_node in adj[curr]:
            if dist[next_node] == dist[curr] - 1:
                path.append(next_node)
                break

    return diameter, path

# ULTRA: Subtree Queries (sum of subtree values)
class SubtreeSum:
    def __init__(self, n, edges, values, root=0):
        self.n = n
        self.values = values[:]

        if n == 0:
            self.euler_in = []
            self.euler_out = []
            return

        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Euler tour
        self.euler_in = [0] * n
        self.euler_out = [0] * n
        self.euler = []

        visited = [False] * n
        stack = [(root, False)]
        time = 0

        while stack:
            u, leaving = stack.pop()
            if leaving:
                self.euler_out[u] = time
            else:
                self.euler_in[u] = time
                self.euler.append(u)
                time += 1
                visited[u] = True
                stack.append((u, True))
                for v in adj[u]:
                    if not visited[v]:
                        stack.append((v, False))

        # Build Fenwick tree for euler order
        self.bit = [0] * (n + 1)
        for i, u in enumerate(self.euler):
            self._update(i + 1, values[u])

    def _update(self, i, delta):
        while i <= self.n:
            self.bit[i] += delta
            i += i & (-i)

    def _prefix_sum(self, i):
        s = 0
        while i > 0:
            s += self.bit[i]
            i -= i & (-i)
        return s

    def subtree_sum(self, u):
        """Return sum of values in subtree of u."""
        l = self.euler_in[u]
        r = self.euler_out[u]
        return self._prefix_sum(r) - self._prefix_sum(l)

    def update_value(self, u, new_val):
        """Update value at node u."""
        delta = new_val - self.values[u]
        self.values[u] = new_val
        pos = self.euler_in[u] + 1
        self._update(pos, delta)

# ULTRA: Tree Flattening (DFS Order)
def tree_dfs_order(n, edges, root=0):
    """Return DFS entry and exit times for tree."""
    if n == 0:
        return [], []

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    entry = [0] * n
    exit_time = [0] * n
    time = [0]

    visited = [False] * n
    stack = [(root, False)]

    while stack:
        u, leaving = stack.pop()
        if leaving:
            exit_time[u] = time[0]
            time[0] += 1
        else:
            entry[u] = time[0]
            time[0] += 1
            visited[u] = True
            stack.append((u, True))
            for v in adj[u]:
                if not visited[v]:
                    stack.append((v, False))

    return entry, exit_time

# Tests
tests = []

# Tree hash
edges1 = [(0, 1), (1, 2)]  # Path of 3
edges2 = [(0, 1), (0, 2)]  # Star of 3
h1 = tree_hash(3, edges1, 1)  # Root at 1 (center)
h2 = tree_hash(3, edges2, 0)  # Root at 0 (center)
tests.append(("tree_hash_diff", h1 != h2, True))

# Tree centroid
edges_path = [(0, 1), (1, 2), (2, 3), (3, 4)]
centroid = tree_centroid(5, edges_path)
tests.append(("centroid_path", centroid, 2))

edges_star = [(0, 1), (0, 2), (0, 3), (0, 4)]
centroid_star = tree_centroid(5, edges_star)
tests.append(("centroid_star", centroid_star, 0))

# All distances
edges_tri = [(0, 1), (1, 2)]
dist = tree_all_distances(3, edges_tri)
tests.append(("all_dist", dist[0][2], 2))
tests.append(("all_dist2", dist[1][2], 1))

# Tree diameter
diam, path = tree_diameter_path(5, edges_path)
tests.append(("diameter", diam, 4))
tests.append(("diam_path_len", len(path), 5))

# Subtree sum
values = [1, 2, 3, 4, 5]
edges_tree = [(0, 1), (0, 2), (1, 3), (1, 4)]
ss = SubtreeSum(5, edges_tree, values, 0)
tests.append(("subtree_root", ss.subtree_sum(0), 15))
tests.append(("subtree_1", ss.subtree_sum(1), 11))  # 2+4+5
tests.append(("subtree_leaf", ss.subtree_sum(3), 4))

# DFS order
entry, exit_t = tree_dfs_order(5, edges_tree, 0)
tests.append(("dfs_order_root", entry[0], 0))
tests.append(("dfs_order_valid", entry[1] > entry[0], True))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
