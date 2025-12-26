# EXTREME: Heavy Light Decomposition & Advanced Tree Problems

from collections import defaultdict, deque

# HARD: Lowest Common Ancestor (Binary Lifting)
class LCA:
    def __init__(self, n, edges, root=0):
        self.n = n
        self.LOG = max(1, n.bit_length())
        self.parent = [[-1] * n for _ in range(self.LOG)]
        self.depth = [0] * n

        # Build tree
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # BFS to find depths and parents
        visited = [False] * n
        queue = deque([root])
        visited[root] = True
        self.parent[0][root] = root

        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    self.depth[v] = self.depth[u] + 1
                    self.parent[0][v] = u
                    queue.append(v)

        # Binary lifting
        for k in range(1, self.LOG):
            for v in range(n):
                if self.parent[k-1][v] != -1:
                    self.parent[k][v] = self.parent[k-1][self.parent[k-1][v]]

    def lca(self, u, v):
        if self.depth[u] < self.depth[v]:
            u, v = v, u

        # Bring u to same depth as v
        diff = self.depth[u] - self.depth[v]
        for k in range(self.LOG):
            if diff & (1 << k):
                u = self.parent[k][u]

        if u == v:
            return u

        # Binary search for LCA
        for k in range(self.LOG - 1, -1, -1):
            if self.parent[k][u] != self.parent[k][v]:
                u = self.parent[k][u]
                v = self.parent[k][v]

        return self.parent[0][u]

    def distance(self, u, v):
        return self.depth[u] + self.depth[v] - 2 * self.depth[self.lca(u, v)]

# HARD: Euler Tour for Range Queries
class EulerTour:
    def __init__(self, n, edges, root=0):
        self.n = n
        self.first = [0] * n
        self.last = [0] * n
        self.tour = []

        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        visited = [False] * n
        time = [0]

        def dfs(u):
            visited[u] = True
            self.first[u] = time[0]
            self.tour.append(u)
            time[0] += 1
            for v in adj[u]:
                if not visited[v]:
                    dfs(v)
            self.last[u] = time[0] - 1

        dfs(root)

    def subtree_range(self, u):
        """Return (start, end) for subtree of u in euler tour."""
        return self.first[u], self.last[u]

# HARD: Tree Diameter
def tree_diameter(n, edges):
    """Find diameter of tree (longest path)."""
    if n <= 1:
        return 0

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

        return farthest, visited[farthest]

    # Find farthest from any node
    node1, _ = bfs_farthest(0)
    # Find farthest from node1
    node2, diameter = bfs_farthest(node1)

    return diameter

# HARD: Tree Center
def tree_center(n, edges):
    """Find center(s) of tree."""
    if n == 1:
        return [0]

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Remove leaves iteratively
    leaves = deque([i for i in range(n) if len(adj[i]) <= 1])
    remaining = n

    while remaining > 2:
        new_leaves = []
        for _ in range(len(leaves)):
            leaf = leaves.popleft()
            remaining -= 1
            for neighbor in adj[leaf]:
                adj[neighbor].remove(leaf)
                if len(adj[neighbor]) == 1:
                    new_leaves.append(neighbor)
        leaves = deque(new_leaves)

    return list(leaves)

# HARD: Centroid Decomposition
def centroid_decomposition(n, edges):
    """Build centroid decomposition of tree."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    removed = [False] * n
    subtree_size = [0] * n
    parent = [-1] * n

    def get_size(u, p):
        subtree_size[u] = 1
        for v in adj[u]:
            if v != p and not removed[v]:
                subtree_size[u] += get_size(v, u)
        return subtree_size[u]

    def get_centroid(u, p, tree_size):
        for v in adj[u]:
            if v != p and not removed[v] and subtree_size[v] > tree_size // 2:
                return get_centroid(v, u, tree_size)
        return u

    def decompose(u, p):
        tree_size = get_size(u, -1)
        centroid = get_centroid(u, -1, tree_size)
        parent[centroid] = p
        removed[centroid] = True

        for v in adj[centroid]:
            if not removed[v]:
                decompose(v, centroid)

    decompose(0, -1)
    return parent

# HARD: Heavy Light Decomposition
class HLD:
    def __init__(self, n, edges, root=0):
        self.n = n
        self.adj = [[] for _ in range(n)]
        for u, v in edges:
            self.adj[u].append(v)
            self.adj[v].append(u)

        self.parent = [-1] * n
        self.depth = [0] * n
        self.subtree_size = [1] * n
        self.chain_head = [0] * n
        self.position = [0] * n
        self.current_pos = [0]

        self._dfs_size(root, -1)
        self._dfs_hld(root, -1, root)

    def _dfs_size(self, u, p):
        self.parent[u] = p
        for i, v in enumerate(self.adj[u]):
            if v == p:
                continue
            self.depth[v] = self.depth[u] + 1
            self._dfs_size(v, u)
            self.subtree_size[u] += self.subtree_size[v]
            # Move heavy child to front
            if self.subtree_size[v] > self.subtree_size[self.adj[u][0]] or self.adj[u][0] == p:
                self.adj[u][0], self.adj[u][i] = self.adj[u][i], self.adj[u][0]

    def _dfs_hld(self, u, p, head):
        self.chain_head[u] = head
        self.position[u] = self.current_pos[0]
        self.current_pos[0] += 1

        for i, v in enumerate(self.adj[u]):
            if v == p:
                continue
            if i == 0:
                # Heavy child - same chain
                self._dfs_hld(v, u, head)
            else:
                # Light child - new chain
                self._dfs_hld(v, u, v)

    def path_ranges(self, u, v):
        """Get ranges on segment tree for path u-v."""
        ranges = []
        while self.chain_head[u] != self.chain_head[v]:
            if self.depth[self.chain_head[u]] < self.depth[self.chain_head[v]]:
                u, v = v, u
            ranges.append((self.position[self.chain_head[u]], self.position[u]))
            u = self.parent[self.chain_head[u]]
        if self.depth[u] > self.depth[v]:
            u, v = v, u
        ranges.append((self.position[u], self.position[v]))
        return ranges

# Tests
tests = []

# LCA
edges = [(0,1), (0,2), (1,3), (1,4), (2,5)]
lca = LCA(6, edges, 0)
tests.append(("lca_same", lca.lca(3, 4), 1))
tests.append(("lca_diff", lca.lca(3, 5), 0))
tests.append(("lca_dist", lca.distance(3, 5), 4))

# Euler Tour
et = EulerTour(6, edges, 0)
start, end = et.subtree_range(1)
tests.append(("euler_range", end - start + 1, 3))  # Node 1 has subtree size 3

# Tree Diameter
tests.append(("diameter", tree_diameter(6, edges), 4))

# Tree Center
tests.append(("center", tree_center(5, [(0,1),(1,2),(2,3),(3,4)]), [2]))
tests.append(("center2", tree_center(4, [(0,1),(1,2),(2,3)]), [1, 2]))

# Centroid Decomposition
cd = centroid_decomposition(7, [(0,1),(1,2),(1,3),(3,4),(3,5),(5,6)])
tests.append(("centroid", cd.count(-1), 1))  # Only root has no parent

# HLD
hld = HLD(6, edges, 0)
ranges = hld.path_ranges(3, 5)
tests.append(("hld_ranges", len(ranges) >= 1, True))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
