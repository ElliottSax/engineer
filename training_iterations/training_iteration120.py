# MILESTONE ITERATION 120: Ultimate Algorithm Collection

from collections import defaultdict, deque
from functools import lru_cache
import heapq

print("=" * 60)
print("🏆 ITERATION 120 MILESTONE - ULTIMATE ALGORITHM COLLECTION 🏆")
print("=" * 60)

# ===== ULTIMATE 1: Persistent Union-Find =====
class PersistentDSU:
    """Persistent DSU with path compression and rollback."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.history = []

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        return root

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False

        if self.rank[px] < self.rank[py]:
            px, py = py, px

        self.history.append((py, self.parent[py], px, self.rank[px]))
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

    def save(self):
        return len(self.history)

    def rollback(self, checkpoint):
        while len(self.history) > checkpoint:
            py, old_parent, px, old_rank = self.history.pop()
            self.parent[py] = old_parent
            self.rank[px] = old_rank

# ===== ULTIMATE 2: Treap (Randomized BST) =====
import random
random.seed(42)

class TreapNode:
    def __init__(self, key):
        self.key = key
        self.priority = random.random()
        self.left = None
        self.right = None
        self.size = 1

def treap_size(node):
    return node.size if node else 0

def treap_update(node):
    if node:
        node.size = 1 + treap_size(node.left) + treap_size(node.right)

def treap_split(node, key):
    """Split treap into nodes < key and nodes >= key."""
    if not node:
        return None, None
    if node.key < key:
        left, right = treap_split(node.right, key)
        node.right = left
        treap_update(node)
        return node, right
    else:
        left, right = treap_split(node.left, key)
        node.left = right
        treap_update(node)
        return left, node

def treap_merge(left, right):
    """Merge two treaps."""
    if not left:
        return right
    if not right:
        return left
    if left.priority > right.priority:
        left.right = treap_merge(left.right, right)
        treap_update(left)
        return left
    else:
        right.left = treap_merge(left, right.left)
        treap_update(right)
        return right

def treap_insert(root, key):
    left, right = treap_split(root, key)
    return treap_merge(treap_merge(left, TreapNode(key)), right)

def treap_kth(node, k):
    """Find k-th smallest element (1-indexed)."""
    if not node:
        return None
    left_size = treap_size(node.left)
    if k <= left_size:
        return treap_kth(node.left, k)
    elif k == left_size + 1:
        return node.key
    else:
        return treap_kth(node.right, k - left_size - 1)

# ===== ULTIMATE 3: Suffix Array with LCP (O(n log n)) =====
def suffix_array_lcp(s):
    """Build suffix array and LCP array in O(n log n)."""
    n = len(s)
    if n == 0:
        return [], []

    # Initial ranking
    sa = list(range(n))
    rank = [ord(c) for c in s]
    tmp = [0] * n

    k = 1
    while k < n:
        # Sort by (rank[i], rank[i+k])
        def key(i):
            return (rank[i], rank[i + k] if i + k < n else -1)
        sa.sort(key=key)

        # Update ranks
        tmp[sa[0]] = 0
        for i in range(1, n):
            tmp[sa[i]] = tmp[sa[i-1]]
            if key(sa[i]) != key(sa[i-1]):
                tmp[sa[i]] += 1
        rank = tmp[:]
        k *= 2

    # Build LCP array
    lcp = [0] * n
    k = 0
    rank_inv = [0] * n
    for i in range(n):
        rank_inv[sa[i]] = i

    for i in range(n):
        if rank_inv[i] == 0:
            k = 0
            continue
        j = sa[rank_inv[i] - 1]
        while i + k < n and j + k < n and s[i + k] == s[j + k]:
            k += 1
        lcp[rank_inv[i]] = k
        if k > 0:
            k -= 1

    return sa, lcp

# ===== ULTIMATE 4: Heavy-Light Decomposition Query =====
def hld_path_max(n, edges, weights, queries):
    """Answer path max queries using HLD."""
    if n == 0:
        return []

    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, weights[i]))
        adj[v].append((u, weights[i]))

    parent = [-1] * n
    depth = [0] * n
    size = [1] * n
    edge_weight = [0] * n
    heavy = [-1] * n

    # DFS for tree structure
    stack = [(0, -1, 0)]
    order = []
    while stack:
        u, p, d = stack.pop()
        parent[u] = p
        depth[u] = d
        order.append(u)
        for v, w in adj[u]:
            if v != p:
                edge_weight[v] = w
                stack.append((v, u, d + 1))

    # Compute sizes and heavy children
    for u in reversed(order):
        max_size = 0
        for v, w in adj[u]:
            if v != parent[u]:
                size[u] += size[v]
                if size[v] > max_size:
                    max_size = size[v]
                    heavy[u] = v

    # Decompose into chains
    chain_head = [0] * n
    chain_pos = [0] * n
    chain_vals = []
    pos = 0

    for u in order:
        if parent[u] == -1 or heavy[parent[u]] != u:
            v = u
            while v != -1:
                chain_head[v] = u
                chain_pos[v] = pos
                chain_vals.append(edge_weight[v])
                pos += 1
                v = heavy[v]

    # Segment tree
    seg_size = max(len(chain_vals), 1)
    tree = [0] * (4 * seg_size)

    def build(node, start, end):
        if start == end:
            tree[node] = chain_vals[start] if start < len(chain_vals) else 0
            return
        mid = (start + end) // 2
        build(2*node+1, start, mid)
        build(2*node+2, mid+1, end)
        tree[node] = max(tree[2*node+1], tree[2*node+2])

    if chain_vals:
        build(0, 0, len(chain_vals) - 1)

    def query_tree(node, start, end, l, r):
        if r < start or end < l:
            return 0
        if l <= start and end <= r:
            return tree[node]
        mid = (start + end) // 2
        return max(query_tree(2*node+1, start, mid, l, r),
                   query_tree(2*node+2, mid+1, end, l, r))

    def path_max(u, v):
        result = 0
        while chain_head[u] != chain_head[v]:
            if depth[chain_head[u]] < depth[chain_head[v]]:
                u, v = v, u
            result = max(result, query_tree(0, 0, len(chain_vals) - 1,
                                           chain_pos[chain_head[u]], chain_pos[u]))
            u = parent[chain_head[u]]

        if depth[u] > depth[v]:
            u, v = v, u
        if u != v:
            result = max(result, query_tree(0, 0, len(chain_vals) - 1,
                                           chain_pos[u] + 1, chain_pos[v]))
        return result

    return [path_max(u, v) for u, v in queries]

# ===== ULTIMATE 5: Maximum Flow (Dinic's) =====
class MaxFlow:
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

# ===== ULTIMATE 6: Centroid Decomposition =====
def centroid_decomposition(n, edges):
    """Build centroid decomposition of tree."""
    if n == 0:
        return [], []

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    size = [0] * n
    removed = [False] * n
    centroid_parent = [-1] * n
    centroids = []

    def get_size(u, p):
        size[u] = 1
        for v in adj[u]:
            if v != p and not removed[v]:
                size[u] += get_size(v, u)
        return size[u]

    def get_centroid(u, p, tree_size):
        for v in adj[u]:
            if v != p and not removed[v] and size[v] > tree_size // 2:
                return get_centroid(v, u, tree_size)
        return u

    def decompose(u, p):
        tree_size = get_size(u, -1)
        c = get_centroid(u, -1, tree_size)
        removed[c] = True
        centroid_parent[c] = p
        centroids.append(c)

        for v in adj[c]:
            if not removed[v]:
                decompose(v, c)

    decompose(0, -1)
    return centroids, centroid_parent

# Tests
tests = []

# Persistent DSU
pdsu = PersistentDSU(5)
pdsu.union(0, 1)
cp1 = pdsu.save()
pdsu.union(1, 2)
tests.append(("pdsu_connected", pdsu.find(0) == pdsu.find(2), True))
pdsu.rollback(cp1)
tests.append(("pdsu_rollback", pdsu.find(0) == pdsu.find(2), False))

# Treap
root = None
for x in [5, 3, 7, 1, 9]:
    root = treap_insert(root, x)
tests.append(("treap_size", treap_size(root), 5))
tests.append(("treap_kth", treap_kth(root, 3), 5))  # 3rd smallest

# Suffix Array
sa, lcp = suffix_array_lcp("banana")
tests.append(("sa_len", len(sa), 6))
tests.append(("sa_first", sa[0], 5))  # "a" is lexicographically first

# HLD
edges = [(0, 1), (0, 2), (1, 3), (1, 4)]
weights = [5, 3, 8, 2]
queries = [(3, 4), (3, 2)]
results = hld_path_max(5, edges, weights, queries)
tests.append(("hld_max", results[0], 8))

# Max Flow
mf = MaxFlow(6)
mf.add_edge(0, 1, 16)
mf.add_edge(0, 2, 13)
mf.add_edge(1, 2, 10)
mf.add_edge(1, 3, 12)
mf.add_edge(2, 1, 4)
mf.add_edge(2, 4, 14)
mf.add_edge(3, 2, 9)
mf.add_edge(3, 5, 20)
mf.add_edge(4, 3, 7)
mf.add_edge(4, 5, 4)
tests.append(("maxflow", mf.max_flow(0, 5), 23))

# Centroid Decomposition
centroids, cparent = centroid_decomposition(5, edges)
tests.append(("centroid_count", len(centroids), 5))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
print("\n" + "=" * 60)
print("🏆 ITERATION 120 MILESTONE COMPLETE! 🏆")
print("=" * 60)
