# ULTRA: Link-Cut Trees and Dynamic Tree Problems

from collections import defaultdict

# ULTRA: Link-Cut Tree (Simplified Operations)
class LinkCutNode:
    def __init__(self, val, idx):
        self.val = val
        self.idx = idx
        self.parent = None
        self.left = None
        self.right = None
        self.reversed = False
        self.subtree_sum = val

    def is_root(self):
        return self.parent is None or (self.parent.left != self and self.parent.right != self)

    def push_down(self):
        if self.reversed:
            self.left, self.right = self.right, self.left
            if self.left:
                self.left.reversed ^= True
            if self.right:
                self.right.reversed ^= True
            self.reversed = False

    def update(self):
        self.subtree_sum = self.val
        if self.left:
            self.subtree_sum += self.left.subtree_sum
        if self.right:
            self.subtree_sum += self.right.subtree_sum

class LinkCutTree:
    def __init__(self, n, values=None):
        self.nodes = []
        for i in range(n):
            val = values[i] if values else 0
            self.nodes.append(LinkCutNode(val, i))

    def _rotate(self, x):
        y = x.parent
        z = y.parent
        y.push_down()
        x.push_down()

        if y.left == x:
            y.left = x.right
            if x.right:
                x.right.parent = y
            x.right = y
        else:
            y.right = x.left
            if x.left:
                x.left.parent = y
            x.left = y

        y.parent = x
        x.parent = z

        if z:
            if z.left == y:
                z.left = x
            elif z.right == y:
                z.right = x

        y.update()
        x.update()

    def _splay(self, x):
        while not x.is_root():
            y = x.parent
            if not y.is_root():
                z = y.parent
                z.push_down()
            y.push_down()
            x.push_down()

            if not y.is_root():
                if (y.left == x) == (z.left == y):
                    self._rotate(y)
                else:
                    self._rotate(x)
            self._rotate(x)

    def _access(self, x):
        last = None
        curr = x
        while curr:
            self._splay(curr)
            curr.right = last
            curr.update()
            last = curr
            curr = curr.parent
        self._splay(x)

    def make_root(self, idx):
        x = self.nodes[idx]
        self._access(x)
        x.reversed ^= True
        x.push_down()

    def find_root(self, idx):
        x = self.nodes[idx]
        self._access(x)
        while x.left:
            x.push_down()
            x = x.left
        self._splay(x)
        return x.idx

    def link(self, u, v):
        """Link trees containing u and v."""
        self.make_root(u)
        x = self.nodes[u]
        self._access(self.nodes[v])
        x.parent = self.nodes[v]

    def cut(self, u, v):
        """Cut edge between u and v."""
        self.make_root(u)
        self._access(self.nodes[v])
        if self.nodes[v].left == self.nodes[u]:
            self.nodes[v].left.parent = None
            self.nodes[v].left = None
            self.nodes[v].update()

    def connected(self, u, v):
        return self.find_root(u) == self.find_root(v)

    def path_sum(self, u, v):
        """Sum of values on path from u to v."""
        self.make_root(u)
        self._access(self.nodes[v])
        return self.nodes[v].subtree_sum

# ULTRA: Euler Tour Tree for Dynamic Connectivity
class EulerTourTree:
    def __init__(self, n):
        self.n = n
        self.parent = list(range(n))
        self.rank = [0] * n
        self.edges = defaultdict(set)

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def link(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.rank[pu] < self.rank[pv]:
            pu, pv = pv, pu
        self.parent[pv] = pu
        if self.rank[pu] == self.rank[pv]:
            self.rank[pu] += 1
        self.edges[u].add(v)
        self.edges[v].add(u)
        return True

    def cut(self, u, v):
        if v not in self.edges[u]:
            return False
        self.edges[u].remove(v)
        self.edges[v].remove(u)
        # Rebuild components
        visited = set()
        queue = [u]
        visited.add(u)
        while queue:
            node = queue.pop()
            self.parent[node] = u
            for neighbor in self.edges[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        # Update other component
        if v not in visited:
            queue = [v]
            visited2 = {v}
            while queue:
                node = queue.pop()
                self.parent[node] = v
                for neighbor in self.edges[node]:
                    if neighbor not in visited2:
                        visited2.add(neighbor)
                        queue.append(neighbor)
        return True

    def connected(self, u, v):
        return self.find(u) == self.find(v)

# ULTRA: Heavy-Light Decomposition with Path Queries
class HLDPathQueries:
    def __init__(self, n, edges, values):
        self.n = n
        self.values = values[:]
        self.adj = [[] for _ in range(n)]
        for u, v in edges:
            self.adj[u].append(v)
            self.adj[v].append(u)

        self.parent = [-1] * n
        self.depth = [0] * n
        self.size = [1] * n
        self.chain_head = [0] * n
        self.pos = [0] * n
        self.arr = [0] * n  # Flattened array for segment tree

        self._dfs_size(0, -1)
        self.timer = [0]
        self._dfs_hld(0, -1, 0)

        # Build segment tree
        self.tree = [0] * (4 * n)
        self._build_tree(0, 0, n - 1)

    def _dfs_size(self, u, p):
        self.parent[u] = p
        for i, v in enumerate(self.adj[u]):
            if v == p:
                continue
            self.depth[v] = self.depth[u] + 1
            self._dfs_size(v, u)
            self.size[u] += self.size[v]
            if self.size[v] > self.size[self.adj[u][0]] or self.adj[u][0] == p:
                self.adj[u][0], self.adj[u][i] = self.adj[u][i], self.adj[u][0]

    def _dfs_hld(self, u, p, head):
        self.chain_head[u] = head
        self.pos[u] = self.timer[0]
        self.arr[self.timer[0]] = self.values[u]
        self.timer[0] += 1

        for i, v in enumerate(self.adj[u]):
            if v == p:
                continue
            if i == 0:
                self._dfs_hld(v, u, head)
            else:
                self._dfs_hld(v, u, v)

    def _build_tree(self, node, start, end):
        if start == end:
            self.tree[node] = self.arr[start]
        else:
            mid = (start + end) // 2
            self._build_tree(2*node+1, start, mid)
            self._build_tree(2*node+2, mid+1, end)
            self.tree[node] = self.tree[2*node+1] + self.tree[2*node+2]

    def _query_tree(self, node, start, end, l, r):
        if r < start or end < l:
            return 0
        if l <= start and end <= r:
            return self.tree[node]
        mid = (start + end) // 2
        return self._query_tree(2*node+1, start, mid, l, r) + \
               self._query_tree(2*node+2, mid+1, end, l, r)

    def _update_tree(self, node, start, end, idx, val):
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            if idx <= mid:
                self._update_tree(2*node+1, start, mid, idx, val)
            else:
                self._update_tree(2*node+2, mid+1, end, idx, val)
            self.tree[node] = self.tree[2*node+1] + self.tree[2*node+2]

    def path_query(self, u, v):
        """Sum of values on path from u to v."""
        result = 0
        while self.chain_head[u] != self.chain_head[v]:
            if self.depth[self.chain_head[u]] < self.depth[self.chain_head[v]]:
                u, v = v, u
            result += self._query_tree(0, 0, self.n-1, self.pos[self.chain_head[u]], self.pos[u])
            u = self.parent[self.chain_head[u]]
        if self.depth[u] > self.depth[v]:
            u, v = v, u
        result += self._query_tree(0, 0, self.n-1, self.pos[u], self.pos[v])
        return result

    def update(self, u, val):
        self._update_tree(0, 0, self.n-1, self.pos[u], val)

# ULTRA: Virtual Tree / Auxiliary Tree
def build_virtual_tree(n, edges, marked_nodes, lca_func):
    """Build virtual tree containing only marked nodes and their LCAs."""
    if not marked_nodes:
        return []

    # Sort by DFS order (we'll use depth as proxy)
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # DFS to get order
    order = {}
    depth = [0] * n
    stack = [(0, -1, 0)]
    timer = 0
    while stack:
        u, p, d = stack.pop()
        if u in order:
            continue
        order[u] = timer
        timer += 1
        depth[u] = d
        for v in adj[u]:
            if v != p and v not in order:
                stack.append((v, u, d + 1))

    # Sort marked nodes by DFS order
    sorted_marked = sorted(marked_nodes, key=lambda x: order.get(x, 0))

    # Add LCAs
    all_nodes = set(sorted_marked)
    for i in range(len(sorted_marked) - 1):
        lca = lca_func(sorted_marked[i], sorted_marked[i + 1])
        all_nodes.add(lca)

    # Sort all nodes
    sorted_all = sorted(all_nodes, key=lambda x: order.get(x, 0))

    # Build virtual tree edges
    virtual_edges = []
    stack = []
    for node in sorted_all:
        while stack and not is_ancestor(stack[-1], node, depth, lca_func):
            stack.pop()
        if stack:
            virtual_edges.append((stack[-1], node))
        stack.append(node)

    return virtual_edges

def is_ancestor(u, v, depth, lca_func):
    """Check if u is ancestor of v."""
    return lca_func(u, v) == u

# Tests
tests = []

# Link-Cut Tree
lct = LinkCutTree(5, [1, 2, 3, 4, 5])
lct.link(0, 1)
lct.link(1, 2)
lct.link(2, 3)
tests.append(("lct_connected", lct.connected(0, 3), True))
tests.append(("lct_not_conn", lct.connected(0, 4), False))
tests.append(("lct_path_sum", lct.path_sum(0, 3), 10))  # 1+2+3+4

# Euler Tour Tree
ett = EulerTourTree(5)
ett.link(0, 1)
ett.link(1, 2)
ett.link(2, 3)
tests.append(("ett_connected", ett.connected(0, 3), True))
ett.cut(1, 2)
tests.append(("ett_cut", ett.connected(0, 3), False))

# HLD Path Queries
edges = [(0, 1), (0, 2), (1, 3), (1, 4)]
values = [1, 2, 3, 4, 5]
hld = HLDPathQueries(5, edges, values)
tests.append(("hld_path", hld.path_query(3, 4), 11))  # 4 + 2 + 5
tests.append(("hld_path2", hld.path_query(3, 2), 10))  # 4 + 2 + 1 + 3
hld.update(1, 10)
tests.append(("hld_update", hld.path_query(3, 4), 19))  # 4 + 10 + 5

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
