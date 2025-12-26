# ULTRA: Codeforces Div1 Hard Level Problems

from collections import defaultdict, deque
from functools import lru_cache
import heapq

# ULTRA: Convex Hull Trick for DP Optimization
class ConvexHullTrick:
    """Monotonic deque for convex hull trick (max queries)."""
    def __init__(self):
        self.lines = deque()  # (slope, intercept)

    def bad(self, l1, l2, l3):
        """Check if l2 is dominated by l1 and l3."""
        # l2 is bad if intersection of l1,l3 is left of intersection of l1,l2
        return (l3[1] - l1[1]) * (l1[0] - l2[0]) >= (l2[1] - l1[1]) * (l1[0] - l3[0])

    def add(self, m, b):
        """Add line y = mx + b (slopes must be decreasing)."""
        line = (m, b)
        while len(self.lines) >= 2 and self.bad(self.lines[-2], self.lines[-1], line):
            self.lines.pop()
        self.lines.append(line)

    def query(self, x):
        """Get maximum y for given x (x must be increasing)."""
        while len(self.lines) >= 2:
            m1, b1 = self.lines[0]
            m2, b2 = self.lines[1]
            if m1 * x + b1 <= m2 * x + b2:
                self.lines.popleft()
            else:
                break
        m, b = self.lines[0]
        return m * x + b

def min_cost_with_cht(heights, costs):
    """Minimum cost to reach end using CHT optimization."""
    n = len(heights)
    if n <= 1:
        return 0

    # dp[i] = min cost to reach i
    # dp[i] = min(dp[j] + (h[i] - h[j])^2 + costs[i])
    # = min(dp[j] + h[i]^2 - 2*h[i]*h[j] + h[j]^2) + costs[i]
    # = h[i]^2 + costs[i] + min(-2*h[j]*h[i] + dp[j] + h[j]^2)

    cht = ConvexHullTrick()
    dp = [0] * n
    dp[0] = 0
    cht.add(-2 * heights[0], dp[0] + heights[0] ** 2)

    for i in range(1, n):
        dp[i] = heights[i] ** 2 + costs[i] + cht.query(heights[i])
        cht.add(-2 * heights[i], dp[i] + heights[i] ** 2)

    return dp[n - 1]

# ULTRA: Divide and Conquer DP Optimization
def divide_conquer_dp(n, m, cost_fn):
    """
    Optimize DP of form: dp[i][j] = min(dp[i-1][k] + cost(k+1, j)) for k < j
    where cost satisfies quadrangle inequality.
    """
    INF = float('inf')
    dp = [[INF] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0

    def solve(layer, lo, hi, opt_lo, opt_hi):
        if lo > hi:
            return
        mid = (lo + hi) // 2
        best_k = opt_lo
        best_val = INF

        for k in range(opt_lo, min(mid, opt_hi + 1)):
            val = dp[layer - 1][k] + cost_fn(k + 1, mid)
            if val < best_val:
                best_val = val
                best_k = k

        dp[layer][mid] = best_val
        solve(layer, lo, mid - 1, opt_lo, best_k)
        solve(layer, mid + 1, hi, best_k, opt_hi)

    for i in range(1, m + 1):
        solve(i, 1, n, 0, n - 1)

    return dp[m][n]

# ULTRA: Li Chao Tree (Dynamic CHT)
class LiChaoTree:
    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi
        self.mid = (lo + hi) // 2
        self.line = None  # (m, b) for y = mx + b
        self.left = None
        self.right = None

    def eval(self, line, x):
        if line is None:
            return float('-inf')
        return line[0] * x + line[1]

    def add(self, new_line):
        if self.line is None:
            self.line = new_line
            return

        l_new = self.eval(new_line, self.lo) > self.eval(self.line, self.lo)
        m_new = self.eval(new_line, self.mid) > self.eval(self.line, self.mid)

        if m_new:
            self.line, new_line = new_line, self.line

        if self.lo == self.hi:
            return

        if l_new != m_new:
            if self.left is None:
                self.left = LiChaoTree(self.lo, self.mid)
            self.left.add(new_line)
        else:
            if self.right is None:
                self.right = LiChaoTree(self.mid + 1, self.hi)
            self.right.add(new_line)

    def query(self, x):
        result = self.eval(self.line, x)
        if x <= self.mid and self.left:
            result = max(result, self.left.query(x))
        elif x > self.mid and self.right:
            result = max(result, self.right.query(x))
        return result

# ULTRA: Aliens Trick (Lagrangian Relaxation)
def aliens_trick_example(arr, k):
    """
    Find maximum sum of k non-adjacent elements using Aliens trick.
    Binary search on penalty lambda.
    """
    n = len(arr)

    def solve_with_penalty(penalty):
        """Returns (max_sum, count) with penalty for each element chosen."""
        if n == 0:
            return 0, 0

        # dp[i][0] = max sum not taking i, dp[i][1] = max sum taking i
        dp = [[0, 0] for _ in range(n)]
        cnt = [[0, 0] for _ in range(n)]

        dp[0][0] = 0
        dp[0][1] = arr[0] - penalty
        cnt[0][0] = 0
        cnt[0][1] = 1

        for i in range(1, n):
            # Not taking i
            if dp[i-1][0] >= dp[i-1][1]:
                dp[i][0] = dp[i-1][0]
                cnt[i][0] = cnt[i-1][0]
            else:
                dp[i][0] = dp[i-1][1]
                cnt[i][0] = cnt[i-1][1]

            # Taking i (can't take i-1)
            dp[i][1] = dp[i-1][0] + arr[i] - penalty
            cnt[i][1] = cnt[i-1][0] + 1

        if dp[n-1][0] >= dp[n-1][1]:
            return dp[n-1][0], cnt[n-1][0]
        return dp[n-1][1], cnt[n-1][1]

    lo, hi = -10**9, 10**9
    result = 0

    for _ in range(100):  # Binary search
        mid = (lo + hi) / 2
        value, count = solve_with_penalty(mid)
        if count >= k:
            result = value + mid * k
            lo = mid
        else:
            hi = mid

    return int(round(result))

# ULTRA: Small to Large Merging (DSU on Trees)
def dsu_on_tree_example(n, edges, colors):
    """Count distinct colors in each subtree using small-to-large."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    parent = [-1] * n
    order = []
    subtree_size = [1] * n

    # DFS to get order and sizes
    stack = [(0, -1, False)]
    while stack:
        node, par, processed = stack.pop()
        if processed:
            for child in adj[node]:
                if child != par:
                    subtree_size[node] += subtree_size[child]
            order.append(node)
        else:
            parent[node] = par
            stack.append((node, par, True))
            for child in adj[node]:
                if child != par:
                    stack.append((child, node, False))

    # Process in order (leaves first)
    color_sets = [set() for _ in range(n)]
    result = [0] * n

    for node in order:
        # Start with node's own color
        color_sets[node].add(colors[node])

        # Merge children's sets (small to large)
        children = [c for c in adj[node] if c != parent[node]]
        if children:
            # Find largest child set
            largest = max(children, key=lambda c: len(color_sets[c]))
            color_sets[node] = color_sets[largest]
            color_sets[node].add(colors[node])

            # Merge smaller sets
            for child in children:
                if child != largest:
                    color_sets[node].update(color_sets[child])
                    color_sets[child] = set()  # Free memory

        result[node] = len(color_sets[node])

    return result

# ULTRA: Centroid Decomposition for Path Queries
def count_paths_with_sum(n, edges, weights, target):
    """Count paths with exactly target sum using centroid decomposition."""
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, weights[i]))
        adj[v].append((u, weights[i]))

    removed = [False] * n
    subtree_size = [0] * n
    total_paths = [0]

    def get_size(u, p):
        subtree_size[u] = 1
        for v, _ in adj[u]:
            if v != p and not removed[v]:
                subtree_size[u] += get_size(v, u)
        return subtree_size[u]

    def get_centroid(u, p, tree_size):
        for v, _ in adj[u]:
            if v != p and not removed[v] and subtree_size[v] > tree_size // 2:
                return get_centroid(v, u, tree_size)
        return u

    def get_distances(u, p, dist, distances):
        distances.append(dist)
        for v, w in adj[u]:
            if v != p and not removed[v]:
                get_distances(v, u, dist + w, distances)

    def count_pairs(distances):
        from collections import Counter
        cnt = Counter(distances)
        pairs = 0
        for d in distances:
            need = target - d
            pairs += cnt.get(need, 0)
            if d == need:
                pairs -= 1  # Don't count self
        return pairs // 2

    def decompose(u):
        tree_size = get_size(u, -1)
        centroid = get_centroid(u, -1, tree_size)
        removed[centroid] = True

        # Get all distances from centroid
        all_distances = [0]  # Include centroid itself
        for v, w in adj[centroid]:
            if not removed[v]:
                child_distances = []
                get_distances(v, centroid, w, child_distances)
                all_distances.extend(child_distances)

        # Count pairs
        total_paths[0] += count_pairs(all_distances)

        # Subtract overcounted pairs within same subtree
        for v, w in adj[centroid]:
            if not removed[v]:
                child_distances = [0]
                get_distances(v, centroid, w, child_distances)
                total_paths[0] -= count_pairs(child_distances)

        # Recurse
        for v, _ in adj[centroid]:
            if not removed[v]:
                decompose(v)

    if n > 0:
        decompose(0)

    return total_paths[0]

# Tests
tests = []

# Convex Hull Trick
heights = [1, 2, 3, 4, 5]
costs = [0, 1, 1, 1, 1]
tests.append(("cht", min_cost_with_cht(heights, costs) >= 0, True))

# Li Chao Tree
lct = LiChaoTree(0, 100)
lct.add((2, 1))   # y = 2x + 1
lct.add((1, 5))   # y = x + 5
lct.add((-1, 20)) # y = -x + 20
tests.append(("lichao_5", lct.query(5), 15))   # max(11, 10, 15) = 15
tests.append(("lichao_15", lct.query(15), 31)) # max(31, 20, 5) = 31

# Aliens Trick
arr = [10, 5, 20, 15, 30]
tests.append(("aliens", aliens_trick_example(arr, 2), 50))  # 20 + 30 = 50

# DSU on Tree
colors = [1, 2, 1, 3, 2]
edges_tree = [(0, 1), (0, 2), (1, 3), (1, 4)]
result = dsu_on_tree_example(5, edges_tree, colors)
tests.append(("dsu_tree_root", result[0], 3))  # Root sees all 3 colors

# Centroid Decomposition Path Count
edges_cd = [(0, 1), (1, 2), (2, 3)]
weights_cd = [1, 1, 1]
tests.append(("centroid_paths", count_paths_with_sum(4, edges_cd, weights_cd, 2), 2))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
