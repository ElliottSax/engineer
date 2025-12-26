# ULTRA: Advanced Segment Tree Operations

import sys
sys.setrecursionlimit(10000)

# ULTRA: Persistent Segment Tree with Range Updates
class PersistentLazySegTree:
    def __init__(self, arr):
        self.n = len(arr)
        # Each node: [value, lazy, left_child, right_child]
        self.nodes = []
        self.roots = []
        if self.n > 0:
            root = self._build(arr, 0, self.n - 1)
            self.roots.append(root)

    def _new_node(self, val=0, lazy=0, left=-1, right=-1):
        idx = len(self.nodes)
        self.nodes.append([val, lazy, left, right])
        return idx

    def _build(self, arr, start, end):
        if start == end:
            return self._new_node(arr[start])
        mid = (start + end) // 2
        left = self._build(arr, start, mid)
        right = self._build(arr, mid + 1, end)
        node = self._new_node(self.nodes[left][0] + self.nodes[right][0], 0, left, right)
        return node

    def _push_down(self, node, start, end):
        if self.nodes[node][1] != 0:
            mid = (start + end) // 2
            # Create new children
            new_left = self._new_node(*self.nodes[self.nodes[node][2]])
            new_right = self._new_node(*self.nodes[self.nodes[node][3]])

            self.nodes[new_left][0] += self.nodes[node][1] * (mid - start + 1)
            self.nodes[new_left][1] += self.nodes[node][1]
            self.nodes[new_right][0] += self.nodes[node][1] * (end - mid)
            self.nodes[new_right][1] += self.nodes[node][1]

            self.nodes[node][2] = new_left
            self.nodes[node][3] = new_right
            self.nodes[node][1] = 0

    def update_range(self, version, l, r, val):
        """Return new version after range update."""
        new_root = self._update(self.roots[version], 0, self.n - 1, l, r, val)
        self.roots.append(new_root)
        return len(self.roots) - 1

    def _update(self, node, start, end, l, r, val):
        new_node = self._new_node(*self.nodes[node])

        if r < start or end < l:
            return new_node

        if l <= start and end <= r:
            self.nodes[new_node][0] += val * (end - start + 1)
            self.nodes[new_node][1] += val
            return new_node

        self._push_down(new_node, start, end)
        mid = (start + end) // 2
        self.nodes[new_node][2] = self._update(self.nodes[new_node][2], start, mid, l, r, val)
        self.nodes[new_node][3] = self._update(self.nodes[new_node][3], mid + 1, end, l, r, val)
        self.nodes[new_node][0] = self.nodes[self.nodes[new_node][2]][0] + self.nodes[self.nodes[new_node][3]][0]

        return new_node

    def query(self, version, l, r):
        return self._query(self.roots[version], 0, self.n - 1, l, r)

    def _query(self, node, start, end, l, r):
        if r < start or end < l:
            return 0
        if l <= start and end <= r:
            return self.nodes[node][0]

        self._push_down(node, start, end)
        mid = (start + end) // 2
        return self._query(self.nodes[node][2], start, mid, l, r) + \
               self._query(self.nodes[node][3], mid + 1, end, l, r)

# ULTRA: Segment Tree Beats (maintaining max/second max)
class SegmentTreeBeats:
    def __init__(self, arr):
        self.n = len(arr)
        self.INF = float('inf')
        # Each node: [max1, max2, max_cnt, sum]
        self.tree = [[0, -self.INF, 0, 0] for _ in range(4 * self.n)]
        if self.n > 0:
            self._build(arr, 0, 0, self.n - 1)

    def _build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = [arr[start], -self.INF, 1, arr[start]]
            return

        mid = (start + end) // 2
        self._build(arr, 2*node+1, start, mid)
        self._build(arr, 2*node+2, mid+1, end)
        self._merge(node)

    def _merge(self, node):
        l, r = 2*node+1, 2*node+2
        if self.tree[l][0] > self.tree[r][0]:
            self.tree[node][0] = self.tree[l][0]
            self.tree[node][2] = self.tree[l][2]
            self.tree[node][1] = max(self.tree[l][1], self.tree[r][0])
        elif self.tree[l][0] < self.tree[r][0]:
            self.tree[node][0] = self.tree[r][0]
            self.tree[node][2] = self.tree[r][2]
            self.tree[node][1] = max(self.tree[l][0], self.tree[r][1])
        else:
            self.tree[node][0] = self.tree[l][0]
            self.tree[node][2] = self.tree[l][2] + self.tree[r][2]
            self.tree[node][1] = max(self.tree[l][1], self.tree[r][1])
        self.tree[node][3] = self.tree[l][3] + self.tree[r][3]

    def _push_down(self, node, start, end):
        # Apply min operation to children
        for child in [2*node+1, 2*node+2]:
            if self.tree[node][0] < self.tree[child][0]:
                self.tree[child][3] -= (self.tree[child][0] - self.tree[node][0]) * self.tree[child][2]
                self.tree[child][0] = self.tree[node][0]

    def update_min(self, l, r, val):
        """Set all elements in [l, r] to min(element, val)."""
        self._update_min(0, 0, self.n - 1, l, r, val)

    def _update_min(self, node, start, end, l, r, val):
        if r < start or end < l or self.tree[node][0] <= val:
            return

        if l <= start and end <= r and self.tree[node][1] < val:
            self.tree[node][3] -= (self.tree[node][0] - val) * self.tree[node][2]
            self.tree[node][0] = val
            return

        mid = (start + end) // 2
        self._push_down(node, start, end)
        self._update_min(2*node+1, start, mid, l, r, val)
        self._update_min(2*node+2, mid+1, end, l, r, val)
        self._merge(node)

    def query_sum(self, l, r):
        return self._query_sum(0, 0, self.n - 1, l, r)

    def _query_sum(self, node, start, end, l, r):
        if r < start or end < l:
            return 0
        if l <= start and end <= r:
            return self.tree[node][3]

        mid = (start + end) // 2
        self._push_down(node, start, end)
        return self._query_sum(2*node+1, start, mid, l, r) + \
               self._query_sum(2*node+2, mid+1, end, l, r)

    def query_max(self, l, r):
        return self._query_max(0, 0, self.n - 1, l, r)

    def _query_max(self, node, start, end, l, r):
        if r < start or end < l:
            return -self.INF
        if l <= start and end <= r:
            return self.tree[node][0]

        mid = (start + end) // 2
        self._push_down(node, start, end)
        return max(self._query_max(2*node+1, start, mid, l, r),
                   self._query_max(2*node+2, mid+1, end, l, r))

# ULTRA: 2D Segment Tree
class SegmentTree2D:
    def __init__(self, matrix):
        self.m = len(matrix)
        self.n = len(matrix[0]) if self.m > 0 else 0
        self.tree = [[0] * (4 * self.n) for _ in range(4 * self.m)]
        if self.m > 0 and self.n > 0:
            self._build_y(matrix, 0, 0, self.m - 1)

    def _build_y(self, matrix, node_y, start_y, end_y):
        if start_y == end_y:
            self._build_x(matrix, node_y, start_y, end_y, 0, 0, self.n - 1)
        else:
            mid = (start_y + end_y) // 2
            self._build_y(matrix, 2*node_y+1, start_y, mid)
            self._build_y(matrix, 2*node_y+2, mid+1, end_y)
            self._merge_y(node_y, 0, 0, self.n - 1)

    def _build_x(self, matrix, node_y, start_y, end_y, node_x, start_x, end_x):
        if start_x == end_x:
            self.tree[node_y][node_x] = matrix[start_y][start_x]
        else:
            mid = (start_x + end_x) // 2
            self._build_x(matrix, node_y, start_y, end_y, 2*node_x+1, start_x, mid)
            self._build_x(matrix, node_y, start_y, end_y, 2*node_x+2, mid+1, end_x)
            self.tree[node_y][node_x] = self.tree[node_y][2*node_x+1] + self.tree[node_y][2*node_x+2]

    def _merge_y(self, node_y, node_x, start_x, end_x):
        if start_x == end_x:
            self.tree[node_y][node_x] = self.tree[2*node_y+1][node_x] + self.tree[2*node_y+2][node_x]
        else:
            mid = (start_x + end_x) // 2
            self._merge_y(node_y, 2*node_x+1, start_x, mid)
            self._merge_y(node_y, 2*node_x+2, mid+1, end_x)
            self.tree[node_y][node_x] = self.tree[node_y][2*node_x+1] + self.tree[node_y][2*node_x+2]

    def query(self, y1, x1, y2, x2):
        return self._query_y(0, 0, self.m - 1, y1, y2, x1, x2)

    def _query_y(self, node_y, start_y, end_y, y1, y2, x1, x2):
        if y2 < start_y or end_y < y1:
            return 0
        if y1 <= start_y and end_y <= y2:
            return self._query_x(node_y, 0, 0, self.n - 1, x1, x2)

        mid = (start_y + end_y) // 2
        return self._query_y(2*node_y+1, start_y, mid, y1, y2, x1, x2) + \
               self._query_y(2*node_y+2, mid+1, end_y, y1, y2, x1, x2)

    def _query_x(self, node_y, node_x, start_x, end_x, x1, x2):
        if x2 < start_x or end_x < x1:
            return 0
        if x1 <= start_x and end_x <= x2:
            return self.tree[node_y][node_x]

        mid = (start_x + end_x) // 2
        return self._query_x(node_y, 2*node_x+1, start_x, mid, x1, x2) + \
               self._query_x(node_y, 2*node_x+2, mid+1, end_x, x1, x2)

# Tests
tests = []

# Persistent Lazy Seg Tree
arr = [1, 2, 3, 4, 5]
plst = PersistentLazySegTree(arr)
tests.append(("plst_v0", plst.query(0, 0, 4), 15))
v1 = plst.update_range(0, 1, 3, 10)
tests.append(("plst_v1", plst.query(v1, 0, 4), 45))
tests.append(("plst_v0_unchanged", plst.query(0, 0, 4), 15))

# Segment Tree Beats
stb = SegmentTreeBeats([5, 3, 8, 2, 7])
tests.append(("stb_sum", stb.query_sum(0, 4), 25))
tests.append(("stb_max", stb.query_max(0, 4), 8))
stb.update_min(1, 3, 4)
tests.append(("stb_after_min", stb.query_sum(0, 4), 21))  # 5+3+4+2+7

# 2D Segment Tree
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
st2d = SegmentTree2D(matrix)
tests.append(("st2d_all", st2d.query(0, 0, 2, 2), 45))
tests.append(("st2d_partial", st2d.query(0, 0, 1, 1), 12))  # 1+2+4+5
tests.append(("st2d_row", st2d.query(1, 0, 1, 2), 15))  # 4+5+6

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
