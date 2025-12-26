# EXTREME: Online Algorithms & Amortized Analysis

from collections import deque

# HARD: Dynamic Connectivity (Link-Cut Trees simplified)
class DynamicConnectivity:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.history = []  # For rollback

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

        self.history.append((py, self.parent[py], self.rank[px]))
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)

    def checkpoint(self):
        return len(self.history)

    def rollback(self, checkpoint):
        while len(self.history) > checkpoint:
            node, old_parent, old_rank = self.history.pop()
            self.parent[node] = old_parent
            # Restore rank of parent
            parent = self.find(node)
            if parent != node:
                self.rank[parent] = old_rank

# HARD: Online Median (Two Heaps)
import heapq

class OnlineMedian:
    def __init__(self):
        self.small = []  # Max heap (negated)
        self.large = []  # Min heap

    def add(self, num):
        if not self.small or num <= -self.small[0]:
            heapq.heappush(self.small, -num)
        else:
            heapq.heappush(self.large, num)

        # Balance
        if len(self.small) > len(self.large) + 1:
            heapq.heappush(self.large, -heapq.heappop(self.small))
        elif len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def median(self):
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2

# HARD: Monotonic Queue for Sliding Window
class MonotonicQueue:
    def __init__(self):
        self.deque = deque()  # (value, index)
        self.left = 0

    def push(self, val, idx):
        while self.deque and self.deque[-1][0] <= val:
            self.deque.pop()
        self.deque.append((val, idx))

    def pop_expired(self, min_idx):
        while self.deque and self.deque[0][1] < min_idx:
            self.deque.popleft()

    def max(self):
        return self.deque[0][0] if self.deque else None

def sliding_window_max(nums, k):
    """Maximum in each sliding window of size k."""
    if not nums or k == 0:
        return []

    mq = MonotonicQueue()
    result = []

    for i, num in enumerate(nums):
        mq.push(num, i)
        if i >= k - 1:
            mq.pop_expired(i - k + 1)
            result.append(mq.max())

    return result

# HARD: Sparse Table for RMQ
class SparseTable:
    def __init__(self, arr):
        self.n = len(arr)
        if self.n == 0:
            self.table = []
            self.log = []
            return

        self.log = [0] * (self.n + 1)
        for i in range(2, self.n + 1):
            self.log[i] = self.log[i // 2] + 1

        k = self.log[self.n] + 1
        self.table = [[0] * k for _ in range(self.n)]

        for i in range(self.n):
            self.table[i][0] = arr[i]

        for j in range(1, k):
            for i in range(self.n - (1 << j) + 1):
                self.table[i][j] = min(self.table[i][j-1],
                                       self.table[i + (1 << (j-1))][j-1])

    def query(self, l, r):
        """Range minimum query [l, r]."""
        if l > r:
            return float('inf')
        j = self.log[r - l + 1]
        return min(self.table[l][j], self.table[r - (1 << j) + 1][j])

# HARD: Disjoint Sparse Table (for non-idempotent operations)
class DisjointSparseTable:
    def __init__(self, arr, op=lambda a, b: a + b):
        self.n = len(arr)
        self.op = op
        if self.n == 0:
            self.table = []
            return

        self.levels = max(1, self.n.bit_length())
        self.table = [[0] * self.n for _ in range(self.levels)]

        for i in range(self.n):
            self.table[0][i] = arr[i]

        for level in range(1, self.levels):
            block_size = 1 << level
            for block_start in range(0, self.n, block_size):
                mid = min(block_start + block_size // 2, self.n)
                # Build from mid going left
                if mid > 0 and mid - 1 >= block_start:
                    self.table[level][mid - 1] = arr[mid - 1]
                    for i in range(mid - 2, block_start - 1, -1):
                        self.table[level][i] = op(arr[i], self.table[level][i + 1])
                # Build from mid going right
                if mid < self.n:
                    self.table[level][mid] = arr[mid]
                    for i in range(mid + 1, min(block_start + block_size, self.n)):
                        self.table[level][i] = op(self.table[level][i - 1], arr[i])

    def query(self, l, r):
        if l == r:
            return self.table[0][l]
        level = (l ^ r).bit_length()
        return self.op(self.table[level][l], self.table[level][r])

# HARD: Splay Tree Operations (simplified)
class SplayNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.parent = None

class SplayTree:
    def __init__(self):
        self.root = None

    def _rotate_right(self, x):
        y = x.left
        x.left = y.right
        if y.right:
            y.right.parent = x
        y.parent = x.parent
        if not x.parent:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

    def _rotate_left(self, x):
        y = x.right
        x.right = y.left
        if y.left:
            y.left.parent = x
        y.parent = x.parent
        if not x.parent:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def _splay(self, x):
        while x.parent:
            if not x.parent.parent:
                if x == x.parent.left:
                    self._rotate_right(x.parent)
                else:
                    self._rotate_left(x.parent)
            elif x == x.parent.left and x.parent == x.parent.parent.left:
                self._rotate_right(x.parent.parent)
                self._rotate_right(x.parent)
            elif x == x.parent.right and x.parent == x.parent.parent.right:
                self._rotate_left(x.parent.parent)
                self._rotate_left(x.parent)
            elif x == x.parent.right and x.parent == x.parent.parent.left:
                self._rotate_left(x.parent)
                self._rotate_right(x.parent)
            else:
                self._rotate_right(x.parent)
                self._rotate_left(x.parent)

    def insert(self, key):
        node = SplayNode(key)
        if not self.root:
            self.root = node
            return

        curr = self.root
        while True:
            if key < curr.key:
                if not curr.left:
                    curr.left = node
                    node.parent = curr
                    break
                curr = curr.left
            else:
                if not curr.right:
                    curr.right = node
                    node.parent = curr
                    break
                curr = curr.right
        self._splay(node)

    def find(self, key):
        curr = self.root
        while curr:
            if key == curr.key:
                self._splay(curr)
                return True
            elif key < curr.key:
                curr = curr.left
            else:
                curr = curr.right
        return False

# Tests
tests = []

# Dynamic Connectivity
dc = DynamicConnectivity(5)
dc.union(0, 1)
dc.union(2, 3)
tests.append(("dc_conn", dc.connected(0, 1), True))
tests.append(("dc_not", dc.connected(0, 2), False))
cp = dc.checkpoint()
dc.union(1, 2)
tests.append(("dc_after", dc.connected(0, 3), True))
dc.rollback(cp)
tests.append(("dc_rollback", dc.connected(0, 3), False))

# Online Median
om = OnlineMedian()
om.add(1)
om.add(2)
tests.append(("median_2", om.median(), 1.5))
om.add(3)
tests.append(("median_3", om.median(), 2))

# Sliding Window Max
tests.append(("slide_max", sliding_window_max([1,3,-1,-3,5,3,6,7], 3), [3,3,5,5,6,7]))

# Sparse Table
st = SparseTable([2, 1, 4, 3, 5])
tests.append(("sparse_rmq", st.query(1, 3), 1))
tests.append(("sparse_rmq2", st.query(2, 4), 3))

# Disjoint Sparse Table (sum)
dst = DisjointSparseTable([1, 2, 3, 4, 5])
tests.append(("disjoint_sum", dst.query(1, 3), 9))

# Splay Tree
splay = SplayTree()
for val in [5, 3, 7, 1, 4]:
    splay.insert(val)
tests.append(("splay_find", splay.find(4), True))
tests.append(("splay_not", splay.find(6), False))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
