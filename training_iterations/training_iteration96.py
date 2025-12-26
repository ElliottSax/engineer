# EXTREME: Persistent Data Structures & Advanced Queries

from collections import defaultdict

# HARD: Persistent Segment Tree
class PersistentSegTree:
    def __init__(self, n):
        self.n = n
        self.nodes = [[0, -1, -1]]  # [value, left_child, right_child]
        self.roots = []

    def _build(self, arr, node, start, end):
        if start == end:
            self.nodes[node][0] = arr[start]
            return
        mid = (start + end) // 2
        left = len(self.nodes)
        self.nodes.append([0, -1, -1])
        right = len(self.nodes)
        self.nodes.append([0, -1, -1])
        self.nodes[node][1] = left
        self.nodes[node][2] = right
        self._build(arr, left, start, mid)
        self._build(arr, right, mid + 1, end)
        self.nodes[node][0] = self.nodes[left][0] + self.nodes[right][0]

    def build(self, arr):
        root = len(self.nodes)
        self.nodes.append([0, -1, -1])
        self._build(arr, root, 0, self.n - 1)
        self.roots.append(root)
        return len(self.roots) - 1

    def _update(self, prev_node, start, end, idx, val):
        new_node = len(self.nodes)
        self.nodes.append(self.nodes[prev_node][:])

        if start == end:
            self.nodes[new_node][0] = val
            return new_node

        mid = (start + end) // 2
        if idx <= mid:
            left = self._update(self.nodes[prev_node][1], start, mid, idx, val)
            self.nodes[new_node][1] = left
        else:
            right = self._update(self.nodes[prev_node][2], mid + 1, end, idx, val)
            self.nodes[new_node][2] = right

        self.nodes[new_node][0] = self.nodes[self.nodes[new_node][1]][0] + \
                                  self.nodes[self.nodes[new_node][2]][0]
        return new_node

    def update(self, version, idx, val):
        new_root = self._update(self.roots[version], 0, self.n - 1, idx, val)
        self.roots.append(new_root)
        return len(self.roots) - 1

    def _query(self, node, start, end, l, r):
        if r < start or end < l:
            return 0
        if l <= start and end <= r:
            return self.nodes[node][0]
        mid = (start + end) // 2
        return self._query(self.nodes[node][1], start, mid, l, r) + \
               self._query(self.nodes[node][2], mid + 1, end, l, r)

    def query(self, version, l, r):
        return self._query(self.roots[version], 0, self.n - 1, l, r)

# HARD: Mo's Algorithm for Range Queries
def mo_algorithm(arr, queries):
    """Answer range queries efficiently using Mo's algorithm."""
    n = len(arr)
    if n == 0:
        return []

    block_size = max(1, int(n ** 0.5))
    q = len(queries)

    # Sort queries by Mo's ordering
    sorted_queries = sorted(range(q), key=lambda i: (
        queries[i][0] // block_size,
        queries[i][1] if (queries[i][0] // block_size) % 2 == 0 else -queries[i][1]
    ))

    # Answer queries
    answers = [0] * q
    freq = defaultdict(int)
    distinct = 0
    curr_l, curr_r = 0, -1

    def add(idx):
        nonlocal distinct
        if freq[arr[idx]] == 0:
            distinct += 1
        freq[arr[idx]] += 1

    def remove(idx):
        nonlocal distinct
        freq[arr[idx]] -= 1
        if freq[arr[idx]] == 0:
            distinct -= 1

    for qi in sorted_queries:
        l, r = queries[qi]
        while curr_r < r:
            curr_r += 1
            add(curr_r)
        while curr_l > l:
            curr_l -= 1
            add(curr_l)
        while curr_r > r:
            remove(curr_r)
            curr_r -= 1
        while curr_l < l:
            remove(curr_l)
            curr_l += 1
        answers[qi] = distinct

    return answers

# HARD: Sqrt Decomposition for Range Sum
class SqrtDecomposition:
    def __init__(self, arr):
        self.arr = arr[:]
        self.n = len(arr)
        self.block_size = max(1, int(self.n ** 0.5))
        self.num_blocks = (self.n + self.block_size - 1) // self.block_size
        self.blocks = [0] * self.num_blocks

        for i in range(self.n):
            self.blocks[i // self.block_size] += arr[i]

    def update(self, idx, val):
        block = idx // self.block_size
        self.blocks[block] += val - self.arr[idx]
        self.arr[idx] = val

    def query(self, l, r):
        result = 0
        while l <= r and l % self.block_size != 0:
            result += self.arr[l]
            l += 1
        while l + self.block_size - 1 <= r:
            result += self.blocks[l // self.block_size]
            l += self.block_size
        while l <= r:
            result += self.arr[l]
            l += 1
        return result

# HARD: Wavelet Tree for Range Queries
class WaveletTree:
    def __init__(self, arr, lo, hi):
        self.lo = lo
        self.hi = hi
        if lo == hi or not arr:
            self.left = self.right = None
            return

        mid = (lo + hi) // 2
        self.b = [0]  # Prefix count of elements <= mid

        left_arr = []
        right_arr = []
        for x in arr:
            if x <= mid:
                left_arr.append(x)
                self.b.append(self.b[-1] + 1)
            else:
                right_arr.append(x)
                self.b.append(self.b[-1])

        self.left = WaveletTree(left_arr, lo, mid) if left_arr else None
        self.right = WaveletTree(right_arr, mid + 1, hi) if right_arr else None

    def kth(self, l, r, k):
        """Find k-th smallest in range [l, r] (1-indexed)."""
        if self.lo == self.hi:
            return self.lo

        left_count = self.b[r + 1] - self.b[l]

        if k <= left_count:
            new_l = self.b[l]
            new_r = self.b[r + 1] - 1
            return self.left.kth(new_l, new_r, k)
        else:
            new_l = l - self.b[l]
            new_r = r - self.b[r + 1]
            return self.right.kth(new_l, new_r, k - left_count)

    def count_less(self, l, r, k):
        """Count elements < k in range [l, r]."""
        if self.lo == self.hi:
            return 0 if self.lo >= k else r - l + 1

        mid = (self.lo + self.hi) // 2
        left_count = self.b[r + 1] - self.b[l]

        if k <= mid:
            if self.left is None:
                return 0
            new_l = self.b[l]
            new_r = self.b[r + 1] - 1
            if new_r < new_l:
                return 0
            return self.left.count_less(new_l, new_r, k)
        else:
            result = left_count
            if self.right is not None:
                new_l = l - self.b[l]
                new_r = r - self.b[r + 1]
                if new_r >= new_l:
                    result += self.right.count_less(new_l, new_r, k)
            return result

# Tests
tests = []

# Persistent Segment Tree
pst = PersistentSegTree(5)
v0 = pst.build([1, 2, 3, 4, 5])
tests.append(("pst_init", pst.query(v0, 0, 4), 15))
v1 = pst.update(v0, 2, 10)
tests.append(("pst_v1", pst.query(v1, 0, 4), 22))
tests.append(("pst_v0", pst.query(v0, 0, 4), 15))  # Original unchanged

# Mo's Algorithm
arr = [1, 2, 1, 3, 1, 2, 1]
queries = [(0, 2), (1, 4), (2, 6)]
results = mo_algorithm(arr, queries)
tests.append(("mo", results, [2, 3, 3]))

# Sqrt Decomposition
sq = SqrtDecomposition([1, 2, 3, 4, 5, 6, 7, 8, 9])
tests.append(("sqrt_query", sq.query(2, 6), 25))
sq.update(4, 10)
tests.append(("sqrt_update", sq.query(2, 6), 30))

# Wavelet Tree
arr = [3, 1, 4, 1, 5, 9, 2, 6]
wt = WaveletTree(arr, 1, 9)
tests.append(("wavelet_kth", wt.kth(0, 4, 2), 1))  # 2nd smallest in [3,1,4,1,5] is 1
tests.append(("wavelet_kth2", wt.kth(0, 4, 4), 4))  # 4th smallest is 4
tests.append(("wavelet_less", wt.count_less(0, 4, 4), 3))  # 3 elements < 4

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
