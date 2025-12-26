# ULTRA: Range Query Data Structures

from math import sqrt, log2, ceil

# ULTRA: Sparse Table (Range Minimum Query)
class SparseTable:
    def __init__(self, arr, func=min):
        """Build sparse table for O(1) range queries."""
        self.n = len(arr)
        self.func = func
        if self.n == 0:
            self.table = []
            self.log = []
            return

        self.k = int(log2(self.n)) + 1
        self.table = [[0] * self.n for _ in range(self.k)]
        self.log = [0] * (self.n + 1)

        # Precompute logs
        for i in range(2, self.n + 1):
            self.log[i] = self.log[i // 2] + 1

        # Build table
        self.table[0] = arr[:]
        for j in range(1, self.k):
            for i in range(self.n - (1 << j) + 1):
                self.table[j][i] = func(self.table[j-1][i],
                                        self.table[j-1][i + (1 << (j-1))])

    def query(self, l, r):
        """Query range [l, r] in O(1)."""
        if l > r or l < 0 or r >= self.n:
            return None
        j = self.log[r - l + 1]
        return self.func(self.table[j][l], self.table[j][r - (1 << j) + 1])

# ULTRA: Mo's Algorithm for Range Queries
def mo_algorithm(arr, queries):
    """Answer range queries using Mo's algorithm."""
    n = len(arr)
    q = len(queries)
    if q == 0:
        return []

    block_size = max(1, int(sqrt(n)))

    # Sort queries by block, then by right endpoint
    indexed_queries = [(l, r, i) for i, (l, r) in enumerate(queries)]
    indexed_queries.sort(key=lambda x: (x[0] // block_size, x[1]))

    # Current window state
    count = {}
    distinct = 0

    def add(idx):
        nonlocal distinct
        val = arr[idx]
        if count.get(val, 0) == 0:
            distinct += 1
        count[val] = count.get(val, 0) + 1

    def remove(idx):
        nonlocal distinct
        val = arr[idx]
        count[val] -= 1
        if count[val] == 0:
            distinct -= 1

    results = [0] * q
    cur_l, cur_r = 0, -1

    for l, r, idx in indexed_queries:
        # Expand/contract window
        while cur_r < r:
            cur_r += 1
            add(cur_r)
        while cur_r > r:
            remove(cur_r)
            cur_r -= 1
        while cur_l < l:
            remove(cur_l)
            cur_l += 1
        while cur_l > l:
            cur_l -= 1
            add(cur_l)

        results[idx] = distinct

    return results

# ULTRA: Sqrt Decomposition with Updates
class SqrtDecomposition:
    def __init__(self, arr):
        self.n = len(arr)
        self.arr = arr[:]
        self.block_size = max(1, int(sqrt(self.n)))
        self.num_blocks = (self.n + self.block_size - 1) // self.block_size
        self.block_sum = [0] * self.num_blocks

        for i in range(self.n):
            self.block_sum[i // self.block_size] += arr[i]

    def update(self, idx, val):
        """Update arr[idx] = val."""
        block = idx // self.block_size
        self.block_sum[block] -= self.arr[idx]
        self.arr[idx] = val
        self.block_sum[block] += val

    def query(self, l, r):
        """Query sum of [l, r]."""
        result = 0
        start_block = l // self.block_size
        end_block = r // self.block_size

        if start_block == end_block:
            for i in range(l, r + 1):
                result += self.arr[i]
        else:
            # Left partial block
            for i in range(l, (start_block + 1) * self.block_size):
                result += self.arr[i]
            # Full blocks
            for b in range(start_block + 1, end_block):
                result += self.block_sum[b]
            # Right partial block
            for i in range(end_block * self.block_size, r + 1):
                result += self.arr[i]

        return result

# ULTRA: Range Update Point Query (Difference Array)
class DifferenceArray:
    def __init__(self, arr):
        self.n = len(arr)
        self.diff = [0] * (self.n + 1)
        for i in range(self.n):
            self.diff[i] = arr[i] - (arr[i-1] if i > 0 else 0)

    def range_add(self, l, r, val):
        """Add val to all elements in [l, r]."""
        self.diff[l] += val
        if r + 1 <= self.n:
            self.diff[r + 1] -= val

    def get(self, idx):
        """Get value at index after all updates."""
        result = 0
        for i in range(idx + 1):
            result += self.diff[i]
        return result

    def get_all(self):
        """Get entire array after updates."""
        result = []
        curr = 0
        for i in range(self.n):
            curr += self.diff[i]
            result.append(curr)
        return result

# ULTRA: 2D Prefix Sum
class PrefixSum2D:
    def __init__(self, matrix):
        self.m = len(matrix)
        self.n = len(matrix[0]) if self.m > 0 else 0
        self.prefix = [[0] * (self.n + 1) for _ in range(self.m + 1)]

        for i in range(1, self.m + 1):
            for j in range(1, self.n + 1):
                self.prefix[i][j] = (matrix[i-1][j-1] +
                                     self.prefix[i-1][j] +
                                     self.prefix[i][j-1] -
                                     self.prefix[i-1][j-1])

    def query(self, r1, c1, r2, c2):
        """Query sum of rectangle [(r1,c1), (r2,c2)]."""
        return (self.prefix[r2+1][c2+1] -
                self.prefix[r1][c2+1] -
                self.prefix[r2+1][c1] +
                self.prefix[r1][c1])

# ULTRA: XOR Range Queries with Updates
class XORSegTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (2 * self.n)
        # Build
        for i in range(self.n):
            self.tree[self.n + i] = arr[i]
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.tree[2*i] ^ self.tree[2*i+1]

    def update(self, idx, val):
        """Set arr[idx] = val."""
        idx += self.n
        self.tree[idx] = val
        while idx > 1:
            idx //= 2
            self.tree[idx] = self.tree[2*idx] ^ self.tree[2*idx+1]

    def query(self, l, r):
        """Query XOR of [l, r]."""
        result = 0
        l += self.n
        r += self.n + 1
        while l < r:
            if l & 1:
                result ^= self.tree[l]
                l += 1
            if r & 1:
                r -= 1
                result ^= self.tree[r]
            l //= 2
            r //= 2
        return result

# Tests
tests = []

# Sparse Table
arr = [1, 3, 2, 7, 9, 11, 3, 5]
st = SparseTable(arr)
tests.append(("sparse_min", st.query(0, 4), 1))
tests.append(("sparse_min2", st.query(2, 5), 2))
tests.append(("sparse_min3", st.query(4, 7), 3))

# Sparse Table Max
st_max = SparseTable(arr, max)
tests.append(("sparse_max", st_max.query(0, 4), 9))
tests.append(("sparse_max2", st_max.query(2, 5), 11))

# Mo's Algorithm
arr_mo = [1, 2, 1, 3, 2, 1]
queries_mo = [(0, 2), (1, 4), (2, 5)]
results_mo = mo_algorithm(arr_mo, queries_mo)
tests.append(("mo_dist", results_mo, [2, 3, 3]))  # distinct counts

# Sqrt Decomposition
sd = SqrtDecomposition([1, 2, 3, 4, 5, 6, 7, 8, 9])
tests.append(("sqrt_sum", sd.query(2, 6), 25))  # 3+4+5+6+7
sd.update(4, 10)  # Change 5 to 10
tests.append(("sqrt_upd", sd.query(2, 6), 30))  # 3+4+10+6+7

# Difference Array
da = DifferenceArray([1, 2, 3, 4, 5])
da.range_add(1, 3, 10)
tests.append(("diff_arr", da.get_all(), [1, 12, 13, 14, 5]))

# 2D Prefix Sum
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
ps2d = PrefixSum2D(matrix)
tests.append(("prefix2d_all", ps2d.query(0, 0, 2, 2), 45))
tests.append(("prefix2d_sub", ps2d.query(1, 1, 2, 2), 28))  # 5+6+8+9

# XOR Segment Tree
xst = XORSegTree([1, 2, 3, 4])
tests.append(("xor_query", xst.query(0, 3), 4))  # 1^2^3^4
tests.append(("xor_query2", xst.query(1, 2), 1))  # 2^3
xst.update(0, 5)
tests.append(("xor_upd", xst.query(0, 3), 0))  # 5^2^3^4

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
