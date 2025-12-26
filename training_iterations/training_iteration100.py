# MILESTONE ITERATION 100: Comprehensive Algorithm Challenge

from collections import defaultdict, deque
import heapq
from functools import lru_cache

# ===== SECTION 1: ADVANCED GRAPH =====

def tarjan_scc(n, edges):
    """Tarjan's SCC algorithm."""
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)

    index_counter = [0]
    stack = []
    lowlink = {}
    index = {}
    on_stack = {}
    sccs = []

    def strongconnect(v):
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True

        for w in graph[v]:
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack.get(w, False):
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            scc = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == v:
                    break
            sccs.append(sorted(scc))

    for v in range(n):
        if v not in index:
            strongconnect(v)

    return sorted(sccs)

# ===== SECTION 2: ADVANCED STRING =====

def aho_corasick(patterns, text):
    """Aho-Corasick multi-pattern matching."""
    # Build trie with failure links
    goto = [{}]
    fail = [0]
    output = [[]]

    for pattern_idx, pattern in enumerate(patterns):
        state = 0
        for char in pattern:
            if char not in goto[state]:
                goto[state][char] = len(goto)
                goto.append({})
                fail.append(0)
                output.append([])
            state = goto[state][char]
        output[state].append(pattern_idx)

    # Build failure function
    queue = deque()
    for char, state in goto[0].items():
        queue.append(state)

    while queue:
        r = queue.popleft()
        for char, s in goto[r].items():
            queue.append(s)
            state = fail[r]
            while state and char not in goto[state]:
                state = fail[state]
            fail[s] = goto[state].get(char, 0)
            output[s] = output[s] + output[fail[s]]

    # Search
    matches = [[] for _ in patterns]
    state = 0
    for i, char in enumerate(text):
        while state and char not in goto[state]:
            state = fail[state]
        state = goto[state].get(char, 0)
        for pattern_idx in output[state]:
            matches[pattern_idx].append(i - len(patterns[pattern_idx]) + 1)

    return matches

# ===== SECTION 3: ADVANCED DP =====

def longest_common_subsequence_3(s1, s2, s3):
    """LCS of three strings."""
    l1, l2, l3 = len(s1), len(s2), len(s3)
    dp = [[[0] * (l3 + 1) for _ in range(l2 + 1)] for _ in range(l1 + 1)]

    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            for k in range(1, l3 + 1):
                if s1[i-1] == s2[j-1] == s3[k-1]:
                    dp[i][j][k] = dp[i-1][j-1][k-1] + 1
                else:
                    dp[i][j][k] = max(dp[i-1][j][k], dp[i][j-1][k], dp[i][j][k-1])

    return dp[l1][l2][l3]

def optimal_bst(keys, freq):
    """Optimal binary search tree cost."""
    n = len(keys)
    dp = [[0] * n for _ in range(n)]
    sum_freq = [[0] * n for _ in range(n)]

    for i in range(n):
        dp[i][i] = freq[i]
        sum_freq[i][i] = freq[i]

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            sum_freq[i][j] = sum_freq[i][j-1] + freq[j]
            dp[i][j] = float('inf')
            for r in range(i, j + 1):
                left = dp[i][r-1] if r > i else 0
                right = dp[r+1][j] if r < j else 0
                cost = left + right + sum_freq[i][j]
                dp[i][j] = min(dp[i][j], cost)

    return dp[0][n-1]

# ===== SECTION 4: COMPUTATIONAL GEOMETRY =====

def line_segment_intersection_point(p1, p2, p3, p4):
    """Find intersection point of two line segments."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (round(x, 6), round(y, 6))
    return None

# ===== SECTION 5: NUMBER THEORY =====

def discrete_log(g, h, p):
    """Baby-step giant-step for discrete log."""
    m = int(p ** 0.5) + 1

    # Baby step
    table = {}
    val = 1
    for j in range(m):
        if val == h:
            return j
        table[val] = j
        val = val * g % p

    # Giant step
    factor = pow(g, m * (p - 2), p)
    gamma = h
    for i in range(m):
        if gamma in table:
            return i * m + table[gamma]
        gamma = gamma * factor % p

    return None

def primitive_root(p):
    """Find primitive root modulo p."""
    if p == 2:
        return 1

    phi = p - 1
    factors = []
    n = phi
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)

    for g in range(2, p):
        is_root = True
        for f in factors:
            if pow(g, phi // f, p) == 1:
                is_root = False
                break
        if is_root:
            return g

    return None

# ===== SECTION 6: ADVANCED DATA STRUCTURES =====

class MergeSortTree:
    """Segment tree with sorted arrays for range queries."""
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [[] for _ in range(4 * self.n)]
        if self.n > 0:
            self._build(arr, 0, 0, self.n - 1)

    def _build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = [arr[start]]
        else:
            mid = (start + end) // 2
            self._build(arr, 2*node+1, start, mid)
            self._build(arr, 2*node+2, mid+1, end)
            # Merge sorted arrays
            self.tree[node] = self._merge(self.tree[2*node+1], self.tree[2*node+2])

    def _merge(self, a, b):
        result = []
        i = j = 0
        while i < len(a) and j < len(b):
            if a[i] <= b[j]:
                result.append(a[i])
                i += 1
            else:
                result.append(b[j])
                j += 1
        result.extend(a[i:])
        result.extend(b[j:])
        return result

    def count_less_than(self, l, r, k):
        """Count elements < k in range [l, r]."""
        return self._query(0, 0, self.n-1, l, r, k)

    def _query(self, node, start, end, l, r, k):
        if r < start or end < l:
            return 0
        if l <= start and end <= r:
            # Binary search for count < k
            lo, hi = 0, len(self.tree[node])
            while lo < hi:
                mid = (lo + hi) // 2
                if self.tree[node][mid] < k:
                    lo = mid + 1
                else:
                    hi = mid
            return lo
        mid = (start + end) // 2
        return self._query(2*node+1, start, mid, l, r, k) + \
               self._query(2*node+2, mid+1, end, l, r, k)

# ===== TESTS =====

tests = []

# Tarjan SCC
tests.append(("tarjan", tarjan_scc(5, [(0,1),(1,2),(2,0),(1,3),(3,4)]),
              [[0,1,2], [3], [4]]))

# Aho-Corasick
matches = aho_corasick(["he", "she", "his", "hers"], "ushers")
tests.append(("aho_corasick", matches, [[2], [1], [], [2]]))

# LCS of 3 strings
tests.append(("lcs3", longest_common_subsequence_3("abc", "cab", "bac"), 1))
tests.append(("lcs3_2", longest_common_subsequence_3("abcd", "bcda", "cdab"), 2))

# Optimal BST
tests.append(("obst", optimal_bst([10, 20, 30], [3, 2, 4]), 17))

# Line segment intersection
p = line_segment_intersection_point((0, 0), (4, 4), (0, 4), (4, 0))
tests.append(("line_intersect", p, (2.0, 2.0)))
tests.append(("line_no_intersect", line_segment_intersection_point((0, 0), (1, 1), (2, 2), (3, 3)), None))

# Discrete log
tests.append(("dlog", discrete_log(2, 3, 5), 3))  # 2^3 = 8 ≡ 3 (mod 5)

# Primitive root
tests.append(("prim_root", primitive_root(7) in [3, 5], True))

# Merge Sort Tree
mst = MergeSortTree([3, 1, 4, 1, 5, 9, 2, 6])
tests.append(("mst_count", mst.count_less_than(0, 4, 4), 3))  # 1, 1, 3 are < 4
tests.append(("mst_count2", mst.count_less_than(2, 6, 5), 3))  # 1, 4, 2 are < 5

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
print("\n🎉 ITERATION 100 COMPLETE! 🎉")
