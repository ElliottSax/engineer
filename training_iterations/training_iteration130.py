#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ██╗ ██████╗  ██████╗     ███╗   ███╗██╗██╗     ███████╗███████╗████████╗   ║
║  ███║ ╚════██╗██╔═████╗    ████╗ ████║██║██║     ██╔════╝██╔════╝╚══██╔══╝   ║
║  ╚██║  █████╔╝██║██╔██║    ██╔████╔██║██║██║     █████╗  ███████╗   ██║      ║
║   ██║  ╚═══██╗████╔╝██║    ██║╚██╔╝██║██║██║     ██╔══╝  ╚════██║   ██║      ║
║   ██║ ██████╔╝╚██████╔╝    ██║ ╚═╝ ██║██║███████╗███████╗███████║   ██║      ║
║   ╚═╝ ╚═════╝  ╚═════╝     ╚═╝     ╚═╝╚═╝╚══════╝╚══════╝╚══════╝   ╚═╝      ║
║                                                                              ║
║                    ULTRA TRAINING - ULTIMATE SHOWCASE                        ║
║              Comprehensive Algorithm & Data Structure Review                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import heapq
from collections import defaultdict, deque, Counter
from functools import lru_cache

# ═══════════════════════════════════════════════════════════════════════════════
#                        SECTION 1: GRAPH ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════════

def dijkstra(n, edges, source):
    """Dijkstra's shortest path algorithm."""
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))

    dist = [float('inf')] * n
    dist[source] = 0
    pq = [(0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))

    return dist

def topological_sort(n, edges):
    """Kahn's algorithm for topological sort."""
    graph = defaultdict(list)
    in_degree = [0] * n

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    queue = deque([i for i in range(n) if in_degree[i] == 0])
    result = []

    while queue:
        u = queue.popleft()
        result.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    return result if len(result) == n else []

# ═══════════════════════════════════════════════════════════════════════════════
#                      SECTION 2: DYNAMIC PROGRAMMING
# ═══════════════════════════════════════════════════════════════════════════════

def longest_increasing_subsequence(nums):
    """LIS with binary search O(n log n)."""
    from bisect import bisect_left

    if not nums:
        return 0

    dp = []
    for num in nums:
        pos = bisect_left(dp, num)
        if pos == len(dp):
            dp.append(num)
        else:
            dp[pos] = num

    return len(dp)

def edit_distance(s1, s2):
    """Levenshtein distance."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n]

# ═══════════════════════════════════════════════════════════════════════════════
#                      SECTION 3: DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class UnionFind:
    """Disjoint Set Union with path compression and union by rank."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        self.components -= 1
        return True

class SegmentTree:
    """Segment tree for range sum queries with point updates."""
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (2 * self.n)
        for i in range(self.n):
            self.tree[self.n + i] = arr[i]
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.tree[2*i] + self.tree[2*i+1]

    def update(self, i, val):
        i += self.n
        self.tree[i] = val
        while i > 1:
            i //= 2
            self.tree[i] = self.tree[2*i] + self.tree[2*i+1]

    def query(self, l, r):
        result = 0
        l += self.n
        r += self.n + 1
        while l < r:
            if l & 1:
                result += self.tree[l]
                l += 1
            if r & 1:
                r -= 1
                result += self.tree[r]
            l //= 2
            r //= 2
        return result

# ═══════════════════════════════════════════════════════════════════════════════
#                      SECTION 4: STRING ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════════

def kmp_search(text, pattern):
    """KMP string matching algorithm."""
    def compute_lps(pattern):
        m = len(pattern)
        lps = [0] * m
        length = 0
        i = 1
        while i < m:
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            elif length > 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
        return lps

    lps = compute_lps(pattern)
    matches = []
    i = j = 0

    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1
            if j == len(pattern):
                matches.append(i - j)
                j = lps[j - 1]
        elif j > 0:
            j = lps[j - 1]
        else:
            i += 1

    return matches

def longest_palindrome_substring(s):
    """Manacher's algorithm for longest palindromic substring."""
    if not s:
        return ""

    t = '#' + '#'.join(s) + '#'
    n = len(t)
    p = [0] * n
    c = r = 0

    for i in range(n):
        if i < r:
            p[i] = min(r - i, p[2 * c - i])
        while i - p[i] - 1 >= 0 and i + p[i] + 1 < n and t[i - p[i] - 1] == t[i + p[i] + 1]:
            p[i] += 1
        if i + p[i] > r:
            c, r = i, i + p[i]

    max_len = max(p)
    center = p.index(max_len)
    start = (center - max_len) // 2

    return s[start:start + max_len]

# ═══════════════════════════════════════════════════════════════════════════════
#                      SECTION 5: NUMBER THEORY
# ═══════════════════════════════════════════════════════════════════════════════

def sieve_of_eratosthenes(n):
    """Find all primes up to n."""
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(n + 1) if is_prime[i]]

def mod_exp(base, exp, mod):
    """Fast modular exponentiation."""
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        exp >>= 1
        base = (base * base) % mod
    return result

# ═══════════════════════════════════════════════════════════════════════════════
#                              TEST SUITE
# ═══════════════════════════════════════════════════════════════════════════════

def run_comprehensive_tests():
    tests = []

    print("\n  ┌─────────────────────────────────────────────────────────┐")
    print("  │              COMPREHENSIVE TEST SUITE                   │")
    print("  └─────────────────────────────────────────────────────────┘\n")

    # ═══════ GRAPH TESTS ═══════
    print("  📊 Graph Algorithms")
    print("  " + "─" * 55)

    edges = [(0, 1, 4), (0, 2, 1), (2, 1, 2), (1, 3, 1), (2, 3, 5)]
    dist = dijkstra(4, edges, 0)
    tests.append(("  Dijkstra shortest path", dist[3], 4))

    dag_edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    topo = topological_sort(4, dag_edges)
    tests.append(("  Topological sort valid", len(topo), 4))

    # ═══════ DP TESTS ═══════
    print("\n  🧮 Dynamic Programming")
    print("  " + "─" * 55)

    tests.append(("  LIS length", longest_increasing_subsequence([10, 9, 2, 5, 3, 7, 101, 18]), 4))
    tests.append(("  Edit distance", edit_distance("kitten", "sitting"), 3))

    # ═══════ DATA STRUCTURE TESTS ═══════
    print("\n  🏗️  Data Structures")
    print("  " + "─" * 55)

    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(2, 3)
    uf.union(0, 2)
    tests.append(("  Union-Find connected", uf.find(0) == uf.find(3), True))
    tests.append(("  Union-Find components", uf.components, 2))

    st = SegmentTree([1, 3, 5, 7, 9, 11])
    tests.append(("  Segment tree query", st.query(1, 4), 24))
    st.update(2, 10)
    tests.append(("  Segment tree update", st.query(1, 4), 29))

    # ═══════ STRING TESTS ═══════
    print("\n  📝 String Algorithms")
    print("  " + "─" * 55)

    tests.append(("  KMP matches", kmp_search("abcabcabc", "abc"), [0, 3, 6]))
    tests.append(("  Longest palindrome", longest_palindrome_substring("babad") in ["bab", "aba"], True))

    # ═══════ NUMBER THEORY TESTS ═══════
    print("\n  🔢 Number Theory")
    print("  " + "─" * 55)

    primes = sieve_of_eratosthenes(30)
    tests.append(("  Sieve primes", primes, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]))
    tests.append(("  Mod exponentiation", mod_exp(2, 10, 1000), 24))

    # ═══════ RUN ALL TESTS ═══════
    passed = 0
    for name, result, expected in tests:
        if result == expected:
            passed += 1
            print(f"    ✅ {name}")
        else:
            print(f"    ❌ {name}: got {result}, expected {expected}")

    return passed, len(tests)

if __name__ == "__main__":
    print(__doc__)

    passed, total = run_comprehensive_tests()

    print("\n  ╔═════════════════════════════════════════════════════════╗")
    print(f"  ║  📊 FINAL RESULTS: {passed}/{total} tests passed" + " " * (35 - len(str(passed)) - len(str(total))) + "║")
    print("  ╠═════════════════════════════════════════════════════════╣")

    if passed == total:
        print("  ║                                                         ║")
        print("  ║   🏆 MILESTONE 130 ACHIEVED - PERFECT SCORE! 🏆        ║")
        print("  ║                                                         ║")
        print("  ║   Algorithms Mastered:                                  ║")
        print("  ║   • Graph: Dijkstra, Topological Sort                   ║")
        print("  ║   • DP: LIS, Edit Distance                              ║")
        print("  ║   • DS: Union-Find, Segment Tree                        ║")
        print("  ║   • String: KMP, Manacher                               ║")
        print("  ║   • Math: Sieve, Modular Exponentiation                 ║")
        print("  ║                                                         ║")
    else:
        print(f"  ║   Keep training! {total - passed} tests remaining.              ║")

    print("  ╚═════════════════════════════════════════════════════════╝")
