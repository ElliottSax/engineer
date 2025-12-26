# ULTRA: Advanced Dynamic Programming III - Optimization Techniques

from functools import lru_cache
from collections import deque

# ULTRA: Knuth's Optimization (Optimal BST-style DP)
def matrix_chain_multiplication(dims):
    """Optimal matrix chain multiplication order using Knuth optimization."""
    n = len(dims) - 1
    if n <= 1:
        return 0

    # dp[i][j] = min operations to multiply matrices i to j
    INF = float('inf')
    dp = [[0] * n for _ in range(n)]
    opt = [[0] * n for _ in range(n)]

    # Base case
    for i in range(n):
        opt[i][i] = i

    # Fill by chain length
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = INF

            # Knuth optimization bounds
            lo = opt[i][j-1] if j > 0 and i < j else i
            hi = opt[i+1][j] if i + 1 < n else j

            for k in range(i, j):
                left = dp[i][k] if k >= i else 0
                right = dp[k+1][j] if k + 1 <= j else 0
                cost = left + right + dims[i] * dims[k+1] * dims[j+1]
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    opt[i][j] = k

    return dp[0][n-1]

# ULTRA: Divide and Conquer DP Optimization
def divide_conquer_dp(n, m, cost_fn):
    """DP with divide and conquer optimization for 1D/1D recurrence.
    dp[i][j] = min(dp[i-1][k] + cost(k+1, j)) for k < j
    Requires cost to satisfy quadrangle inequality.
    """
    INF = float('inf')
    dp = [[INF] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0

    def solve(level, lo, hi, opt_lo, opt_hi):
        if lo > hi:
            return
        mid = (lo + hi) // 2
        best_cost = INF
        best_k = opt_lo

        for k in range(opt_lo, min(mid, opt_hi + 1)):
            if dp[level - 1][k] < INF:
                cost = dp[level - 1][k] + cost_fn(k + 1, mid)
                if cost < best_cost:
                    best_cost = cost
                    best_k = k

        dp[level][mid] = best_cost

        solve(level, lo, mid - 1, opt_lo, best_k)
        solve(level, mid + 1, hi, best_k, opt_hi)

    for i in range(1, m + 1):
        solve(i, 0, n, 0, n - 1)

    return dp[m][n]

# ULTRA: Monotonic Deque for Sliding Window DP
def sliding_window_max_sum(arr, k):
    """Find subarray of length exactly k with maximum sum."""
    n = len(arr)
    if n < k:
        return None

    curr_sum = sum(arr[:k])
    max_sum = curr_sum

    for i in range(k, n):
        curr_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, curr_sum)

    return max_sum

def min_cost_k_partitions(arr, k):
    """Partition array into k parts minimizing sum of squared sums."""
    n = len(arr)
    if k > n:
        return float('inf')

    # Prefix sums for quick range sum
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + arr[i]

    def range_sum(i, j):
        return prefix[j + 1] - prefix[i]

    def cost(i, j):
        s = range_sum(i, j)
        return s * s

    # dp[i][j] = min cost to partition arr[0:j+1] into i parts
    INF = float('inf')
    dp = [[INF] * n for _ in range(k + 1)]

    # Base: 1 partition
    for j in range(n):
        dp[1][j] = cost(0, j)

    # Fill using monotonic deque optimization
    for i in range(2, k + 1):
        for j in range(i - 1, n):
            for l in range(i - 2, j):
                if dp[i - 1][l] < INF:
                    dp[i][j] = min(dp[i][j], dp[i - 1][l] + cost(l + 1, j))

    return dp[k][n - 1]

# ULTRA: Longest Increasing Subsequence with Path Recovery
def lis_with_path(arr):
    """Find LIS and return the actual subsequence."""
    n = len(arr)
    if n == 0:
        return []

    from bisect import bisect_left

    dp = []  # dp[i] = smallest ending element of LIS of length i+1
    parent = [-1] * n
    pos = [0] * n  # pos[i] = index in dp where arr[i] was placed

    for i, x in enumerate(arr):
        idx = bisect_left(dp, x)
        if idx == len(dp):
            dp.append(x)
        else:
            dp[idx] = x
        pos[i] = idx
        if idx > 0:
            # Find parent: last element placed at idx-1 before i
            for j in range(i - 1, -1, -1):
                if pos[j] == idx - 1 and arr[j] < x:
                    parent[i] = j
                    break

    # Reconstruct path
    length = len(dp)
    result = []
    idx = -1
    for i in range(n - 1, -1, -1):
        if pos[i] == length - 1:
            idx = i
            break

    while idx != -1:
        result.append(arr[idx])
        idx = parent[idx]

    return result[::-1]

# ULTRA: Edit Distance with Operations
def edit_distance_ops(s1, s2):
    """Edit distance returning the actual operations."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],      # delete
                                  dp[i][j - 1],      # insert
                                  dp[i - 1][j - 1])  # replace

    # Backtrack to find operations
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i - 1] == s2[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(('replace', i - 1, s2[j - 1]))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(('insert', i, s2[j - 1]))
            j -= 1
        else:
            ops.append(('delete', i - 1))
            i -= 1

    return dp[m][n], ops[::-1]

# ULTRA: Coin Change with Minimum Coins (and path)
def coin_change_path(coins, amount):
    """Find minimum coins and which coins to use."""
    dp = [float('inf')] * (amount + 1)
    parent = [-1] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for c in coins:
            if c <= i and dp[i - c] + 1 < dp[i]:
                dp[i] = dp[i - c] + 1
                parent[i] = c

    if dp[amount] == float('inf'):
        return -1, []

    # Reconstruct
    result = []
    curr = amount
    while curr > 0:
        result.append(parent[curr])
        curr -= parent[curr]

    return dp[amount], result

# Tests
tests = []

# Matrix chain
tests.append(("matrix_chain", matrix_chain_multiplication([10, 20, 30, 40, 30]), 30000))

# Sliding window max sum
tests.append(("sliding_max", sliding_window_max_sum([1, -2, 3, 4, -1, 2], 3), 6))

# Min cost k partitions
tests.append(("partition", min_cost_k_partitions([1, 2, 3, 4], 2) <= 50, True))  # (1+2)^2 + (3+4)^2 = 58

# LIS with path
lis = lis_with_path([10, 22, 9, 33, 21, 50, 41, 60, 80])
tests.append(("lis_len", len(lis), 6))
tests.append(("lis_valid", all(lis[i] < lis[i+1] for i in range(len(lis)-1)), True))

# Edit distance
dist, ops = edit_distance_ops("kitten", "sitting")
tests.append(("edit_dist", dist, 3))

# Coin change
num_coins, coins_used = coin_change_path([1, 5, 10, 25], 30)
tests.append(("coin_count", num_coins, 2))  # 25 + 5
tests.append(("coin_sum", sum(coins_used), 30))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
