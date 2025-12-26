# EXTREME: Advanced DP - Bitmask & Digit DP

from functools import lru_cache

# HARD: Traveling Salesman Problem (Bitmask DP)
def tsp(dist):
    """Minimum cost to visit all cities and return to start."""
    n = len(dist)
    if n == 0:
        return 0

    INF = float('inf')
    # dp[mask][i] = min cost to visit cities in mask, ending at i
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0  # Start at city 0

    for mask in range(1 << n):
        for u in range(n):
            if not (mask & (1 << u)):
                continue
            if dp[mask][u] == INF:
                continue
            for v in range(n):
                if mask & (1 << v):
                    continue
                new_mask = mask | (1 << v)
                dp[new_mask][v] = min(dp[new_mask][v], dp[mask][u] + dist[u][v])

    # Return to start
    full_mask = (1 << n) - 1
    return min(dp[full_mask][i] + dist[i][0] for i in range(n))

# HARD: Hamiltonian Path (Bitmask DP)
def has_hamiltonian_path(graph):
    """Check if graph has Hamiltonian path."""
    n = len(graph)
    if n == 0:
        return True

    # dp[mask][i] = True if can visit cities in mask, ending at i
    dp = [[False] * n for _ in range(1 << n)]

    for i in range(n):
        dp[1 << i][i] = True

    for mask in range(1 << n):
        for u in range(n):
            if not (mask & (1 << u)) or not dp[mask][u]:
                continue
            for v in graph[u]:
                if mask & (1 << v):
                    continue
                dp[mask | (1 << v)][v] = True

    full_mask = (1 << n) - 1
    return any(dp[full_mask][i] for i in range(n))

# HARD: Minimum Vertex Cover (Bitmask)
def min_vertex_cover(n, edges):
    """Minimum vertex cover using bitmask."""
    for size in range(n + 1):
        for mask in range(1 << n):
            if bin(mask).count('1') != size:
                continue
            if all((mask & (1 << u)) or (mask & (1 << v)) for u, v in edges):
                return size
    return n

# HARD: Maximum Independent Set (Bitmask)
def max_independent_set(n, edges):
    """Maximum independent set using bitmask."""
    # Convert edges to adjacency
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    max_size = 0
    for mask in range(1 << n):
        valid = True
        for i in range(n):
            if (mask & (1 << i)) and (mask & adj[i]):
                valid = False
                break
        if valid:
            max_size = max(max_size, bin(mask).count('1'))

    return max_size

# HARD: Count Numbers with Unique Digits (Digit DP)
def count_unique_digits(n):
    """Count numbers with unique digits from 0 to 10^n - 1."""
    if n == 0:
        return 1
    if n == 1:
        return 10

    count = 10  # 0-9
    unique = 9
    available = 9

    for i in range(2, n + 1):
        unique *= available
        count += unique
        available -= 1

    return count

# HARD: Numbers At Most N Given Digit Set (Digit DP)
def at_most_n_given_digits(digits, n):
    """Count numbers formed from digits that are <= n."""
    s = str(n)
    k = len(s)
    d = len(digits)

    @lru_cache(maxsize=None)
    def dp(pos, tight, started):
        if pos == k:
            return 1 if started else 0

        limit = int(s[pos]) if tight else 9
        result = 0

        if not started:
            result += dp(pos + 1, False, False)  # Skip

        for digit in digits:
            digit = int(digit)
            if digit > limit:
                break
            if digit == 0 and not started:
                continue
            result += dp(pos + 1, tight and digit == limit, True)

        return result

    return dp(0, True, False)

# HARD: Count Special Integers (Distinct Digits)
def count_special_integers(n):
    """Count positive integers <= n with distinct digits."""
    s = str(n)
    k = len(s)

    @lru_cache(maxsize=None)
    def dp(pos, mask, tight, started):
        if pos == k:
            return 1 if started else 0

        limit = int(s[pos]) if tight else 9
        result = 0

        if not started:
            result += dp(pos + 1, mask, False, False)

        for digit in range(0 if started else 1, limit + 1):
            if mask & (1 << digit):
                continue
            result += dp(pos + 1, mask | (1 << digit), tight and digit == limit, True)

        return result

    return dp(0, 0, True, False)

# HARD: Numbers With Repeated Digits (Digit DP)
def num_with_repeated_digits(n):
    """Count numbers with at least one repeated digit."""
    return n - count_special_integers(n)

# HARD: Count Digit Sum = S (Digit DP)
def count_digit_sum(n, target_sum):
    """Count numbers <= n with digit sum = target_sum."""
    s = str(n)
    k = len(s)

    @lru_cache(maxsize=None)
    def dp(pos, curr_sum, tight, started):
        if curr_sum > target_sum:
            return 0
        if pos == k:
            return 1 if curr_sum == target_sum and started else 0

        limit = int(s[pos]) if tight else 9
        result = 0

        for digit in range(0, limit + 1):
            new_started = started or digit > 0
            result += dp(pos + 1, curr_sum + digit, tight and digit == limit, new_started)

        return result

    return dp(0, 0, True, False)

# Tests
tests = []

# TSP
dist = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]
tests.append(("tsp", tsp(dist), 80))

# Hamiltonian Path
graph = [[1, 2], [0, 2], [0, 1, 3], [2]]
tests.append(("ham_yes", has_hamiltonian_path(graph), True))
graph_no = [[1], [0], [3], [2]]  # Two components
tests.append(("ham_no", has_hamiltonian_path(graph_no), False))

# Vertex Cover
tests.append(("vertex_cover", min_vertex_cover(4, [(0,1), (0,2), (1,3), (2,3)]), 2))

# Independent Set
tests.append(("indep_set", max_independent_set(4, [(0,1), (1,2), (2,3)]), 2))

# Unique Digits
tests.append(("unique_2", count_unique_digits(2), 91))
tests.append(("unique_3", count_unique_digits(3), 739))

# At Most N Given Digits
tests.append(("at_most", at_most_n_given_digits(["1", "3", "5", "7"], 100), 20))

# Special Integers
tests.append(("special", count_special_integers(20), 19))
tests.append(("special_100", count_special_integers(100), 90))

# Repeated Digits
tests.append(("repeated", num_with_repeated_digits(100), 10))

# Digit Sum
tests.append(("digit_sum", count_digit_sum(100, 5), 6))  # 5, 14, 23, 32, 41, 50

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
