from functools import lru_cache

def climbing_stairs(n):
    """Number of ways to climb n stairs."""
    if n <= 2:
        return n
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b

def tribonacci(n):
    """Nth tribonacci number."""
    if n == 0:
        return 0
    if n <= 2:
        return 1
    a, b, c = 0, 1, 1
    for _ in range(3, n + 1):
        a, b, c = b, c, a + b + c
    return c

def unique_paths_with_memo(m, n):
    """Unique paths in grid with memoization."""
    @lru_cache(None)
    def dp(i, j):
        if i == 0 or j == 0:
            return 1
        return dp(i - 1, j) + dp(i, j - 1)
    return dp(m - 1, n - 1)

def longest_common_subseq_memo(text1, text2):
    """LCS with memoization."""
    @lru_cache(None)
    def dp(i, j):
        if i < 0 or j < 0:
            return 0
        if text1[i] == text2[j]:
            return 1 + dp(i - 1, j - 1)
        return max(dp(i - 1, j), dp(i, j - 1))
    return dp(len(text1) - 1, len(text2) - 1)

def pow_recursive(x, n):
    """Calculate x^n."""
    if n == 0:
        return 1
    if n < 0:
        return 1 / pow_recursive(x, -n)
    if n % 2 == 0:
        half = pow_recursive(x, n // 2)
        return half * half
    return x * pow_recursive(x, n - 1)

def generate_trees(n):
    """Generate all structurally unique BSTs."""
    def generate(start, end):
        if start > end:
            return [None]
        trees = []
        for root_val in range(start, end + 1):
            left_trees = generate(start, root_val - 1)
            right_trees = generate(root_val + 1, end)
            for left in left_trees:
                for right in right_trees:
                    trees.append({'val': root_val, 'left': left, 'right': right})
        return trees
    return generate(1, n) if n > 0 else []

def num_trees(n):
    """Count structurally unique BSTs."""
    @lru_cache(None)
    def count(n):
        if n <= 1:
            return 1
        total = 0
        for i in range(1, n + 1):
            total += count(i - 1) * count(n - i)
        return total
    return count(n)

def regular_expression_matching_memo(s, p):
    """Regex matching with memoization."""
    @lru_cache(None)
    def dp(i, j):
        if j == len(p):
            return i == len(s)
        first_match = i < len(s) and (p[j] == '.' or p[j] == s[i])
        if j + 1 < len(p) and p[j + 1] == '*':
            return dp(i, j + 2) or (first_match and dp(i + 1, j))
        return first_match and dp(i + 1, j + 1)
    return dp(0, 0)

def min_cost_for_tickets(days, costs):
    """Minimum cost for train tickets."""
    day_set = set(days)
    last_day = days[-1]

    @lru_cache(None)
    def dp(day):
        if day > last_day:
            return 0
        if day not in day_set:
            return dp(day + 1)
        return min(
            costs[0] + dp(day + 1),
            costs[1] + dp(day + 7),
            costs[2] + dp(day + 30)
        )

    return dp(1)

def decode_ways_memo(s):
    """Decode ways with memoization."""
    @lru_cache(None)
    def dp(i):
        if i == len(s):
            return 1
        if s[i] == '0':
            return 0
        result = dp(i + 1)
        if i + 1 < len(s) and int(s[i:i+2]) <= 26:
            result += dp(i + 2)
        return result
    return dp(0) if s else 0

def knight_probability(n, k, row, col):
    """Probability knight stays on board after k moves."""
    moves = [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]

    @lru_cache(None)
    def dp(r, c, remaining):
        if r < 0 or r >= n or c < 0 or c >= n:
            return 0
        if remaining == 0:
            return 1
        prob = 0
        for dr, dc in moves:
            prob += dp(r + dr, c + dc, remaining - 1) / 8
        return prob

    return dp(row, col, k)

def champagne_tower(poured, query_row, query_glass):
    """Amount of champagne in glass."""
    dp = [[0] * 101 for _ in range(101)]
    dp[0][0] = poured

    for row in range(query_row + 1):
        for col in range(row + 1):
            overflow = (dp[row][col] - 1) / 2
            if overflow > 0:
                dp[row + 1][col] += overflow
                dp[row + 1][col + 1] += overflow

    return min(1, dp[query_row][query_glass])

# Tests
tests = [
    ("stairs", climbing_stairs(4), 5),
    ("tribonacci", tribonacci(25), 1389537),
    ("unique_paths", unique_paths_with_memo(3, 7), 28),
    ("lcs", longest_common_subseq_memo("abcde", "ace"), 3),
    ("pow_pos", round(pow_recursive(2.0, 10), 5), 1024.0),
    ("pow_neg", round(pow_recursive(2.0, -2), 5), 0.25),
    ("num_trees", num_trees(4), 14),
    ("regex", regular_expression_matching_memo("aa", "a*"), True),
    ("tickets", min_cost_for_tickets([1,4,6,7,8,20], [2,7,15]), 11),
    ("decode", decode_ways_memo("226"), 3),
    ("knight", round(knight_probability(3, 2, 0, 0), 5), 0.0625),
    ("champagne", round(champagne_tower(2, 1, 1), 5), 0.5),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
