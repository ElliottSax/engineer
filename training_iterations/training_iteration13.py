def min_edit_distance(word1, word2):
    """Minimum edit distance (insert, delete, replace)."""
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]

def longest_palindromic_subsequence(s):
    """Length of longest palindromic subsequence."""
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    for i in range(n - 1, -1, -1):
        dp[i][i] = 1
        for j in range(i + 1, n):
            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1] + 2
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])
    return dp[0][n-1]

def interleaving_string(s1, s2, s3):
    """Can s3 be formed by interleaving s1 and s2?"""
    m, n = len(s1), len(s2)
    if m + n != len(s3):
        return False
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1]
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] and s2[j-1] == s3[j-1]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = (dp[i-1][j] and s1[i-1] == s3[i+j-1]) or \
                       (dp[i][j-1] and s2[j-1] == s3[i+j-1])
    return dp[m][n]

def burst_balloons(nums):
    """Maximum coins from bursting all balloons."""
    nums = [1] + nums + [1]
    n = len(nums)
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n):
        for left in range(n - length):
            right = left + length
            for k in range(left + 1, right):
                dp[left][right] = max(dp[left][right],
                    dp[left][k] + dp[k][right] + nums[left] * nums[k] * nums[right])
    return dp[0][n-1]

def russian_doll_envelopes(envelopes):
    """Maximum number of envelopes that can be nested."""
    from bisect import bisect_left
    if not envelopes:
        return 0
    # Sort by width ascending, then height descending
    envelopes.sort(key=lambda x: (x[0], -x[1]))
    heights = [e[1] for e in envelopes]
    # LIS on heights
    tails = []
    for h in heights:
        pos = bisect_left(tails, h)
        if pos == len(tails):
            tails.append(h)
        else:
            tails[pos] = h
    return len(tails)

def best_time_buy_sell_cooldown(prices):
    """Max profit with cooldown (must wait 1 day after selling)."""
    if len(prices) < 2:
        return 0
    sold = 0
    held = float('-inf')
    reset = 0
    for price in prices:
        prev_sold = sold
        sold = held + price
        held = max(held, reset - price)
        reset = max(reset, prev_sold)
    return max(sold, reset)

def distinct_subsequences(s, t):
    """Number of distinct subsequences of s equal to t."""
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = 1
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j]
            if s[i-1] == t[j-1]:
                dp[i][j] += dp[i-1][j-1]
    return dp[m][n]

def super_egg_drop(k, n):
    """Minimum moves to find critical floor with k eggs, n floors."""
    dp = [[0] * (n + 1) for _ in range(k + 1)]
    for i in range(1, n + 1):
        dp[1][i] = i
    for eggs in range(2, k + 1):
        for floors in range(1, n + 1):
            dp[eggs][floors] = float('inf')
            lo, hi = 1, floors
            while lo <= hi:
                mid = (lo + hi) // 2
                breaks = dp[eggs - 1][mid - 1]
                survives = dp[eggs][floors - mid]
                if breaks > survives:
                    hi = mid - 1
                    dp[eggs][floors] = min(dp[eggs][floors], 1 + breaks)
                else:
                    lo = mid + 1
                    dp[eggs][floors] = min(dp[eggs][floors], 1 + survives)
    return dp[k][n]

def perfect_squares(n):
    """Minimum perfect squares that sum to n."""
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    for i in range(1, n + 1):
        j = 1
        while j * j <= i:
            dp[i] = min(dp[i], dp[i - j * j] + 1)
            j += 1
    return dp[n]

def ugly_number_ii(n):
    """Returns nth ugly number (factors only 2, 3, 5)."""
    ugly = [1]
    i2 = i3 = i5 = 0
    while len(ugly) < n:
        next2 = ugly[i2] * 2
        next3 = ugly[i3] * 3
        next5 = ugly[i5] * 5
        next_ugly = min(next2, next3, next5)
        ugly.append(next_ugly)
        if next_ugly == next2:
            i2 += 1
        if next_ugly == next3:
            i3 += 1
        if next_ugly == next5:
            i5 += 1
    return ugly[-1]

# Tests
tests = [
    ("edit_distance", min_edit_distance("horse", "ros"), 3),
    ("lps", longest_palindromic_subsequence("bbbab"), 4),
    ("interleave_yes", interleaving_string("aabcc", "dbbca", "aadbbcbcac"), True),
    ("interleave_no", interleaving_string("aabcc", "dbbca", "aadbbbaccc"), False),
    ("burst_balloons", burst_balloons([3,1,5,8]), 167),
    ("russian_dolls", russian_doll_envelopes([[5,4],[6,4],[6,7],[2,3]]), 3),
    ("cooldown", best_time_buy_sell_cooldown([1,2,3,0,2]), 3),
    ("distinct_subseq", distinct_subsequences("rabbbit", "rabbit"), 3),
    ("egg_drop", super_egg_drop(2, 6), 3),
    ("egg_drop_2", super_egg_drop(1, 10), 10),
    ("perfect_squares", perfect_squares(12), 3),
    ("ugly_10", ugly_number_ii(10), 12),
    ("ugly_1", ugly_number_ii(1), 1),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
