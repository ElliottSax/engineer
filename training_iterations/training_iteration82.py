# EXTREME: Competitive Programming Level Problems

from collections import defaultdict, deque
import heapq

# HARD: Palindrome Partitioning II - Minimum Cuts
def min_cut_palindrome(s):
    """Minimum cuts for palindrome partitioning."""
    n = len(s)
    # is_pal[i][j] = True if s[i:j+1] is palindrome
    is_pal = [[False] * n for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for j in range(i, n):
            if s[i] == s[j] and (j - i < 2 or is_pal[i + 1][j - 1]):
                is_pal[i][j] = True

    # dp[i] = min cuts for s[0:i+1]
    dp = list(range(n))
    for i in range(1, n):
        if is_pal[0][i]:
            dp[i] = 0
        else:
            for j in range(i):
                if is_pal[j + 1][i]:
                    dp[i] = min(dp[i], dp[j] + 1)
    return dp[n - 1]

# HARD: Distinct Subsequences II
def distinct_subsequences_ii(s):
    """Count distinct non-empty subsequences."""
    MOD = 10**9 + 7
    dp = [0] * 26
    for c in s:
        idx = ord(c) - ord('a')
        dp[idx] = (sum(dp) + 1) % MOD
    return sum(dp) % MOD

# HARD: Frog Jump
def can_frog_cross(stones):
    """Check if frog can cross to last stone."""
    if stones[1] != 1:
        return False
    stone_set = set(stones)
    target = stones[-1]
    memo = {}

    def dp(pos, k):
        if pos == target:
            return True
        if (pos, k) in memo:
            return memo[(pos, k)]
        for jump in [k - 1, k, k + 1]:
            if jump > 0 and pos + jump in stone_set:
                if dp(pos + jump, jump):
                    memo[(pos, k)] = True
                    return True
        memo[(pos, k)] = False
        return False

    return dp(1, 1)

# HARD: Strange Printer
def strange_printer(s):
    """Minimum turns to print string."""
    # Remove consecutive duplicates
    s = ''.join(c for i, c in enumerate(s) if i == 0 or c != s[i-1])
    n = len(s)
    if n == 0:
        return 0

    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 1

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = dp[i][j - 1] + 1
            for k in range(i, j):
                if s[k] == s[j]:
                    dp[i][j] = min(dp[i][j], dp[i][k] + (dp[k + 1][j - 1] if k + 1 <= j - 1 else 0))

    return dp[0][n - 1]

# HARD: Super Egg Drop
def super_egg_drop(k, n):
    """Minimum moves to find critical floor."""
    # dp[m][k] = max floors checkable with m moves and k eggs
    dp = [[0] * (k + 1) for _ in range(n + 1)]
    m = 0
    while dp[m][k] < n:
        m += 1
        for j in range(1, k + 1):
            dp[m][j] = dp[m - 1][j - 1] + dp[m - 1][j] + 1
    return m

# HARD: Number of Music Playlists
def num_music_playlists(n, goal, k):
    """Count playlists with n songs, goal length, k gap."""
    MOD = 10**9 + 7
    # dp[i][j] = playlists of length i with j unique songs
    dp = [[0] * (n + 1) for _ in range(goal + 1)]
    dp[0][0] = 1

    for i in range(1, goal + 1):
        for j in range(1, min(i, n) + 1):
            # Add new song
            dp[i][j] = dp[i - 1][j - 1] * (n - j + 1) % MOD
            # Replay old song (must have played k songs since)
            if j > k:
                dp[i][j] = (dp[i][j] + dp[i - 1][j] * (j - k)) % MOD

    return dp[goal][n]

# HARD: Count Vowels Permutation
def count_vowel_permutation(n):
    """Count strings of length n with vowel rules."""
    MOD = 10**9 + 7
    # a, e, i, o, u
    dp = [1, 1, 1, 1, 1]

    for _ in range(n - 1):
        new_dp = [
            dp[1] + dp[2] + dp[4],  # a follows e, i, u
            dp[0] + dp[2],          # e follows a, i
            dp[1] + dp[3],          # i follows e, o
            dp[2],                   # o follows i
            dp[2] + dp[3]           # u follows i, o
        ]
        dp = [x % MOD for x in new_dp]

    return sum(dp) % MOD

# HARD: Tallest Billboard
def tallest_billboard(rods):
    """Maximum height of two equal supports."""
    dp = {0: 0}  # diff -> max shorter support

    for rod in rods:
        new_dp = dp.copy()
        for diff, shorter in dp.items():
            # Add to taller
            new_dp[diff + rod] = max(new_dp.get(diff + rod, 0), shorter)
            # Add to shorter
            new_diff = abs(diff - rod)
            new_shorter = shorter + min(diff, rod)
            new_dp[new_diff] = max(new_dp.get(new_diff, 0), new_shorter)
        dp = new_dp

    return dp[0]

# HARD: Minimum Cost to Merge Stones
def merge_stones(stones, k):
    """Minimum cost to merge all stones into one pile."""
    n = len(stones)
    if (n - 1) % (k - 1) != 0:
        return -1

    prefix = [0]
    for s in stones:
        prefix.append(prefix[-1] + s)

    dp = [[0] * n for _ in range(n)]

    for length in range(k, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for mid in range(i, j, k - 1):
                dp[i][j] = min(dp[i][j], dp[i][mid] + dp[mid + 1][j])
            if (length - 1) % (k - 1) == 0:
                dp[i][j] += prefix[j + 1] - prefix[i]

    return dp[0][n - 1]

# HARD: Minimum Number of Refueling Stops
def min_refuel_stops(target, tank, stations):
    """Minimum stops to reach target."""
    heap = []
    stops = 0
    prev = 0

    for pos, fuel in stations + [[target, 0]]:
        tank -= pos - prev
        while heap and tank < 0:
            tank += -heapq.heappop(heap)
            stops += 1
        if tank < 0:
            return -1
        heapq.heappush(heap, -fuel)
        prev = pos

    return stops

# Tests
tests = [
    ("min_cut", min_cut_palindrome("aab"), 1),
    ("min_cut_pal", min_cut_palindrome("aba"), 0),
    ("min_cut_hard", min_cut_palindrome("aabba"), 1),
    ("distinct_sub", distinct_subsequences_ii("abc"), 7),
    ("distinct_dup", distinct_subsequences_ii("aba"), 6),
    ("frog", can_frog_cross([0,1,3,5,6,8,12,17]), True),
    ("frog_no", can_frog_cross([0,1,2,3,4,8,9,11]), False),
    ("printer", strange_printer("aaabbb"), 2),
    ("printer_hard", strange_printer("aba"), 2),
    ("egg", super_egg_drop(2, 6), 3),
    ("egg_hard", super_egg_drop(3, 14), 4),
    ("playlist", num_music_playlists(3, 3, 1), 6),
    ("vowel", count_vowel_permutation(2), 10),
    ("vowel_long", count_vowel_permutation(5), 68),
    ("billboard", tallest_billboard([1,2,3,6]), 6),
    ("merge_stones", merge_stones([3,2,4,1], 2), 20),
    ("refuel", min_refuel_stops(100, 10, [[10,60],[20,30],[30,30],[60,40]]), 2),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
