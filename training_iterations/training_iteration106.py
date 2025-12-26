# ULTRA: Competitive Programming Classic Hard Problems

from collections import defaultdict, deque
import heapq
from functools import lru_cache

# ULTRA: Maximum Rectangle in Histogram
def largest_rectangle_histogram(heights):
    """Largest rectangle in histogram."""
    stack = []
    max_area = 0
    index = 0

    while index < len(heights):
        if not stack or heights[index] >= heights[stack[-1]]:
            stack.append(index)
            index += 1
        else:
            top = stack.pop()
            width = index if not stack else index - stack[-1] - 1
            max_area = max(max_area, heights[top] * width)

    while stack:
        top = stack.pop()
        width = index if not stack else index - stack[-1] - 1
        max_area = max(max_area, heights[top] * width)

    return max_area

# ULTRA: Maximal Rectangle in Binary Matrix
def maximal_rectangle(matrix):
    """Largest rectangle containing only 1s."""
    if not matrix or not matrix[0]:
        return 0

    m, n = len(matrix), len(matrix[0])
    heights = [0] * n
    max_area = 0

    for i in range(m):
        for j in range(n):
            heights[j] = heights[j] + 1 if matrix[i][j] == 1 else 0
        max_area = max(max_area, largest_rectangle_histogram(heights))

    return max_area

# ULTRA: Trapping Rain Water 2D
def trap_rain_water_2d(height_map):
    """3D rain water trapping."""
    if not height_map or not height_map[0]:
        return 0

    m, n = len(height_map), len(height_map[0])
    visited = [[False] * n for _ in range(m)]
    heap = []

    # Add boundary
    for i in range(m):
        for j in range(n):
            if i == 0 or i == m - 1 or j == 0 or j == n - 1:
                heapq.heappush(heap, (height_map[i][j], i, j))
                visited[i][j] = True

    water = 0
    while heap:
        h, r, c = heapq.heappop(heap)
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and not visited[nr][nc]:
                visited[nr][nc] = True
                water += max(0, h - height_map[nr][nc])
                heapq.heappush(heap, (max(h, height_map[nr][nc]), nr, nc))

    return water

# ULTRA: Russian Doll Envelopes
def max_envelopes(envelopes):
    """Maximum number of envelopes that can be nested."""
    if not envelopes:
        return 0

    # Sort by width ascending, then height descending
    envelopes.sort(key=lambda x: (x[0], -x[1]))

    # LIS on heights
    from bisect import bisect_left
    dp = []
    for _, h in envelopes:
        pos = bisect_left(dp, h)
        if pos == len(dp):
            dp.append(h)
        else:
            dp[pos] = h

    return len(dp)

# ULTRA: Best Time to Buy/Sell Stock with Cooldown
def max_profit_cooldown(prices):
    """Maximum profit with cooldown period."""
    if len(prices) < 2:
        return 0

    n = len(prices)
    # hold[i] = max profit on day i if holding stock
    # sold[i] = max profit on day i if just sold
    # rest[i] = max profit on day i if resting

    hold = float('-inf')
    sold = 0
    rest = 0

    for price in prices:
        prev_hold = hold
        hold = max(hold, rest - price)
        rest = max(rest, sold)
        sold = prev_hold + price

    return max(sold, rest)

# ULTRA: Best Time to Buy/Sell Stock IV
def max_profit_k_transactions(k, prices):
    """Maximum profit with at most k transactions."""
    n = len(prices)
    if n < 2 or k == 0:
        return 0

    # If k >= n/2, unlimited transactions
    if k >= n // 2:
        return sum(max(0, prices[i+1] - prices[i]) for i in range(n-1))

    # dp[i][j] = max profit using at most i transactions up to day j
    dp = [[0] * n for _ in range(k + 1)]

    for i in range(1, k + 1):
        max_diff = -prices[0]
        for j in range(1, n):
            dp[i][j] = max(dp[i][j-1], prices[j] + max_diff)
            max_diff = max(max_diff, dp[i-1][j] - prices[j])

    return dp[k][n-1]

# ULTRA: Longest Valid Parentheses
def longest_valid_parentheses(s):
    """Length of longest valid parentheses substring."""
    stack = [-1]
    max_len = 0

    for i, c in enumerate(s):
        if c == '(':
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)
            else:
                max_len = max(max_len, i - stack[-1])

    return max_len

# ULTRA: Minimum Window Substring
def min_window_substring(s, t):
    """Minimum window in s containing all characters of t."""
    if not t or not s:
        return ""

    from collections import Counter
    t_count = Counter(t)
    required = len(t_count)

    left = 0
    formed = 0
    window_counts = {}
    ans = (float('inf'), None, None)

    for right, char in enumerate(s):
        window_counts[char] = window_counts.get(char, 0) + 1

        if char in t_count and window_counts[char] == t_count[char]:
            formed += 1

        while left <= right and formed == required:
            if right - left + 1 < ans[0]:
                ans = (right - left + 1, left, right)

            left_char = s[left]
            window_counts[left_char] -= 1
            if left_char in t_count and window_counts[left_char] < t_count[left_char]:
                formed -= 1
            left += 1

    return "" if ans[0] == float('inf') else s[ans[1]:ans[2]+1]

# ULTRA: Median of Two Sorted Arrays
def find_median_sorted_arrays(nums1, nums2):
    """Find median of two sorted arrays in O(log(min(m,n)))."""
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    m, n = len(nums1), len(nums2)
    left, right = 0, m

    while left <= right:
        partition1 = (left + right) // 2
        partition2 = (m + n + 1) // 2 - partition1

        max_left1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
        min_right1 = float('inf') if partition1 == m else nums1[partition1]

        max_left2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
        min_right2 = float('inf') if partition2 == n else nums2[partition2]

        if max_left1 <= min_right2 and max_left2 <= min_right1:
            if (m + n) % 2 == 0:
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2
            else:
                return max(max_left1, max_left2)
        elif max_left1 > min_right2:
            right = partition1 - 1
        else:
            left = partition1 + 1

    return 0.0

# Tests
tests = []

# Largest rectangle histogram
tests.append(("histogram", largest_rectangle_histogram([2,1,5,6,2,3]), 10))

# Maximal rectangle
matrix = [
    [1, 0, 1, 0, 0],
    [1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 0, 0, 1, 0]
]
tests.append(("max_rect", maximal_rectangle(matrix), 6))

# Trap water 2D
height_map = [
    [1, 4, 3, 1, 3, 2],
    [3, 2, 1, 3, 2, 4],
    [2, 3, 3, 2, 3, 1]
]
tests.append(("trap_2d", trap_rain_water_2d(height_map), 4))

# Russian dolls
tests.append(("envelopes", max_envelopes([[5,4],[6,4],[6,7],[2,3]]), 3))

# Stock cooldown
tests.append(("stock_cool", max_profit_cooldown([1,2,3,0,2]), 3))

# Stock k transactions
tests.append(("stock_k", max_profit_k_transactions(2, [2,4,1]), 2))
tests.append(("stock_k2", max_profit_k_transactions(2, [3,2,6,5,0,3]), 7))

# Valid parentheses
tests.append(("valid_paren", longest_valid_parentheses("(()"), 2))
tests.append(("valid_paren2", longest_valid_parentheses(")()())"), 4))

# Min window
tests.append(("min_window", min_window_substring("ADOBECODEBANC", "ABC"), "BANC"))

# Median
tests.append(("median", find_median_sorted_arrays([1, 3], [2]), 2.0))
tests.append(("median2", find_median_sorted_arrays([1, 2], [3, 4]), 2.5))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
