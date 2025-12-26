# HARD: Suffix Array O(n log^2 n) + Complex Problems

def suffix_array(s):
    """Build suffix array O(n log^2 n)."""
    n = len(s)
    sa = list(range(n))
    rank = [ord(c) for c in s]

    k = 1
    while k < n:
        def key(i):
            return (rank[i], rank[i + k] if i + k < n else -1)
        sa.sort(key=key)
        new_rank = [0] * n
        for i in range(1, n):
            new_rank[sa[i]] = new_rank[sa[i-1]] + (1 if key(sa[i]) != key(sa[i-1]) else 0)
        rank = new_rank
        k *= 2
    return sa

def kasai_lcp(s, sa):
    """Kasai's algorithm for LCP array."""
    n = len(s)
    rank = [0] * n
    for i in range(n):
        rank[sa[i]] = i
    lcp = [0] * n
    k = 0
    for i in range(n):
        if rank[i] == 0:
            k = 0
            continue
        j = sa[rank[i] - 1]
        while i + k < n and j + k < n and s[i + k] == s[j + k]:
            k += 1
        lcp[rank[i]] = k
        if k > 0:
            k -= 1
    return lcp

def longest_repeated_k_times(s, k):
    """Find longest substring appearing at least k times."""
    if not s or k <= 0:
        return ""
    if k == 1:
        return s
    n = len(s)
    sa = suffix_array(s)
    lcp = kasai_lcp(s, sa)

    from collections import deque
    dq = deque()
    max_len = 0
    start = 0

    for i in range(1, n):
        while dq and lcp[dq[-1]] >= lcp[i]:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            while dq and dq[0] <= i - k + 1:
                dq.popleft()
            if dq:
                min_lcp = lcp[dq[0]]
                if min_lcp > max_len:
                    max_len = min_lcp
                    start = sa[i]

    return s[start:start + max_len] if max_len > 0 else ""

# HARD: Count of Range Sum with merge sort
def count_range_sum_hard(nums, lower, upper):
    """Count range sums in [lower, upper] - O(n log n)."""
    prefix = [0]
    for num in nums:
        prefix.append(prefix[-1] + num)

    def merge_count(arr, start, end):
        if end - start <= 1:
            return 0
        mid = (start + end) // 2
        count = merge_count(arr, start, mid) + merge_count(arr, mid, end)

        j = k = mid
        for i in range(start, mid):
            while j < end and arr[j] - arr[i] < lower:
                j += 1
            while k < end and arr[k] - arr[i] <= upper:
                k += 1
            count += k - j

        arr[start:end] = sorted(arr[start:end])
        return count

    return merge_count(prefix[:], 0, len(prefix))

# HARD: Minimum Window Subsequence
def min_window_subsequence(s1, s2):
    """Find minimum window in s1 containing s2 as subsequence."""
    m, n = len(s1), len(s2)
    if n == 0:
        return ""
    if m < n:
        return ""

    dp = [[-1] * n for _ in range(m)]

    for i in range(m):
        if s1[i] == s2[0]:
            dp[i][0] = i
        elif i > 0:
            dp[i][0] = dp[i-1][0]

    for j in range(1, n):
        for i in range(j, m):
            if s1[i] == s2[j]:
                dp[i][j] = dp[i-1][j-1] if i > 0 else -1
            else:
                dp[i][j] = dp[i-1][j] if i > 0 else -1

    min_len = float('inf')
    result = ""
    for i in range(n - 1, m):
        if dp[i][n-1] != -1:
            length = i - dp[i][n-1] + 1
            if length < min_len:
                min_len = length
                result = s1[dp[i][n-1]:i+1]

    return result

# HARD: Maximum Sum of 3 Non-Overlapping Subarrays
def max_sum_3_subarrays(nums, k):
    """Find 3 non-overlapping subarrays of size k with max sum."""
    n = len(nums)
    prefix = [0]
    for num in nums:
        prefix.append(prefix[-1] + num)

    def get_sum(i):
        return prefix[i + k] - prefix[i]

    left = [0] * n
    best = 0
    for i in range(k - 1, n):
        if get_sum(i - k + 1) > get_sum(best):
            best = i - k + 1
        left[i] = best

    right = [0] * n
    best = n - k
    for i in range(n - k, -1, -1):
        if get_sum(i) >= get_sum(best):
            best = i
        right[i] = best

    result = [0, k, 2 * k]
    max_sum = 0
    for mid in range(k, n - 2 * k + 1):
        l, r = left[mid - 1], right[mid + k]
        total = get_sum(l) + get_sum(mid) + get_sum(r)
        if total > max_sum:
            max_sum = total
            result = [l, mid, r]

    return result

# HARD: Smallest Range Covering Elements from K Lists
def smallest_range(nums):
    """Find smallest range including at least one number from each list."""
    import heapq
    heap = []
    max_val = float('-inf')

    for i, arr in enumerate(nums):
        heapq.heappush(heap, (arr[0], i, 0))
        max_val = max(max_val, arr[0])

    result = [float('-inf'), float('inf')]

    while True:
        min_val, list_idx, elem_idx = heapq.heappop(heap)
        if max_val - min_val < result[1] - result[0]:
            result = [min_val, max_val]
        if elem_idx + 1 == len(nums[list_idx]):
            break
        next_val = nums[list_idx][elem_idx + 1]
        max_val = max(max_val, next_val)
        heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))

    return result

# HARD: Split Array Largest Sum with K splits
def split_array_k_ways(nums, k):
    """Minimize largest sum when splitting into k parts."""
    def can_split(max_sum):
        count = 1
        curr = 0
        for num in nums:
            if curr + num > max_sum:
                count += 1
                curr = num
            else:
                curr += num
        return count <= k

    left, right = max(nums), sum(nums)
    while left < right:
        mid = (left + right) // 2
        if can_split(mid):
            right = mid
        else:
            left = mid + 1
    return left

# HARD: Burst Balloons
def max_coins_balloons(nums):
    """Maximum coins from bursting balloons."""
    nums = [1] + nums + [1]
    n = len(nums)
    dp = [[0] * n for _ in range(n)]

    for length in range(2, n):
        for left in range(n - length):
            right = left + length
            for k in range(left + 1, right):
                dp[left][right] = max(
                    dp[left][right],
                    dp[left][k] + nums[left] * nums[k] * nums[right] + dp[k][right]
                )

    return dp[0][n - 1]

# Tests
tests = [
    ("sa_basic", suffix_array("banana"), [5, 3, 1, 0, 4, 2]),
    ("lcp_basic", kasai_lcp("banana", [5, 3, 1, 0, 4, 2]), [0, 1, 3, 0, 0, 2]),
    ("repeat_2", longest_repeated_k_times("abcabcabc", 3), "abc"),
    ("repeat_4", longest_repeated_k_times("aaaa", 4), "a"),
    ("repeat_none", longest_repeated_k_times("abcd", 2), ""),
    ("range_sum", count_range_sum_hard([-2, 5, -1], -2, 2), 3),
    ("range_hard", count_range_sum_hard([0, -1, 1, 2, -3, -3], -3, 1), 14),
    ("min_win_sub", min_window_subsequence("abcdebdde", "bde"), "bcde"),
    ("max_3_sub", max_sum_3_subarrays([1,2,1,2,6,7,5,1], 2), [0, 3, 5]),
    ("small_range", smallest_range([[4,10,15,24,26],[0,9,12,20],[5,18,22,30]]), [20, 24]),
    ("split_k", split_array_k_ways([7,2,5,10,8], 2), 18),
    ("balloons", max_coins_balloons([3,1,5,8]), 167),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
