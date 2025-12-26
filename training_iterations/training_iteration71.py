from collections import deque
import heapq

def sliding_window_maximum(nums, k):
    """Maximum in each sliding window of size k."""
    result = []
    dq = deque()

    for i, num in enumerate(nums):
        while dq and nums[dq[-1]] < num:
            dq.pop()
        dq.append(i)

        if dq[0] <= i - k:
            dq.popleft()

        if i >= k - 1:
            result.append(nums[dq[0]])

    return result

def sliding_window_median(nums, k):
    """Median in each sliding window of size k."""
    import bisect

    window = sorted(nums[:k])
    result = []

    def get_median():
        if k % 2:
            return float(window[k // 2])
        return (window[k // 2 - 1] + window[k // 2]) / 2

    result.append(get_median())

    for i in range(k, len(nums)):
        # Remove outgoing element
        window.pop(bisect.bisect_left(window, nums[i - k]))
        # Insert incoming element
        bisect.insort(window, nums[i])
        result.append(get_median())

    return result

def contains_duplicate_iii(nums, k, t):
    """Contains duplicate within k indices and value diff <= t."""
    if t < 0:
        return False

    buckets = {}
    bucket_size = t + 1

    for i, num in enumerate(nums):
        bucket_id = num // bucket_size

        if bucket_id in buckets:
            return True
        if bucket_id - 1 in buckets and num - buckets[bucket_id - 1] <= t:
            return True
        if bucket_id + 1 in buckets and buckets[bucket_id + 1] - num <= t:
            return True

        buckets[bucket_id] = num

        if i >= k:
            del buckets[nums[i - k] // bucket_size]

    return False

def max_sliding_window_min_sum(nums, k):
    """Sum of minimums of all subarrays of size k."""
    dq = deque()
    result = 0
    window_min_sum = 0

    for i, num in enumerate(nums):
        while dq and nums[dq[-1]] > num:
            dq.pop()
        dq.append(i)

        if dq[0] <= i - k:
            dq.popleft()

        if i >= k - 1:
            result += nums[dq[0]]

    return result

def longest_subarray_absolute_diff(nums, limit):
    """Longest subarray with max absolute diff <= limit."""
    min_dq = deque()
    max_dq = deque()
    left = 0
    result = 0

    for right, num in enumerate(nums):
        while min_dq and nums[min_dq[-1]] > num:
            min_dq.pop()
        min_dq.append(right)

        while max_dq and nums[max_dq[-1]] < num:
            max_dq.pop()
        max_dq.append(right)

        while nums[max_dq[0]] - nums[min_dq[0]] > limit:
            left += 1
            if min_dq[0] < left:
                min_dq.popleft()
            if max_dq[0] < left:
                max_dq.popleft()

        result = max(result, right - left + 1)

    return result

def shortest_subarray_sum_at_least_k(nums, k):
    """Shortest subarray with sum >= k."""
    n = len(nums)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + nums[i]

    dq = deque()
    result = float('inf')

    for i in range(n + 1):
        while dq and prefix[i] - prefix[dq[0]] >= k:
            result = min(result, i - dq.popleft())
        while dq and prefix[i] <= prefix[dq[-1]]:
            dq.pop()
        dq.append(i)

    return result if result != float('inf') else -1

def constrained_subsequence_sum(nums, k):
    """Maximum sum subsequence with adjacent indices diff <= k."""
    n = len(nums)
    dp = nums[:]
    dq = deque([0])

    for i in range(1, n):
        while dq and dq[0] < i - k:
            dq.popleft()
        dp[i] = max(dp[i], dp[dq[0]] + nums[i])
        while dq and dp[dq[-1]] <= dp[i]:
            dq.pop()
        dq.append(i)

    return max(dp)

def jump_game_vi(nums, k):
    """Maximum score reaching end with max k jumps."""
    n = len(nums)
    dp = [float('-inf')] * n
    dp[0] = nums[0]
    dq = deque([0])

    for i in range(1, n):
        while dq and dq[0] < i - k:
            dq.popleft()
        dp[i] = dp[dq[0]] + nums[i]
        while dq and dp[dq[-1]] <= dp[i]:
            dq.pop()
        dq.append(i)

    return dp[-1]

def max_result_jump_game(nums, k):
    """Same as jump_game_vi."""
    return jump_game_vi(nums, k)

def max_sum_of_rectangle_no_larger_than_k(matrix, k):
    """Maximum sum rectangle <= k."""
    import bisect

    if not matrix:
        return 0

    m, n = len(matrix), len(matrix[0])
    result = float('-inf')

    for left in range(n):
        row_sum = [0] * m
        for right in range(left, n):
            for i in range(m):
                row_sum[i] += matrix[i][right]

            # Find max subarray sum <= k
            prefix_sums = [0]
            curr_sum = 0
            for s in row_sum:
                curr_sum += s
                idx = bisect.bisect_left(prefix_sums, curr_sum - k)
                if idx < len(prefix_sums):
                    result = max(result, curr_sum - prefix_sums[idx])
                bisect.insort(prefix_sums, curr_sum)

    return result

# Tests
tests = [
    ("sliding_max", sliding_window_maximum([1,3,-1,-3,5,3,6,7], 3), [3,3,5,5,6,7]),
    ("sliding_median", sliding_window_median([1,3,-1,-3,5,3,6,7], 3), [1.0,-1.0,-1.0,3.0,5.0,6.0]),
    ("dup_iii_yes", contains_duplicate_iii([1,2,3,1], 3, 0), True),
    ("dup_iii_no", contains_duplicate_iii([1,5,9,1,5,9], 2, 3), False),
    ("min_sum", max_sliding_window_min_sum([1,3,2,4], 2), 6),
    ("longest_diff", longest_subarray_absolute_diff([8,2,4,7], 4), 2),
    ("shortest_sum", shortest_subarray_sum_at_least_k([2,-1,2], 3), 3),
    ("constrained", constrained_subsequence_sum([10,2,-10,5,20], 2), 37),
    ("jump_vi", jump_game_vi([1,-1,-2,4,-7,3], 2), 7),
    ("max_rect", max_sum_of_rectangle_no_larger_than_k([[1,0,1],[0,-2,3]], 2), 2),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
