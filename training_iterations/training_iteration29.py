def shortest_common_supersequence(str1, str2):
    """Shortest string containing both as subsequences."""
    m, n = len(str1), len(str2)
    # LCS DP
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    # Reconstruct
    result = []
    i, j = m, n
    while i > 0 and j > 0:
        if str1[i-1] == str2[j-1]:
            result.append(str1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            result.append(str1[i-1])
            i -= 1
        else:
            result.append(str2[j-1])
            j -= 1
    while i > 0:
        result.append(str1[i-1])
        i -= 1
    while j > 0:
        result.append(str2[j-1])
        j -= 1
    return ''.join(reversed(result))

def maxdotproduct_of_two_subseq(nums1, nums2):
    """Maximum dot product of non-empty subsequences."""
    m, n = len(nums1), len(nums2)
    dp = [[float('-inf')] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            product = nums1[i-1] * nums2[j-1]
            dp[i][j] = max(
                product,
                dp[i-1][j],
                dp[i][j-1],
                dp[i-1][j-1] + product,
                product + max(0, dp[i-1][j-1])
            )
    return dp[m][n]

def longest_arithmetic_subsequence(arr):
    """Length of longest arithmetic subsequence."""
    n = len(arr)
    dp = [{} for _ in range(n)]
    max_len = 2
    for i in range(n):
        for j in range(i):
            diff = arr[i] - arr[j]
            dp[i][diff] = dp[j].get(diff, 1) + 1
            max_len = max(max_len, dp[i][diff])
    return max_len

def longest_arithmetic_subseq_diff(arr, difference):
    """Longest arithmetic subsequence with given difference."""
    dp = {}
    max_len = 0
    for num in arr:
        dp[num] = dp.get(num - difference, 0) + 1
        max_len = max(max_len, dp[num])
    return max_len

def number_of_lis(nums):
    """Number of longest increasing subsequences."""
    n = len(nums)
    if n == 0:
        return 0
    length = [1] * n
    count = [1] * n
    for i in range(n):
        for j in range(i):
            if nums[j] < nums[i]:
                if length[j] + 1 > length[i]:
                    length[i] = length[j] + 1
                    count[i] = count[j]
                elif length[j] + 1 == length[i]:
                    count[i] += count[j]
    max_len = max(length)
    return sum(c for l, c in zip(length, count) if l == max_len)

def largest_divisible_subset(nums):
    """Largest subset where every pair is divisible."""
    if not nums:
        return []
    nums.sort()
    n = len(nums)
    dp = [1] * n
    parent = [-1] * n
    max_idx = 0
    for i in range(n):
        for j in range(i):
            if nums[i] % nums[j] == 0 and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
        if dp[i] > dp[max_idx]:
            max_idx = i
    result = []
    while max_idx != -1:
        result.append(nums[max_idx])
        max_idx = parent[max_idx]
    return result[::-1]

def delete_and_earn(nums):
    """Maximum points from deleting elements."""
    if not nums:
        return 0
    max_num = max(nums)
    points = [0] * (max_num + 1)
    for num in nums:
        points[num] += num
    take, skip = 0, 0
    for i in range(max_num + 1):
        take, skip = skip + points[i], max(take, skip)
    return max(take, skip)

def integer_break(n):
    """Maximum product after breaking n into sum of positive integers."""
    if n == 2:
        return 1
    if n == 3:
        return 2
    product = 1
    while n > 4:
        product *= 3
        n -= 3
    return product * n

def last_stone_weight_ii(stones):
    """Minimum possible weight of last stone after smashing."""
    total = sum(stones)
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    for stone in stones:
        for j in range(target, stone - 1, -1):
            dp[j] = dp[j] or dp[j - stone]
    for i in range(target, -1, -1):
        if dp[i]:
            return total - 2 * i
    return total

def ones_and_zeroes(strs, m, n):
    """Maximum strings with at most m 0s and n 1s."""
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for s in strs:
        zeros = s.count('0')
        ones = len(s) - zeros
        for i in range(m, zeros - 1, -1):
            for j in range(n, ones - 1, -1):
                dp[i][j] = max(dp[i][j], 1 + dp[i - zeros][j - ones])
    return dp[m][n]

# Tests
tests = [
    ("scs", len(shortest_common_supersequence("abac", "cab")), 5),
    ("max_dot", maxdotproduct_of_two_subseq([2,1,-2,5], [3,0,-6]), 18),
    ("max_dot_neg", maxdotproduct_of_two_subseq([-1,-1], [1,1]), -1),
    ("longest_arith", longest_arithmetic_subsequence([3,6,9,12]), 4),
    ("longest_arith_2", longest_arithmetic_subsequence([9,4,7,2,10]), 3),
    ("arith_diff", longest_arithmetic_subseq_diff([1,2,3,4], 1), 4),
    ("arith_diff_2", longest_arithmetic_subseq_diff([1,3,5,7], 1), 1),
    ("num_lis", number_of_lis([1,3,5,4,7]), 2),
    ("num_lis_2", number_of_lis([2,2,2,2,2]), 5),
    ("divisible", largest_divisible_subset([1,2,3]), [1, 2]),
    ("divisible_2", largest_divisible_subset([1,2,4,8]), [1, 2, 4, 8]),
    ("delete_earn", delete_and_earn([3,4,2]), 6),
    ("delete_earn_2", delete_and_earn([2,2,3,3,3,4]), 9),
    ("int_break", integer_break(10), 36),
    ("int_break_2", integer_break(2), 1),
    ("last_stone", last_stone_weight_ii([2,7,4,1,8,1]), 1),
    ("ones_zeros", ones_and_zeroes(["10","0001","111001","1","0"], 5, 3), 4),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
