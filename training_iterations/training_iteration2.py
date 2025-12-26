def longest_common_subsequence(s1, s2):
    """Finds the longest common subsequence of two strings."""
    m, n = len(s1), len(s2)
    dp = [["" for _ in range(n + 1)] for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + s1[i-1]
            else:
                dp[i][j] = dp[i-1][j] if len(dp[i-1][j]) > len(dp[i][j-1]) else dp[i][j-1]
    return dp[m][n]

def edit_distance(s1, s2):
    """Calculates minimum edit distance between two strings."""
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

def knapsack(weights, values, capacity):
    """Solves 0/1 knapsack problem."""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], values[i-1] + dp[i-1][w - weights[i-1]])
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][capacity]

def longest_increasing_subsequence(arr):
    """Returns length of longest increasing subsequence."""
    if not arr:
        return 0
    n = len(arr)
    dp = [1] * n
    for i in range(1, n):
        for j in range(i):
            if arr[j] < arr[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

def detect_cycle(graph):
    """Detects if directed graph contains a cycle."""
    visited = set()
    rec_stack = set()

    def dfs(node):
        visited.add(node)
        rec_stack.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        rec_stack.remove(node)
        return False

    for node in graph:
        if node not in visited:
            if dfs(node):
                return True
    return False

def three_sum(nums):
    """Finds all unique triplets that sum to zero."""
    nums.sort()
    result = []
    n = len(nums)
    for i in range(n - 2):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        left, right = i + 1, n - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1
    return result

def trap_water(heights):
    """Calculates trapped rainwater between bars."""
    if not heights:
        return 0
    n = len(heights)
    left_max = [0] * n
    right_max = [0] * n
    left_max[0] = heights[0]
    for i in range(1, n):
        left_max[i] = max(left_max[i-1], heights[i])
    right_max[n-1] = heights[n-1]
    for i in range(n-2, -1, -1):
        right_max[i] = max(right_max[i+1], heights[i])
    water = 0
    for i in range(n):
        water += min(left_max[i], right_max[i]) - heights[i]
    return water

# Tests
tests = [
    ("LCS", longest_common_subsequence("ABCDGH", "AEDFHR"), "ADH"),
    ("edit_distance", edit_distance("kitten", "sitting"), 3),
    ("knapsack", knapsack([10,20,30], [60,100,120], 50), 220),
    ("LIS", longest_increasing_subsequence([10,9,2,5,3,7,101,18]), 4),
    ("cycle_yes", detect_cycle({0:[1], 1:[2], 2:[0]}), True),
    ("cycle_no", detect_cycle({0:[1], 1:[2], 2:[]}), False),
    ("three_sum", three_sum([-1,0,1,2,-1,-4]), [[-1,-1,2],[-1,0,1]]),
    ("trap_water", trap_water([0,1,0,2,1,0,1,3,2,1,2,1]), 6),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}: {result}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
