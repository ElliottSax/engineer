def coin_change(coins, amount):
    """Minimum coins to make amount."""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] = min(dp[x], dp[x - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1

def coin_change_ways(coins, amount):
    """Number of ways to make amount."""
    dp = [0] * (amount + 1)
    dp[0] = 1

    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] += dp[x - coin]

    return dp[amount]

def unbounded_knapsack(weights, values, capacity):
    """Unbounded knapsack - items can be reused."""
    dp = [0] * (capacity + 1)

    for w in range(1, capacity + 1):
        for i in range(len(weights)):
            if weights[i] <= w:
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[capacity]

def partition_equal_sum(nums):
    """Check if array can be partitioned into two equal sum subsets."""
    total = sum(nums)
    if total % 2:
        return False

    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True

    for num in nums:
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]

    return dp[target]

def target_sum(nums, target):
    """Count ways to assign +/- to reach target."""
    total = sum(nums)
    if (total + target) % 2 or abs(target) > total:
        return 0

    subset_sum = (total + target) // 2
    dp = [0] * (subset_sum + 1)
    dp[0] = 1

    for num in nums:
        for j in range(subset_sum, num - 1, -1):
            dp[j] += dp[j - num]

    return dp[subset_sum]

def word_break(s, word_dict):
    """Check if string can be segmented into dictionary words."""
    word_set = set(word_dict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True

    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break

    return dp[n]

def perfect_squares(n):
    """Minimum perfect squares summing to n."""
    dp = [float('inf')] * (n + 1)
    dp[0] = 0

    for i in range(1, n + 1):
        j = 1
        while j * j <= i:
            dp[i] = min(dp[i], dp[i - j*j] + 1)
            j += 1

    return dp[n]

def integer_break(n):
    """Maximum product of integers summing to n."""
    dp = [0] * (n + 1)
    dp[1] = 1

    for i in range(2, n + 1):
        for j in range(1, i):
            dp[i] = max(dp[i], max(j, dp[j]) * max(i - j, dp[i - j]))

    return dp[n]

def unique_paths(m, n):
    """Number of unique paths in m x n grid."""
    dp = [[1] * n for _ in range(m)]

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]

    return dp[m-1][n-1]

def unique_paths_with_obstacles(grid):
    """Unique paths with obstacles."""
    m, n = len(grid), len(grid[0])
    if grid[0][0] or grid[m-1][n-1]:
        return 0

    dp = [[0] * n for _ in range(m)]
    dp[0][0] = 1

    for j in range(1, n):
        dp[0][j] = 0 if grid[0][j] else dp[0][j-1]
    for i in range(1, m):
        dp[i][0] = 0 if grid[i][0] else dp[i-1][0]

    for i in range(1, m):
        for j in range(1, n):
            if grid[i][j]:
                dp[i][j] = 0
            else:
                dp[i][j] = dp[i-1][j] + dp[i][j-1]

    return dp[m-1][n-1]

def min_path_sum(grid):
    """Minimum path sum from top-left to bottom-right."""
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]

    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]

    return dp[m-1][n-1]

def triangle_min_path(triangle):
    """Minimum path sum in triangle."""
    n = len(triangle)
    dp = triangle[-1][:]

    for i in range(n - 2, -1, -1):
        for j in range(len(triangle[i])):
            dp[j] = triangle[i][j] + min(dp[j], dp[j+1])

    return dp[0]

def max_product_subarray(nums):
    """Maximum product subarray."""
    max_prod = min_prod = result = nums[0]

    for num in nums[1:]:
        if num < 0:
            max_prod, min_prod = min_prod, max_prod
        max_prod = max(num, max_prod * num)
        min_prod = min(num, min_prod * num)
        result = max(result, max_prod)

    return result

# Tests
tests = [
    ("coin_min", coin_change([1,2,5], 11), 3),
    ("coin_min_no", coin_change([2], 3), -1),
    ("coin_ways", coin_change_ways([1,2,5], 5), 4),
    ("unbounded", unbounded_knapsack([1,3,4,5], [10,40,50,70], 8), 110),
    ("partition", partition_equal_sum([1,5,11,5]), True),
    ("partition_no", partition_equal_sum([1,2,3,5]), False),
    ("target_sum", target_sum([1,1,1,1,1], 3), 5),
    ("word_break", word_break("leetcode", ["leet","code"]), True),
    ("word_break_no", word_break("catsandog", ["cats","dog","sand","and","cat"]), False),
    ("perfect_sq", perfect_squares(12), 3),
    ("int_break", integer_break(10), 36),
    ("unique_paths", unique_paths(3, 7), 28),
    ("unique_obs", unique_paths_with_obstacles([[0,0,0],[0,1,0],[0,0,0]]), 2),
    ("min_path", min_path_sum([[1,3,1],[1,5,1],[4,2,1]]), 7),
    ("triangle", triangle_min_path([[2],[3,4],[6,5,7],[4,1,8,3]]), 11),
    ("max_product", max_product_subarray([2,3,-2,4]), 6),
    ("max_product_2", max_product_subarray([-2,0,-1]), 0),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
