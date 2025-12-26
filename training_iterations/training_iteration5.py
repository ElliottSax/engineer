def serialize_tree(root):
    """Serializes binary tree to string."""
    if root is None:
        return "null"
    return f"{root['val']},{serialize_tree(root.get('left'))},{serialize_tree(root.get('right'))}"

def deserialize_tree(data):
    """Deserializes string to binary tree."""
    def helper(nodes):
        val = next(nodes)
        if val == "null":
            return None
        return {'val': int(val), 'left': helper(nodes), 'right': helper(nodes)}
    return helper(iter(data.split(',')))

def unique_paths(m, n):
    """Counts unique paths in m x n grid from top-left to bottom-right."""
    dp = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[m-1][n-1]

def count_islands(grid):
    """Counts number of islands (connected 1s) in 2D grid."""
    if not grid or not grid[0]:
        return 0
    rows, cols = len(grid), len(grid[0])
    count = 0
    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != '1':
            return
        grid[r][c] = '0'
        dfs(r+1, c); dfs(r-1, c); dfs(r, c+1); dfs(r, c-1)
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1':
                dfs(i, j)
                count += 1
    return count

def max_product_subarray(nums):
    """Finds maximum product of contiguous subarray."""
    if not nums:
        return 0
    max_prod = min_prod = result = nums[0]
    for num in nums[1:]:
        if num < 0:
            max_prod, min_prod = min_prod, max_prod
        max_prod = max(num, max_prod * num)
        min_prod = min(num, min_prod * num)
        result = max(result, max_prod)
    return result

def house_robber(nums):
    """Maximum money from non-adjacent houses."""
    if not nums:
        return 0
    if len(nums) <= 2:
        return max(nums)
    dp = [0] * len(nums)
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    for i in range(2, len(nums)):
        dp[i] = max(dp[i-1], dp[i-2] + nums[i])
    return dp[-1]

def find_kth_largest(nums, k):
    """Finds kth largest element in array."""
    import heapq
    return heapq.nlargest(k, nums)[-1]

def product_except_self(nums):
    """Returns array where each element is product of all others."""
    n = len(nums)
    result = [1] * n
    prefix = 1
    for i in range(n):
        result[i] = prefix
        prefix *= nums[i]
    suffix = 1
    for i in range(n-1, -1, -1):
        result[i] *= suffix
        suffix *= nums[i]
    return result

def subarray_sum_equals_k(nums, k):
    """Counts subarrays that sum to k."""
    count = 0
    prefix_sum = 0
    prefix_counts = {0: 1}
    for num in nums:
        prefix_sum += num
        if prefix_sum - k in prefix_counts:
            count += prefix_counts[prefix_sum - k]
        prefix_counts[prefix_sum] = prefix_counts.get(prefix_sum, 0) + 1
    return count

def decode_ways(s):
    """Counts ways to decode string of digits to letters (1=A, 26=Z)."""
    if not s or s[0] == '0':
        return 0
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1
    for i in range(2, n + 1):
        if s[i-1] != '0':
            dp[i] += dp[i-1]
        two_digit = int(s[i-2:i])
        if 10 <= two_digit <= 26:
            dp[i] += dp[i-2]
    return dp[n]

# Tests
tests = [
    ("unique_paths", unique_paths(3, 7), 28),
    ("count_islands", count_islands([['1','1','0'],['1','1','0'],['0','0','1']]), 2),
    ("max_product", max_product_subarray([2,3,-2,4]), 6),
    ("house_robber", house_robber([1,2,3,1]), 4),
    ("kth_largest", find_kth_largest([3,2,1,5,6,4], 2), 5),
    ("product_except", product_except_self([1,2,3,4]), [24,12,8,6]),
    ("subarray_sum", subarray_sum_equals_k([1,1,1], 2), 2),
    ("decode_ways", decode_ways("226"), 3),
]

# Serialize/deserialize test
tree = {'val': 1, 'left': {'val': 2, 'left': None, 'right': None}, 'right': {'val': 3, 'left': None, 'right': None}}
serialized = serialize_tree(tree)
deserialized = deserialize_tree(serialized)
tests.append(("serialize_tree", deserialized['val'] == 1 and deserialized['left']['val'] == 2, True))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
