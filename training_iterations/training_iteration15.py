def minimum_path_sum(grid):
    """Minimum path sum from top-left to bottom-right."""
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
    return dp[m-1][n-1]

def unique_paths(m, n):
    """Number of unique paths from top-left to bottom-right."""
    dp = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[m-1][n-1]

def unique_paths_with_obstacles(grid):
    """Unique paths with obstacles (1 = obstacle)."""
    m, n = len(grid), len(grid[0])
    if grid[0][0] == 1:
        return 0
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = 1
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] if grid[i][0] == 0 else 0
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] if grid[0][j] == 0 else 0
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = (dp[i-1][j] + dp[i][j-1]) if grid[i][j] == 0 else 0
    return dp[m-1][n-1]

def dungeon_game(dungeon):
    """Minimum initial HP to reach princess."""
    m, n = len(dungeon), len(dungeon[0])
    dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
    dp[m][n-1] = dp[m-1][n] = 1
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            dp[i][j] = max(1, min(dp[i+1][j], dp[i][j+1]) - dungeon[i][j])
    return dp[0][0]

def triangle_min_path(triangle):
    """Minimum path sum in triangle (top to bottom)."""
    n = len(triangle)
    dp = triangle[-1][:]
    for i in range(n - 2, -1, -1):
        for j in range(len(triangle[i])):
            dp[j] = min(dp[j], dp[j+1]) + triangle[i][j]
    return dp[0]

def cherry_pickup(grid):
    """Maximum cherries collected on round trip."""
    n = len(grid)
    memo = {}

    def dp(r1, c1, r2):
        c2 = r1 + c1 - r2
        if r1 >= n or c1 >= n or r2 >= n or c2 >= n:
            return float('-inf')
        if grid[r1][c1] == -1 or grid[r2][c2] == -1:
            return float('-inf')
        if r1 == n - 1 and c1 == n - 1:
            return grid[r1][c1]
        if (r1, c1, r2) in memo:
            return memo[(r1, c1, r2)]

        cherries = grid[r1][c1]
        if r1 != r2 or c1 != c2:
            cherries += grid[r2][c2]

        cherries += max(
            dp(r1 + 1, c1, r2 + 1),
            dp(r1 + 1, c1, r2),
            dp(r1, c1 + 1, r2 + 1),
            dp(r1, c1 + 1, r2)
        )
        memo[(r1, c1, r2)] = cherries
        return cherries

    return max(0, dp(0, 0, 0))

def maximal_square(matrix):
    """Side length of largest square of 1s."""
    if not matrix:
        return 0
    m, n = len(matrix), len(matrix[0])
    dp = [[0] * n for _ in range(m)]
    max_side = 0
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == '1':
                if i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                max_side = max(max_side, dp[i][j])
    return max_side * max_side

def count_bits(n):
    """Counts bits for all numbers from 0 to n."""
    result = [0] * (n + 1)
    for i in range(1, n + 1):
        result[i] = result[i >> 1] + (i & 1)
    return result

def hamming_distance(x, y):
    """Hamming distance between two integers."""
    xor = x ^ y
    count = 0
    while xor:
        count += xor & 1
        xor >>= 1
    return count

def reverse_bits(n):
    """Reverses bits of 32-bit unsigned integer."""
    result = 0
    for _ in range(32):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result

def add_binary(a, b):
    """Adds two binary strings."""
    result = []
    carry = 0
    i, j = len(a) - 1, len(b) - 1
    while i >= 0 or j >= 0 or carry:
        total = carry
        if i >= 0:
            total += int(a[i])
            i -= 1
        if j >= 0:
            total += int(b[j])
            j -= 1
        result.append(str(total % 2))
        carry = total // 2
    return ''.join(reversed(result))

def divide_two_integers(dividend, divisor):
    """Divides without multiplication, division, or mod."""
    if dividend == -2**31 and divisor == -1:
        return 2**31 - 1
    negative = (dividend < 0) != (divisor < 0)
    dividend, divisor = abs(dividend), abs(divisor)
    result = 0
    while dividend >= divisor:
        temp, multiple = divisor, 1
        while dividend >= temp << 1:
            temp <<= 1
            multiple <<= 1
        dividend -= temp
        result += multiple
    return -result if negative else result

# Tests
tests = [
    ("min_path_sum", minimum_path_sum([[1,3,1],[1,5,1],[4,2,1]]), 7),
    ("unique_paths", unique_paths(3, 7), 28),
    ("unique_paths_obs", unique_paths_with_obstacles([[0,0,0],[0,1,0],[0,0,0]]), 2),
    ("dungeon", dungeon_game([[-2,-3,3],[-5,-10,1],[10,30,-5]]), 7),
    ("triangle", triangle_min_path([[2],[3,4],[6,5,7],[4,1,8,3]]), 11),
    ("cherry", cherry_pickup([[0,1,-1],[1,0,-1],[1,1,1]]), 5),
    ("maximal_square", maximal_square([['1','0','1','0','0'],['1','0','1','1','1'],['1','1','1','1','1'],['1','0','0','1','0']]), 4),
    ("count_bits", count_bits(5), [0,1,1,2,1,2]),
    ("hamming", hamming_distance(1, 4), 2),
    ("reverse_bits", reverse_bits(43261596), 964176192),
    ("add_binary", add_binary("11", "1"), "100"),
    ("divide", divide_two_integers(10, 3), 3),
    ("divide_neg", divide_two_integers(-7, 2), -3),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
