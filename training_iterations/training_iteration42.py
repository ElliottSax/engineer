def max_profit_with_fee(prices, fee):
    """Maximum profit with transaction fee."""
    cash = 0
    hold = -prices[0]
    for i in range(1, len(prices)):
        cash = max(cash, hold + prices[i] - fee)
        hold = max(hold, cash - prices[i])
    return cash

def max_profit_k_transactions(k, prices):
    """Maximum profit with at most k transactions."""
    if not prices:
        return 0
    n = len(prices)
    if k >= n // 2:
        return sum(max(0, prices[i] - prices[i-1]) for i in range(1, n))

    dp = [[0] * n for _ in range(k + 1)]
    for i in range(1, k + 1):
        max_diff = -prices[0]
        for j in range(1, n):
            dp[i][j] = max(dp[i][j-1], prices[j] + max_diff)
            max_diff = max(max_diff, dp[i-1][j] - prices[j])
    return dp[k][n-1]

def max_profit_with_freeze(prices):
    """Maximum profit with cooldown (wait 1 day after selling)."""
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

def minimum_falling_path_sum(matrix):
    """Minimum sum of falling path through matrix."""
    n = len(matrix)
    dp = matrix[0][:]
    for i in range(1, n):
        new_dp = [0] * n
        for j in range(n):
            new_dp[j] = matrix[i][j] + min(
                dp[j],
                dp[j-1] if j > 0 else float('inf'),
                dp[j+1] if j < n-1 else float('inf')
            )
        dp = new_dp
    return min(dp)

def minimum_falling_path_sum_ii(grid):
    """Minimum falling path where can't use same column twice."""
    n = len(grid)
    if n == 1:
        return grid[0][0]

    def get_two_mins(row):
        min1 = min2 = float('inf')
        min1_idx = -1
        for j, val in enumerate(row):
            if val < min1:
                min2 = min1
                min1 = val
                min1_idx = j
            elif val < min2:
                min2 = val
        return min1, min1_idx, min2

    prev = grid[0][:]
    for i in range(1, n):
        min1, min1_idx, min2 = get_two_mins(prev)
        curr = [0] * n
        for j in range(n):
            curr[j] = grid[i][j] + (min2 if j == min1_idx else min1)
        prev = curr
    return min(prev)

def knight_dialer(n):
    """Number of distinct phone numbers of length n."""
    MOD = 10**9 + 7
    moves = {
        0: [4, 6], 1: [6, 8], 2: [7, 9], 3: [4, 8],
        4: [0, 3, 9], 5: [], 6: [0, 1, 7], 7: [2, 6],
        8: [1, 3], 9: [2, 4]
    }
    dp = [1] * 10
    for _ in range(n - 1):
        new_dp = [0] * 10
        for digit in range(10):
            for next_digit in moves[digit]:
                new_dp[next_digit] = (new_dp[next_digit] + dp[digit]) % MOD
        dp = new_dp
    return sum(dp) % MOD

def knight_probability(n, k, row, col):
    """Probability knight stays on board after k moves."""
    moves = [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]
    dp = [[0] * n for _ in range(n)]
    dp[row][col] = 1

    for _ in range(k):
        new_dp = [[0] * n for _ in range(n)]
        for r in range(n):
            for c in range(n):
                if dp[r][c] > 0:
                    for dr, dc in moves:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < n and 0 <= nc < n:
                            new_dp[nr][nc] += dp[r][c] / 8
        dp = new_dp
    return sum(sum(row) for row in dp)

def out_of_boundary_paths(m, n, max_move, start_row, start_col):
    """Number of paths to move ball out of grid."""
    MOD = 10**9 + 7
    dp = [[0] * n for _ in range(m)]
    dp[start_row][start_col] = 1
    result = 0

    for _ in range(max_move):
        new_dp = [[0] * n for _ in range(m)]
        for r in range(m):
            for c in range(n):
                if dp[r][c] > 0:
                    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < m and 0 <= nc < n:
                            new_dp[nr][nc] = (new_dp[nr][nc] + dp[r][c]) % MOD
                        else:
                            result = (result + dp[r][c]) % MOD
        dp = new_dp
    return result

def unique_paths_iii(grid):
    """Count paths visiting every non-obstacle cell exactly once."""
    m, n = len(grid), len(grid[0])
    start = end = None
    empty = 1  # Count start as well

    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                start = (i, j)
            elif grid[i][j] == 2:
                end = (i, j)
            elif grid[i][j] == 0:
                empty += 1

    def dfs(r, c, remaining):
        if (r, c) == end:
            return 1 if remaining == 0 else 0
        count = 0
        grid[r][c] = -1
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] >= 0:
                count += dfs(nr, nc, remaining - 1)
        grid[r][c] = 0
        return count

    return dfs(start[0], start[1], empty)

def domino_tromino_tiling(n):
    """Ways to tile 2xN board with dominoes and trominoes."""
    MOD = 10**9 + 7
    if n <= 2:
        return n
    dp = [0] * (n + 1)
    dp[1], dp[2], dp[3] = 1, 2, 5
    for i in range(4, n + 1):
        dp[i] = (2 * dp[i-1] + dp[i-3]) % MOD
    return dp[n]

# Tests
tests = [
    ("profit_fee", max_profit_with_fee([1,3,2,8,4,9], 2), 8),
    ("profit_k", max_profit_k_transactions(2, [2,4,1]), 2),
    ("profit_k_2", max_profit_k_transactions(2, [3,2,6,5,0,3]), 7),
    ("profit_freeze", max_profit_with_freeze([1,2,3,0,2]), 3),
    ("falling_path", minimum_falling_path_sum([[2,1,3],[6,5,4],[7,8,9]]), 13),
    ("falling_path_ii", minimum_falling_path_sum_ii([[1,2,3],[4,5,6],[7,8,9]]), 13),
    ("knight_dialer", knight_dialer(1), 10),
    ("knight_dialer_2", knight_dialer(2), 20),
    ("knight_prob", round(knight_probability(3, 2, 0, 0), 5), 0.0625),
    ("out_boundary", out_of_boundary_paths(2, 2, 2, 0, 0), 6),
    ("unique_paths_iii", unique_paths_iii([[1,0,0,0],[0,0,0,0],[0,0,2,-1]]), 2),
    ("domino", domino_tromino_tiling(3), 5),
    ("domino_2", domino_tromino_tiling(4), 11),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
