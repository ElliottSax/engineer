# EXTREME: Game Theory & Matrix Operations

import numpy as np
from functools import lru_cache

# HARD: Nim Game
def nim_winner(piles):
    """Determine if first player wins in Nim (XOR all piles)."""
    xor = 0
    for pile in piles:
        xor ^= pile
    return xor != 0

# HARD: Sprague-Grundy for general games
def sprague_grundy(n, moves):
    """Compute Grundy number for game with given moves."""
    grundy = [0] * (n + 1)
    for i in range(1, n + 1):
        reachable = set()
        for m in moves:
            if i >= m:
                reachable.add(grundy[i - m])
        mex = 0
        while mex in reachable:
            mex += 1
        grundy[i] = mex
    return grundy[n]

# HARD: Stone Game DP
def stone_game(piles):
    """Maximum score difference in stone game."""
    n = len(piles)
    dp = [[0] * n for _ in range(n)]

    for i in range(n):
        dp[i][i] = piles[i]

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = max(piles[i] - dp[i+1][j], piles[j] - dp[i][j-1])

    return dp[0][n-1]

# HARD: Predict the Winner
def predict_winner(nums):
    """Check if first player can win."""
    return stone_game(nums) >= 0

# HARD: Matrix Exponentiation for Fibonacci
def matrix_mult(A, B, mod=10**9+7):
    """Multiply two 2x2 matrices."""
    return [
        [(A[0][0] * B[0][0] + A[0][1] * B[1][0]) % mod,
         (A[0][0] * B[0][1] + A[0][1] * B[1][1]) % mod],
        [(A[1][0] * B[0][0] + A[1][1] * B[1][0]) % mod,
         (A[1][0] * B[0][1] + A[1][1] * B[1][1]) % mod]
    ]

def matrix_pow(M, n, mod=10**9+7):
    """Matrix exponentiation."""
    result = [[1, 0], [0, 1]]  # Identity
    while n > 0:
        if n % 2 == 1:
            result = matrix_mult(result, M, mod)
        M = matrix_mult(M, M, mod)
        n //= 2
    return result

def fib_matrix(n, mod=10**9+7):
    """Compute nth Fibonacci using matrix exponentiation."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    M = [[1, 1], [1, 0]]
    result = matrix_pow(M, n - 1, mod)
    return result[0][0]

# HARD: Linear Recurrence using Matrix Exponentiation
def linear_recurrence(coeffs, initial, n, mod=10**9+7):
    """Solve linear recurrence a[n] = c[0]*a[n-1] + c[1]*a[n-2] + ..."""
    k = len(coeffs)
    if n < k:
        return initial[n]

    # Build companion matrix
    M = [[0] * k for _ in range(k)]
    for i in range(k):
        M[0][i] = coeffs[i]
    for i in range(1, k):
        M[i][i-1] = 1

    # Matrix exponentiation
    def mult(A, B):
        r = [[0] * k for _ in range(k)]
        for i in range(k):
            for j in range(k):
                for l in range(k):
                    r[i][j] = (r[i][j] + A[i][l] * B[l][j]) % mod
        return r

    result = [[1 if i == j else 0 for j in range(k)] for i in range(k)]
    power = n - k + 1
    while power > 0:
        if power % 2 == 1:
            result = mult(result, M)
        M = mult(M, M)
        power //= 2

    ans = 0
    for i in range(k):
        ans = (ans + result[0][i] * initial[k - 1 - i]) % mod
    return ans

# HARD: Gaussian Elimination
def gaussian_elimination(matrix):
    """Solve system of linear equations."""
    m = len(matrix)
    n = len(matrix[0]) - 1

    # Forward elimination
    for col in range(min(m, n)):
        # Find pivot
        max_row = col
        for row in range(col + 1, m):
            if abs(matrix[row][col]) > abs(matrix[max_row][col]):
                max_row = row
        matrix[col], matrix[max_row] = matrix[max_row], matrix[col]

        if abs(matrix[col][col]) < 1e-10:
            continue

        # Eliminate column
        for row in range(col + 1, m):
            factor = matrix[row][col] / matrix[col][col]
            for j in range(col, n + 1):
                matrix[row][j] -= factor * matrix[col][j]

    # Back substitution
    solution = [0] * n
    for i in range(min(m, n) - 1, -1, -1):
        if abs(matrix[i][i]) < 1e-10:
            continue
        solution[i] = matrix[i][n]
        for j in range(i + 1, n):
            solution[i] -= matrix[i][j] * solution[j]
        solution[i] /= matrix[i][i]

    return solution

# HARD: Determinant using LU Decomposition
def determinant(matrix):
    """Calculate determinant of square matrix."""
    n = len(matrix)
    mat = [row[:] for row in matrix]  # Copy
    det = 1

    for col in range(n):
        # Find pivot
        max_row = col
        for row in range(col + 1, n):
            if abs(mat[row][col]) > abs(mat[max_row][col]):
                max_row = row
        if max_row != col:
            mat[col], mat[max_row] = mat[max_row], mat[col]
            det *= -1

        if abs(mat[col][col]) < 1e-10:
            return 0

        det *= mat[col][col]

        for row in range(col + 1, n):
            factor = mat[row][col] / mat[col][col]
            for j in range(col, n):
                mat[row][j] -= factor * mat[col][j]

    return det

# Tests
tests = []

# Nim
tests.append(("nim_win", nim_winner([3, 4, 5]), True))
tests.append(("nim_lose", nim_winner([1, 2, 3]), False))

# Sprague-Grundy (take 1 or 2 stones)
tests.append(("sg", sprague_grundy(10, [1, 2]), 1))  # 10 % 3 = 1

# Stone Game
tests.append(("stone", stone_game([5, 3, 4, 5]), 2))

# Predict Winner
tests.append(("predict", predict_winner([1, 5, 2]), False))
tests.append(("predict2", predict_winner([1, 5, 233, 7]), True))

# Fibonacci
tests.append(("fib_10", fib_matrix(10), 55))
tests.append(("fib_50", fib_matrix(50), 12586269025 % (10**9+7)))

# Linear Recurrence (Tribonacci: a[n] = a[n-1] + a[n-2] + a[n-3])
tests.append(("tribonacci", linear_recurrence([1, 1, 1], [0, 0, 1], 10), 149))

# Gaussian Elimination
solution = gaussian_elimination([
    [2, 1, -1, 8],
    [-3, -1, 2, -11],
    [-2, 1, 2, -3]
])
tests.append(("gauss", [round(x) for x in solution], [2, 3, -1]))

# Determinant
tests.append(("det", round(determinant([[1, 2], [3, 4]])), -2))
tests.append(("det_3x3", round(determinant([[6, 1, 1], [4, -2, 5], [2, 8, 7]])), -306))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
