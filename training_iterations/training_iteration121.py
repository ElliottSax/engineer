# ULTRA: Advanced Matrix Algorithms

import copy

# ULTRA: Matrix Exponentiation for Linear Recurrences
def matrix_mult(A, B, mod=None):
    """Multiply two matrices, optionally with modulo."""
    n, m, p = len(A), len(A[0]), len(B[0])
    result = [[0] * p for _ in range(n)]
    for i in range(n):
        for j in range(p):
            for k in range(m):
                result[i][j] += A[i][k] * B[k][j]
                if mod:
                    result[i][j] %= mod
    return result

def matrix_pow(M, n, mod=None):
    """Compute M^n using binary exponentiation."""
    size = len(M)
    result = [[1 if i == j else 0 for j in range(size)] for i in range(size)]

    while n > 0:
        if n & 1:
            result = matrix_mult(result, M, mod)
        M = matrix_mult(M, M, mod)
        n >>= 1

    return result

def fibonacci_matrix(n, mod=10**9+7):
    """Compute n-th Fibonacci number using matrix exponentiation."""
    if n <= 0:
        return 0
    if n == 1:
        return 1

    M = [[1, 1], [1, 0]]
    result = matrix_pow(M, n - 1, mod)
    return result[0][0]

# ULTRA: Gaussian Elimination
def gaussian_elimination(matrix):
    """Solve system of linear equations using Gaussian elimination."""
    m = [row[:] for row in matrix]  # Copy
    n = len(m)
    if n == 0:
        return []

    cols = len(m[0]) - 1  # Last column is RHS

    # Forward elimination
    pivot_row = 0
    for col in range(min(n, cols)):
        # Find pivot
        max_row = pivot_row
        for i in range(pivot_row + 1, n):
            if abs(m[i][col]) > abs(m[max_row][col]):
                max_row = i

        if abs(m[max_row][col]) < 1e-10:
            continue

        m[pivot_row], m[max_row] = m[max_row], m[pivot_row]

        # Eliminate
        for i in range(pivot_row + 1, n):
            if abs(m[pivot_row][col]) > 1e-10:
                factor = m[i][col] / m[pivot_row][col]
                for j in range(col, cols + 1):
                    m[i][j] -= factor * m[pivot_row][j]

        pivot_row += 1

    # Back substitution
    solution = [0] * cols
    for i in range(min(n, cols) - 1, -1, -1):
        # Find pivot column
        pivot_col = -1
        for j in range(cols):
            if abs(m[i][j]) > 1e-10:
                pivot_col = j
                break

        if pivot_col == -1:
            continue

        solution[pivot_col] = m[i][cols]
        for j in range(pivot_col + 1, cols):
            solution[pivot_col] -= m[i][j] * solution[j]
        solution[pivot_col] /= m[i][pivot_col]

    return solution

# ULTRA: Matrix Determinant
def determinant(matrix):
    """Compute determinant using LU decomposition."""
    n = len(matrix)
    if n == 0:
        return 1
    if n == 1:
        return matrix[0][0]

    m = [row[:] for row in matrix]
    det = 1

    for col in range(n):
        # Find pivot
        max_row = col
        for i in range(col + 1, n):
            if abs(m[i][col]) > abs(m[max_row][col]):
                max_row = i

        if abs(m[max_row][col]) < 1e-10:
            return 0

        if max_row != col:
            m[col], m[max_row] = m[max_row], m[col]
            det *= -1

        det *= m[col][col]

        for i in range(col + 1, n):
            factor = m[i][col] / m[col][col]
            for j in range(col, n):
                m[i][j] -= factor * m[col][j]

    return det

# ULTRA: Matrix Rank
def matrix_rank(matrix):
    """Compute rank of matrix using Gaussian elimination."""
    if not matrix or not matrix[0]:
        return 0

    m = [row[:] for row in matrix]
    rows, cols = len(m), len(m[0])
    rank = 0

    for col in range(cols):
        # Find pivot
        pivot = -1
        for i in range(rank, rows):
            if abs(m[i][col]) > 1e-10:
                pivot = i
                break

        if pivot == -1:
            continue

        m[rank], m[pivot] = m[pivot], m[rank]

        for i in range(rank + 1, rows):
            if abs(m[rank][col]) > 1e-10:
                factor = m[i][col] / m[rank][col]
                for j in range(col, cols):
                    m[i][j] -= factor * m[rank][j]

        rank += 1

    return rank

# ULTRA: Matrix Inverse
def matrix_inverse(matrix):
    """Compute inverse using Gauss-Jordan elimination."""
    n = len(matrix)
    if n == 0:
        return []

    # Augment with identity
    aug = [row[:] + [1 if i == j else 0 for j in range(n)]
           for i, row in enumerate(matrix)]

    # Forward elimination
    for col in range(n):
        # Find pivot
        max_row = col
        for i in range(col + 1, n):
            if abs(aug[i][col]) > abs(aug[max_row][col]):
                max_row = i

        if abs(aug[max_row][col]) < 1e-10:
            return None  # Singular

        aug[col], aug[max_row] = aug[max_row], aug[col]

        # Scale pivot row
        pivot = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pivot

        # Eliminate
        for i in range(n):
            if i != col:
                factor = aug[i][col]
                for j in range(2 * n):
                    aug[i][j] -= factor * aug[col][j]

    return [row[n:] for row in aug]

# ULTRA: Linear Recurrence Solver
def solve_linear_recurrence(coeffs, initial, n, mod=10**9+7):
    """Solve a_n = c_1*a_{n-1} + c_2*a_{n-2} + ... + c_k*a_{n-k}."""
    k = len(coeffs)
    if n < k:
        return initial[n]

    # Build companion matrix
    M = [[0] * k for _ in range(k)]
    for j in range(k):
        M[0][j] = coeffs[j]
    for i in range(1, k):
        M[i][i - 1] = 1

    # Compute M^(n-k+1)
    result = matrix_pow(M, n - k + 1, mod)

    # Multiply by initial vector (reversed)
    ans = 0
    for j in range(k):
        ans += result[0][j] * initial[k - 1 - j]
        ans %= mod

    return ans

# ULTRA: Strassen's Matrix Multiplication (simplified for 2x2)
def strassen_2x2(A, B):
    """Strassen's algorithm for 2x2 matrices."""
    a, b, c, d = A[0][0], A[0][1], A[1][0], A[1][1]
    e, f, g, h = B[0][0], B[0][1], B[1][0], B[1][1]

    p1 = a * (f - h)
    p2 = (a + b) * h
    p3 = (c + d) * e
    p4 = d * (g - e)
    p5 = (a + d) * (e + h)
    p6 = (b - d) * (g + h)
    p7 = (a - c) * (e + f)

    return [
        [p5 + p4 - p2 + p6, p1 + p2],
        [p3 + p4, p1 + p5 - p3 - p7]
    ]

# Tests
tests = []

# Matrix multiplication
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
C = matrix_mult(A, B)
tests.append(("mat_mult", C, [[19, 22], [43, 50]]))

# Fibonacci
tests.append(("fib_10", fibonacci_matrix(10), 55))
tests.append(("fib_50", fibonacci_matrix(50, 10**9+7), 12586269025 % (10**9+7)))

# Gaussian elimination
# 2x + y = 5, x + 3y = 10 => x=1, y=3
aug = [[2, 1, 5], [1, 3, 10]]
sol = gaussian_elimination(aug)
tests.append(("gauss_x", round(sol[0], 2), 1.0))
tests.append(("gauss_y", round(sol[1], 2), 3.0))

# Determinant
tests.append(("det_2x2", determinant([[1, 2], [3, 4]]), -2))
tests.append(("det_3x3", round(determinant([[1, 2, 3], [4, 5, 6], [7, 8, 10]]), 2), -3.0))

# Matrix rank
tests.append(("rank_full", matrix_rank([[1, 2], [3, 4]]), 2))
tests.append(("rank_dep", matrix_rank([[1, 2], [2, 4]]), 1))

# Matrix inverse
inv = matrix_inverse([[4, 7], [2, 6]])
tests.append(("inv_valid", inv is not None, True))
if inv:
    tests.append(("inv_00", round(inv[0][0], 1), 0.6))

# Linear recurrence (Fibonacci: a_n = a_{n-1} + a_{n-2})
tests.append(("recur_fib", solve_linear_recurrence([1, 1], [0, 1], 10), 55))

# Strassen
S = strassen_2x2(A, B)
tests.append(("strassen", S, [[19, 22], [43, 50]]))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
