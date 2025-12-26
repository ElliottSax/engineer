# ULTRA: Advanced Combinatorics and Generating Functions

from functools import lru_cache
from math import factorial, comb

# ULTRA: Burnside's Lemma for Counting Distinct Colorings
def burnside_necklace(n, k):
    """Count distinct necklaces of n beads with k colors."""
    from math import gcd

    total = 0
    for i in range(n):
        # Rotation by i positions
        cycle_len = n // gcd(n, i) if i > 0 else 1
        if i == 0:
            total += k ** n
        else:
            total += k ** gcd(n, i)

    return total // n

# ULTRA: Polya Enumeration for Cube Colorings
def polya_cube_faces(k):
    """Count distinct colorings of cube faces with k colors."""
    # Cube has 24 rotations
    # Identity: k^6
    # 6 face rotations (90, 180, 270 degrees around axis through face centers): 3 axes
    #   90/270: k^3 (2 each)
    #   180: k^4 (1 each)
    # 8 vertex rotations (120, 240 degrees): 4 axes
    #   k^2 (8 total)
    # 6 edge rotations (180 degrees): 6 axes
    #   k^3 (6 total)

    total = k**6  # Identity
    total += 2 * 3 * k**3  # 90 and 270 degree face rotations (6)
    total += 3 * k**4      # 180 degree face rotations (3)
    total += 8 * k**2      # Vertex rotations (8)
    total += 6 * k**3      # Edge rotations (6)

    return total // 24

# ULTRA: Derangements
def derangements(n):
    """Count permutations with no fixed points."""
    if n == 0:
        return 1
    if n == 1:
        return 0

    # D(n) = (n-1) * (D(n-1) + D(n-2))
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 0
    for i in range(2, n + 1):
        dp[i] = (i - 1) * (dp[i - 1] + dp[i - 2])

    return dp[n]

# ULTRA: Catalan Number Variations
def catalan_applications():
    """Various Catalan number applications."""

    def catalan(n):
        if n <= 1:
            return 1
        dp = [0] * (n + 1)
        dp[0] = dp[1] = 1
        for i in range(2, n + 1):
            for j in range(i):
                dp[i] += dp[j] * dp[i - 1 - j]
        return dp[n]

    # Parenthesizations
    def count_parenthesizations(n):
        return catalan(n)

    # Binary trees
    def count_binary_trees(n):
        return catalan(n)

    # Paths above diagonal
    def count_paths_above_diagonal(n):
        return catalan(n)

    # Non-crossing partitions
    def count_non_crossing(n):
        return catalan(n)

    return catalan(5), catalan(6), catalan(7)

# ULTRA: Partition Numbers with Restrictions
def partition_distinct(n):
    """Count partitions with distinct parts."""
    dp = [0] * (n + 1)
    dp[0] = 1

    for i in range(1, n + 1):
        for j in range(n, i - 1, -1):
            dp[j] += dp[j - i]

    return dp[n]

def partition_odd(n):
    """Count partitions with all odd parts."""
    dp = [0] * (n + 1)
    dp[0] = 1

    for i in range(1, n + 1, 2):  # Only odd parts
        for j in range(i, n + 1):
            dp[j] += dp[j - i]

    return dp[n]

# ULTRA: Generating Functions - Fibonacci via Matrix
def fibonacci_closed(n, mod=10**9 + 7):
    """Fibonacci using matrix exponentiation."""
    if n <= 0:
        return 0
    if n == 1:
        return 1

    def matrix_mult(A, B, mod):
        return [
            [(A[0][0] * B[0][0] + A[0][1] * B[1][0]) % mod,
             (A[0][0] * B[0][1] + A[0][1] * B[1][1]) % mod],
            [(A[1][0] * B[0][0] + A[1][1] * B[1][0]) % mod,
             (A[1][0] * B[0][1] + A[1][1] * B[1][1]) % mod]
        ]

    def matrix_pow(M, n, mod):
        result = [[1, 0], [0, 1]]
        while n > 0:
            if n % 2:
                result = matrix_mult(result, M, mod)
            M = matrix_mult(M, M, mod)
            n //= 2
        return result

    M = [[1, 1], [1, 0]]
    result = matrix_pow(M, n - 1, mod)
    return result[0][0]

# ULTRA: Inclusion-Exclusion Principle
def count_coprime(n, primes):
    """Count integers from 1 to n coprime to all given primes."""
    from itertools import combinations

    total = n
    for r in range(1, len(primes) + 1):
        for combo in combinations(primes, r):
            product = 1
            for p in combo:
                product *= p
            if (-1) ** r == -1:
                total -= n // product
            else:
                total += n // product

    return total

def euler_totient_formula(n, prime_factors):
    """Euler's totient using inclusion-exclusion."""
    result = n
    for p in prime_factors:
        result = result * (p - 1) // p
    return result

# ULTRA: Stirling Numbers
def stirling_first(n, k):
    """Stirling numbers of the first kind (unsigned)."""
    # Count permutations of n elements with exactly k cycles
    dp = [[0] * (k + 1) for _ in range(n + 1)]
    dp[0][0] = 1

    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            dp[i][j] = (i - 1) * dp[i - 1][j] + dp[i - 1][j - 1]

    return dp[n][k]

def stirling_second(n, k):
    """Stirling numbers of the second kind."""
    # Count ways to partition n elements into k non-empty subsets
    dp = [[0] * (k + 1) for _ in range(n + 1)]
    dp[0][0] = 1

    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            dp[i][j] = j * dp[i - 1][j] + dp[i - 1][j - 1]

    return dp[n][k]

# ULTRA: Lucas' Theorem Implementation
def lucas(n, r, p):
    """Compute nCr mod p using Lucas' theorem."""
    def nCr_small(n, r, p):
        if r > n:
            return 0
        if r == 0:
            return 1
        num = den = 1
        for i in range(r):
            num = num * (n - i) % p
            den = den * (i + 1) % p
        return num * pow(den, p - 2, p) % p

    result = 1
    while n > 0 or r > 0:
        ni, ri = n % p, r % p
        if ri > ni:
            return 0
        result = result * nCr_small(ni, ri, p) % p
        n //= p
        r //= p

    return result

# Tests
tests = []

# Burnside necklace
tests.append(("necklace_4_2", burnside_necklace(4, 2), 6))  # 2 same, 2 alternating, etc.
tests.append(("necklace_6_3", burnside_necklace(6, 3), 130))

# Polya cube
tests.append(("cube_2colors", polya_cube_faces(2), 10))
tests.append(("cube_3colors", polya_cube_faces(3), 57))

# Derangements
tests.append(("derangements_4", derangements(4), 9))
tests.append(("derangements_5", derangements(5), 44))

# Catalan
c5, c6, c7 = catalan_applications()
tests.append(("catalan_5", c5, 42))
tests.append(("catalan_6", c6, 132))

# Partition distinct
tests.append(("partition_distinct_5", partition_distinct(5), 3))  # 5, 4+1, 3+2
tests.append(("partition_odd_5", partition_odd(5), 3))  # 5, 3+1+1, 1+1+1+1+1

# Fibonacci
tests.append(("fib_10", fibonacci_closed(10), 55))
tests.append(("fib_50", fibonacci_closed(50) % (10**9+7), 12586269025 % (10**9+7)))

# Inclusion-exclusion
tests.append(("coprime", count_coprime(30, [2, 3, 5]), 8))  # 1,7,11,13,17,19,23,29

# Stirling
tests.append(("stirling1_4_2", stirling_first(4, 2), 11))
tests.append(("stirling2_4_2", stirling_second(4, 2), 7))

# Lucas
tests.append(("lucas", lucas(10, 3, 13), 120 % 13))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
