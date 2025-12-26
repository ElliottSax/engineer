# EXTREME: Number Theory & Combinatorics

from functools import lru_cache
from math import gcd, isqrt

# HARD: Extended Euclidean Algorithm
def extended_gcd(a, b):
    """Extended GCD returning (gcd, x, y) where ax + by = gcd."""
    if b == 0:
        return a, 1, 0
    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y

# HARD: Modular Inverse
def mod_inverse(a, m):
    """Modular multiplicative inverse using extended GCD."""
    g, x, _ = extended_gcd(a, m)
    if g != 1:
        return None  # No inverse
    return x % m

# HARD: Chinese Remainder Theorem
def chinese_remainder(remainders, moduli):
    """Find x such that x ≡ r_i (mod m_i) for all i."""
    if len(remainders) != len(moduli):
        return None

    M = 1
    for m in moduli:
        M *= m

    result = 0
    for r, m in zip(remainders, moduli):
        Mi = M // m
        yi = mod_inverse(Mi, m)
        if yi is None:
            return None
        result += r * Mi * yi

    return result % M

# HARD: Miller-Rabin Primality Test
def is_prime_miller_rabin(n, k=10):
    """Probabilistic primality test."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False

    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Witnesses to test
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    for a in witnesses:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

# HARD: Pollard's Rho Factorization
def pollard_rho(n):
    """Find a non-trivial factor of n."""
    if n % 2 == 0:
        return 2
    x = 2
    y = 2
    d = 1
    c = 1

    def f(x):
        return (x * x + c) % n

    while d == 1:
        x = f(x)
        y = f(f(y))
        d = gcd(abs(x - y), n)

    return d if d != n else None

def prime_factors(n):
    """Complete prime factorization."""
    factors = []
    # Small primes
    for p in [2, 3, 5, 7, 11, 13]:
        while n % p == 0:
            factors.append(p)
            n //= p

    if n == 1:
        return factors

    if is_prime_miller_rabin(n):
        factors.append(n)
        return factors

    # Use Pollard's rho for larger factors
    while n > 1:
        if is_prime_miller_rabin(n):
            factors.append(n)
            break
        d = pollard_rho(n)
        if d is None:
            factors.append(n)
            break
        while n % d == 0:
            factors.append(d)
            n //= d

    return sorted(factors)

# HARD: Euler's Totient Function
def euler_totient(n):
    """Count integers 1..n coprime to n."""
    result = n
    p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1
    if n > 1:
        result -= result // n
    return result

# HARD: Mobius Function
def mobius(n):
    """Mobius function: 0 if n has squared prime factor, (-1)^k otherwise."""
    if n == 1:
        return 1
    p = 2
    factors = 0
    while p * p <= n:
        if n % p == 0:
            n //= p
            factors += 1
            if n % p == 0:
                return 0  # Squared factor
        p += 1
    if n > 1:
        factors += 1
    return -1 if factors % 2 else 1

# HARD: Lucas' Theorem for nCr mod p
def lucas(n, r, p):
    """Compute nCr mod p using Lucas' theorem."""
    def nCr_small(n, r, p):
        if r > n:
            return 0
        if r == 0 or r == n:
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

# HARD: Catalan Numbers
def catalan(n):
    """nth Catalan number."""
    if n <= 1:
        return 1
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1
    for i in range(2, n + 1):
        for j in range(i):
            dp[i] += dp[j] * dp[i - 1 - j]
    return dp[n]

# HARD: Stirling Numbers of Second Kind
def stirling2(n, k):
    """Count ways to partition n elements into k non-empty subsets."""
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    dp = [[0] * (k + 1) for _ in range(n + 1)]
    dp[0][0] = 1
    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            dp[i][j] = j * dp[i-1][j] + dp[i-1][j-1]
    return dp[n][k]

# HARD: Bell Numbers
def bell(n):
    """nth Bell number - total partitions of n elements."""
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    dp[0][0] = 1
    for i in range(1, n + 1):
        dp[i][0] = dp[i-1][i-1]
        for j in range(1, i + 1):
            dp[i][j] = dp[i-1][j-1] + dp[i][j-1]
    return dp[n][0]

# HARD: Partition Function
def partition_count(n):
    """Count integer partitions of n."""
    dp = [0] * (n + 1)
    dp[0] = 1
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            dp[j] += dp[j - i]
    return dp[n]

# Tests
tests = [
    # Extended GCD
    ("ext_gcd", extended_gcd(35, 15), (5, 1, -2)),
    ("ext_gcd2", extended_gcd(240, 46), (2, -9, 47)),

    # Modular inverse
    ("mod_inv", mod_inverse(3, 11), 4),
    ("mod_inv2", mod_inverse(17, 43), 38),

    # Chinese Remainder Theorem
    ("crt", chinese_remainder([2, 3, 2], [3, 5, 7]), 23),
    ("crt2", chinese_remainder([1, 4, 6], [3, 5, 7]), 34),

    # Miller-Rabin
    ("prime1", is_prime_miller_rabin(104729), True),
    ("prime2", is_prime_miller_rabin(104730), False),
    ("prime3", is_prime_miller_rabin(999999937), True),

    # Prime factorization
    ("factors", prime_factors(60), [2, 2, 3, 5]),
    ("factors2", prime_factors(84), [2, 2, 3, 7]),

    # Euler's totient
    ("totient", euler_totient(36), 12),
    ("totient2", euler_totient(97), 96),

    # Mobius
    ("mobius1", mobius(1), 1),
    ("mobius2", mobius(6), 1),
    ("mobius3", mobius(12), 0),
    ("mobius4", mobius(30), -1),

    # Lucas
    ("lucas", lucas(1000, 500, 13), 2),

    # Catalan
    ("catalan", catalan(5), 42),
    ("catalan2", catalan(10), 16796),

    # Stirling
    ("stirling", stirling2(5, 3), 25),
    ("stirling2", stirling2(4, 2), 7),

    # Bell
    ("bell", bell(5), 52),
    ("bell2", bell(6), 203),

    # Partition
    ("partition", partition_count(5), 7),
    ("partition2", partition_count(10), 42),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
