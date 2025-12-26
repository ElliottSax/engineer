def gcd(a, b):
    """Greatest common divisor using Euclidean algorithm."""
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    """Least common multiple."""
    return a * b // gcd(a, b)

def extended_gcd(a, b):
    """Extended Euclidean algorithm: returns (gcd, x, y) where ax + by = gcd."""
    if b == 0:
        return a, 1, 0
    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y

def mod_inverse(a, m):
    """Modular multiplicative inverse using extended GCD."""
    g, x, _ = extended_gcd(a, m)
    if g != 1:
        return None
    return x % m

def fast_power(base, exp, mod=None):
    """Fast exponentiation."""
    result = 1
    while exp > 0:
        if exp & 1:
            result = result * base if mod is None else (result * base) % mod
        base = base * base if mod is None else (base * base) % mod
        exp >>= 1
    return result

def sieve_of_eratosthenes(n):
    """Generate all primes up to n."""
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(n + 1) if is_prime[i]]

def prime_factorization(n):
    """Return prime factors with their powers."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def count_divisors(n):
    """Count number of divisors."""
    factors = prime_factorization(n)
    count = 1
    for power in factors.values():
        count *= (power + 1)
    return count

def euler_totient(n):
    """Euler's totient function - count of coprime numbers <= n."""
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

def chinese_remainder(remainders, moduli):
    """Chinese Remainder Theorem solver."""
    from functools import reduce
    M = reduce(lambda a, b: a * b, moduli)
    result = 0
    for r, m in zip(remainders, moduli):
        Mi = M // m
        yi = mod_inverse(Mi, m)
        result += r * Mi * yi
    return result % M

def count_trailing_zeros_factorial(n):
    """Count trailing zeros in n!"""
    count = 0
    power = 5
    while power <= n:
        count += n // power
        power *= 5
    return count

def is_perfect_square(n):
    """Check if n is a perfect square."""
    if n < 0:
        return False
    root = int(n ** 0.5)
    return root * root == n

def integer_sqrt(n):
    """Integer square root using Newton's method."""
    if n < 0:
        raise ValueError("Square root of negative number")
    if n == 0:
        return 0
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x

def nth_fibonacci_matrix(n):
    """Nth Fibonacci using matrix exponentiation O(log n)."""
    if n <= 1:
        return n

    def matrix_mult(A, B):
        return [
            [A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
            [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]]
        ]

    def matrix_power(M, p):
        result = [[1, 0], [0, 1]]
        while p:
            if p & 1:
                result = matrix_mult(result, M)
            M = matrix_mult(M, M)
            p >>= 1
        return result

    M = [[1, 1], [1, 0]]
    return matrix_power(M, n)[0][1]

def catalan_number(n):
    """Nth Catalan number."""
    if n <= 1:
        return 1
    from math import factorial
    return factorial(2*n) // (factorial(n+1) * factorial(n))

def pascal_triangle(n):
    """Generate n rows of Pascal's triangle."""
    result = [[1]]
    for i in range(1, n):
        row = [1]
        for j in range(1, i):
            row.append(result[i-1][j-1] + result[i-1][j])
        row.append(1)
        result.append(row)
    return result

def nCr_mod(n, r, p):
    """Binomial coefficient mod p using Lucas theorem."""
    if r > n:
        return 0
    if r == 0 or r == n:
        return 1

    # Precompute factorials
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = (fact[i-1] * i) % p

    return (fact[n] * mod_inverse(fact[r], p) * mod_inverse(fact[n-r], p)) % p

# Tests
tests = [
    ("gcd", gcd(48, 18), 6),
    ("gcd_coprime", gcd(17, 13), 1),
    ("lcm", lcm(12, 18), 36),
    ("ext_gcd", extended_gcd(35, 15)[0], 5),
    ("mod_inv", mod_inverse(3, 11), 4),
    ("fast_pow", fast_power(2, 10), 1024),
    ("fast_pow_mod", fast_power(2, 10, 1000), 24),
    ("sieve", sieve_of_eratosthenes(20), [2, 3, 5, 7, 11, 13, 17, 19]),
    ("prime_fact", prime_factorization(60), {2: 2, 3: 1, 5: 1}),
    ("divisors", count_divisors(12), 6),
    ("totient", euler_totient(10), 4),
    ("crt", chinese_remainder([2, 3, 2], [3, 5, 7]), 23),
    ("trailing_zeros", count_trailing_zeros_factorial(25), 6),
    ("perfect_sq_true", is_perfect_square(49), True),
    ("perfect_sq_false", is_perfect_square(50), False),
    ("int_sqrt", integer_sqrt(17), 4),
    ("fib_matrix", nth_fibonacci_matrix(10), 55),
    ("catalan", catalan_number(5), 42),
    ("pascal", pascal_triangle(5)[-1], [1, 4, 6, 4, 1]),
    ("ncr_mod", nCr_mod(10, 3, 1000000007), 120),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
