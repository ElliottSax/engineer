# ULTRA: Advanced Number Theory II

from math import gcd, isqrt
from functools import lru_cache

# ULTRA: Extended Euclidean Algorithm
def extended_gcd(a, b):
    """Return (gcd, x, y) such that a*x + b*y = gcd(a, b)."""
    if b == 0:
        return a, 1, 0
    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y

# ULTRA: Modular Inverse
def mod_inverse(a, m):
    """Find modular inverse of a mod m using extended GCD."""
    g, x, _ = extended_gcd(a % m, m)
    if g != 1:
        return None  # No inverse exists
    return (x % m + m) % m

# ULTRA: Chinese Remainder Theorem
def chinese_remainder(remainders, moduli):
    """Solve system of congruences x ≡ r_i (mod m_i)."""
    if not remainders or not moduli:
        return None

    # Start with first equation
    x, m = remainders[0], moduli[0]

    for i in range(1, len(remainders)):
        r, n = remainders[i], moduli[i]
        g = gcd(m, n)

        if (r - x) % g != 0:
            return None  # No solution

        lcm = m * n // g
        _, p, q = extended_gcd(m, n)

        x = (x + m * ((r - x) // g) * p) % lcm
        m = lcm

    return x % m if m > 0 else x

# ULTRA: Euler's Totient Function
def euler_phi(n):
    """Compute Euler's totient function phi(n)."""
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

def euler_phi_sieve(n):
    """Compute phi for all numbers 1 to n using sieve."""
    phi = list(range(n + 1))
    for i in range(2, n + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, n + 1, i):
                phi[j] -= phi[j] // i
    return phi

# ULTRA: Mobius Function
def mobius_sieve(n):
    """Compute Mobius function for all numbers 1 to n."""
    mu = [1] * (n + 1)
    is_prime = [True] * (n + 1)
    primes = []

    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1

        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]

    return mu

# ULTRA: Discrete Logarithm (Baby-step Giant-step)
def discrete_log(g, h, p):
    """Find x such that g^x ≡ h (mod p)."""
    n = isqrt(p) + 1

    # Baby step: compute g^j for j = 0..n-1
    baby = {}
    val = 1
    for j in range(n):
        if val == h:
            return j
        baby[val] = j
        val = (val * g) % p

    # Giant step: compute g^(-n)
    factor = pow(g, n * (p - 2), p)  # g^(-n) mod p

    # Look for match
    gamma = h
    for i in range(n):
        if gamma in baby:
            x = i * n + baby[gamma]
            return x
        gamma = (gamma * factor) % p

    return None

# ULTRA: Primitive Root
def primitive_root(p):
    """Find smallest primitive root modulo prime p."""
    if p == 2:
        return 1

    phi = p - 1
    # Find prime factors of phi
    factors = []
    n = phi
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)

    for g in range(2, p):
        is_primitive = True
        for f in factors:
            if pow(g, phi // f, p) == 1:
                is_primitive = False
                break
        if is_primitive:
            return g

    return None

# ULTRA: Legendre Symbol
def legendre(a, p):
    """Compute Legendre symbol (a/p)."""
    return pow(a, (p - 1) // 2, p)

# ULTRA: Tonelli-Shanks (Modular Square Root)
def mod_sqrt(n, p):
    """Find x such that x^2 ≡ n (mod p)."""
    if legendre(n, p) != 1:
        return None  # No solution

    if p % 4 == 3:
        return pow(n, (p + 1) // 4, p)

    # Tonelli-Shanks
    q, s = p - 1, 0
    while q % 2 == 0:
        q //= 2
        s += 1

    # Find quadratic non-residue
    z = 2
    while legendre(z, p) != p - 1:
        z += 1

    m = s
    c = pow(z, q, p)
    t = pow(n, q, p)
    r = pow(n, (q + 1) // 2, p)

    while True:
        if t == 1:
            return r

        # Find least i such that t^(2^i) = 1
        i = 1
        temp = (t * t) % p
        while temp != 1:
            temp = (temp * temp) % p
            i += 1

        b = pow(c, 1 << (m - i - 1), p)
        m = i
        c = (b * b) % p
        t = (t * c) % p
        r = (r * b) % p

# ULTRA: Lucas' Theorem
def lucas(n, k, p):
    """Compute C(n, k) mod p using Lucas' theorem."""
    if k > n:
        return 0

    # Precompute factorials mod p
    fact = [1] * p
    for i in range(1, p):
        fact[i] = (fact[i - 1] * i) % p

    def small_comb(a, b):
        if b > a:
            return 0
        return (fact[a] * pow(fact[b], p - 2, p) * pow(fact[a - b], p - 2, p)) % p

    result = 1
    while n > 0 or k > 0:
        ni, ki = n % p, k % p
        result = (result * small_comb(ni, ki)) % p
        n //= p
        k //= p

    return result

# Tests
tests = []

# Extended GCD
g, x, y = extended_gcd(35, 15)
tests.append(("ext_gcd", g, 5))
tests.append(("ext_gcd_verify", 35 * x + 15 * y, 5))

# Mod inverse
tests.append(("mod_inv", mod_inverse(3, 11), 4))  # 3 * 4 = 12 ≡ 1 (mod 11)
tests.append(("mod_inv2", mod_inverse(7, 26), 15))  # 7 * 15 = 105 ≡ 1 (mod 26)

# Chinese remainder theorem
tests.append(("crt", chinese_remainder([2, 3, 2], [3, 5, 7]), 23))

# Euler phi
tests.append(("phi_12", euler_phi(12), 4))
tests.append(("phi_prime", euler_phi(13), 12))

# Mobius
mu = mobius_sieve(10)
tests.append(("mobius_1", mu[1], 1))
tests.append(("mobius_6", mu[6], 1))  # 6 = 2*3, squarefree with 2 factors
tests.append(("mobius_4", mu[4], 0))  # 4 = 2^2, has square factor

# Discrete log
tests.append(("dlog", discrete_log(2, 8, 13), 3))  # 2^3 = 8 mod 13

# Primitive root
tests.append(("prim_root_7", primitive_root(7), 3))

# Legendre symbol
tests.append(("legendre_qr", legendre(2, 7), 1))  # 2 is QR mod 7 (3^2=9≡2)
tests.append(("legendre_nqr", legendre(3, 7), 6))  # 3 is NQR mod 7

# Mod sqrt
r = mod_sqrt(2, 7)
tests.append(("mod_sqrt", (r * r) % 7 if r else None, 2))

# Lucas
tests.append(("lucas", lucas(10, 3, 7), 1))  # C(10,3) = 120 ≡ 1 (mod 7)

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
