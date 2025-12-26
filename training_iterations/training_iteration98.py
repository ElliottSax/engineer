# EXTREME: FFT, NTT, and Polynomial Algorithms

import cmath
from math import pi

# HARD: Fast Fourier Transform
def fft(a, invert=False):
    """Cooley-Tukey FFT algorithm."""
    n = len(a)
    if n == 1:
        return a

    # Bit reversal
    result = a[:]
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            result[i], result[j] = result[j], result[i]

    # Butterfly
    length = 2
    while length <= n:
        angle = 2 * pi / length * (-1 if invert else 1)
        wlen = complex(cmath.cos(angle), cmath.sin(angle))
        for i in range(0, n, length):
            w = complex(1)
            for j in range(length // 2):
                u = result[i + j]
                v = result[i + j + length // 2] * w
                result[i + j] = u + v
                result[i + j + length // 2] = u - v
                w *= wlen
        length *= 2

    if invert:
        result = [x / n for x in result]

    return result

def multiply_polynomials(a, b):
    """Multiply two polynomials using FFT."""
    result_len = len(a) + len(b) - 1
    n = 1
    while n < result_len:
        n *= 2

    fa = [complex(x) for x in a] + [complex(0)] * (n - len(a))
    fb = [complex(x) for x in b] + [complex(0)] * (n - len(b))

    fa = fft(fa)
    fb = fft(fb)

    fc = [fa[i] * fb[i] for i in range(n)]
    fc = fft(fc, invert=True)

    return [round(x.real) for x in fc[:result_len]]

# HARD: Number Theoretic Transform
def ntt(a, mod, root, invert=False):
    """Number Theoretic Transform."""
    n = len(a)
    if n == 1:
        return a

    result = a[:]
    # Bit reversal
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            result[i], result[j] = result[j], result[i]

    # Butterfly
    length = 2
    while length <= n:
        w = pow(root, (mod - 1) // length, mod)
        if invert:
            w = pow(w, mod - 2, mod)
        for i in range(0, n, length):
            wn = 1
            for k in range(length // 2):
                u = result[i + k]
                v = result[i + k + length // 2] * wn % mod
                result[i + k] = (u + v) % mod
                result[i + k + length // 2] = (u - v) % mod
                wn = wn * w % mod
        length *= 2

    if invert:
        n_inv = pow(n, mod - 2, mod)
        result = [x * n_inv % mod for x in result]

    return result

def multiply_mod(a, b, mod=998244353, root=3):
    """Multiply polynomials modulo mod using NTT."""
    result_len = len(a) + len(b) - 1
    n = 1
    while n < result_len:
        n *= 2

    fa = a + [0] * (n - len(a))
    fb = b + [0] * (n - len(b))

    fa = ntt(fa, mod, root)
    fb = ntt(fb, mod, root)

    fc = [fa[i] * fb[i] % mod for i in range(n)]
    fc = ntt(fc, mod, root, invert=True)

    return fc[:result_len]

# HARD: Polynomial Division
def poly_divide(a, b):
    """Divide polynomial a by b, return (quotient, remainder)."""
    if not b or all(x == 0 for x in b):
        raise ValueError("Division by zero polynomial")

    # Remove leading zeros
    while a and a[-1] == 0:
        a = a[:-1]
    while b and b[-1] == 0:
        b = b[:-1]

    if not a:
        return [0], [0]
    if len(a) < len(b):
        return [0], a[:]

    quotient = [0] * (len(a) - len(b) + 1)
    remainder = a[:]

    for i in range(len(quotient) - 1, -1, -1):
        quotient[i] = remainder[i + len(b) - 1] // b[-1]
        for j in range(len(b)):
            remainder[i + j] -= quotient[i] * b[j]

    # Remove leading zeros from remainder
    while remainder and remainder[-1] == 0:
        remainder.pop()
    if not remainder:
        remainder = [0]

    return quotient, remainder

# HARD: Polynomial GCD
def poly_gcd(a, b):
    """GCD of two polynomials."""
    while b and any(x != 0 for x in b):
        _, remainder = poly_divide(a, b)
        a, b = b, remainder
    # Normalize
    if a and a[-1] != 0:
        factor = a[-1]
        a = [x // factor for x in a]
    return a if a else [0]

# HARD: Karatsuba Multiplication
def karatsuba(x, y):
    """Karatsuba multiplication for large integers."""
    if x < 10 or y < 10:
        return x * y

    n = max(len(str(x)), len(str(y)))
    m = n // 2

    high_x, low_x = divmod(x, 10**m)
    high_y, low_y = divmod(y, 10**m)

    z0 = karatsuba(low_x, low_y)
    z2 = karatsuba(high_x, high_y)
    z1 = karatsuba(low_x + high_x, low_y + high_y) - z0 - z2

    return z2 * 10**(2*m) + z1 * 10**m + z0

# HARD: Strassen Matrix Multiplication (simplified 2x2)
def strassen_2x2(A, B):
    """Strassen multiplication for 2x2 matrices."""
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

# Polynomial multiplication with FFT
a = [1, 2, 1]  # 1 + 2x + x^2 = (1+x)^2
b = [1, 1]     # 1 + x
result = multiply_polynomials(a, b)
tests.append(("fft_mult", result, [1, 3, 3, 1]))  # (1+x)^3

# NTT multiplication
a = [1, 2, 1]
b = [1, 1]
result = multiply_mod(a, b)
tests.append(("ntt_mult", result, [1, 3, 3, 1]))

# Polynomial division
q, r = poly_divide([1, 0, -1], [1, 1])  # (x^2 - 1) / (x + 1)
tests.append(("poly_div_q", q, [-1, 1]))  # x - 1
tests.append(("poly_div_r", r, [0]))

# Polynomial GCD
gcd = poly_gcd([1, 0, -1], [1, 1])  # gcd(x^2-1, x+1) = x+1
tests.append(("poly_gcd", gcd, [1, 1]))

# Karatsuba
tests.append(("karatsuba", karatsuba(1234, 5678), 1234 * 5678))
tests.append(("karatsuba_big", karatsuba(123456789, 987654321), 123456789 * 987654321))

# Strassen 2x2
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
result = strassen_2x2(A, B)
expected = [[19, 22], [43, 50]]
tests.append(("strassen", result, expected))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
