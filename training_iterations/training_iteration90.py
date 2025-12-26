# EXTREME: Advanced Bit Manipulation

# HARD: Single Number III - Two different numbers appear once
def single_number_iii(nums):
    """Find two numbers that appear exactly once."""
    xor = 0
    for num in nums:
        xor ^= num

    # Get rightmost set bit
    diff_bit = xor & (-xor)

    a = b = 0
    for num in nums:
        if num & diff_bit:
            a ^= num
        else:
            b ^= num

    return sorted([a, b])

# HARD: Maximum XOR of Two Numbers
def max_xor(nums):
    """Find maximum XOR of any two numbers."""
    max_xor_val = 0
    mask = 0

    for i in range(31, -1, -1):
        mask |= (1 << i)
        prefixes = {num & mask for num in nums}
        candidate = max_xor_val | (1 << i)

        for prefix in prefixes:
            if (prefix ^ candidate) in prefixes:
                max_xor_val = candidate
                break

    return max_xor_val

# HARD: Count Bits in Range
def count_bits_range(n):
    """Count total bits set in 0 to n."""
    if n < 0:
        return 0

    # dp[i] = total bits in 0 to 2^i - 1
    def count_up_to(x):
        if x < 0:
            return 0
        if x == 0:
            return 0

        # Find highest bit
        highest = x.bit_length() - 1
        # Count for numbers below 2^highest
        count = highest * (1 << (highest - 1)) if highest > 0 else 0
        # Add the highest bit contribution
        count += x - (1 << highest) + 1
        # Add remaining bits recursively
        count += count_up_to(x - (1 << highest))
        return count

    return count_up_to(n)

# HARD: Divide Two Integers (bit manipulation)
def divide(dividend, divisor):
    """Divide without multiplication/division/mod."""
    if dividend == -2**31 and divisor == -1:
        return 2**31 - 1

    negative = (dividend < 0) != (divisor < 0)
    dividend, divisor = abs(dividend), abs(divisor)

    result = 0
    while dividend >= divisor:
        shift = 0
        while dividend >= (divisor << (shift + 1)):
            shift += 1
        result += (1 << shift)
        dividend -= (divisor << shift)

    return -result if negative else result

# HARD: Gray Code
def gray_code(n):
    """Generate n-bit Gray code sequence."""
    result = [0]
    for i in range(n):
        # Mirror and add 1 << i
        result.extend([x | (1 << i) for x in reversed(result)])
    return result

# HARD: Minimum Flips to Make a OR b Equal to c
def min_flips(a, b, c):
    """Minimum bit flips to make a | b = c."""
    flips = 0
    for i in range(32):
        bit_a = (a >> i) & 1
        bit_b = (b >> i) & 1
        bit_c = (c >> i) & 1

        if bit_c == 0:
            flips += bit_a + bit_b
        else:  # bit_c == 1
            if bit_a == 0 and bit_b == 0:
                flips += 1

    return flips

# HARD: Bitwise AND of Numbers Range
def range_bitwise_and(left, right):
    """Bitwise AND of all numbers in [left, right]."""
    shift = 0
    while left < right:
        left >>= 1
        right >>= 1
        shift += 1
    return left << shift

# HARD: UTF-8 Validation
def valid_utf8(data):
    """Check if byte array represents valid UTF-8."""
    remaining = 0

    for byte in data:
        if remaining > 0:
            if byte >> 6 != 0b10:
                return False
            remaining -= 1
        else:
            if byte >> 7 == 0:
                remaining = 0
            elif byte >> 5 == 0b110:
                remaining = 1
            elif byte >> 4 == 0b1110:
                remaining = 2
            elif byte >> 3 == 0b11110:
                remaining = 3
            else:
                return False

    return remaining == 0

# HARD: Concatenation of Consecutive Binary Numbers
def concatenated_binary(n):
    """Decimal value of concatenating binary of 1 to n."""
    MOD = 10**9 + 7
    result = 0
    length = 0

    for i in range(1, n + 1):
        if i & (i - 1) == 0:  # Power of 2
            length += 1
        result = ((result << length) | i) % MOD

    return result

# HARD: Reverse Bits
def reverse_bits(n):
    """Reverse bits of 32-bit unsigned integer."""
    result = 0
    for _ in range(32):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result

# HARD: Number of 1 Bits (Hamming Weight)
def hamming_weight(n):
    """Count number of 1 bits."""
    count = 0
    while n:
        n &= (n - 1)  # Remove lowest set bit
        count += 1
    return count

# HARD: Power of Two, Three, Four
def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0

def is_power_of_three(n):
    return n > 0 and 1162261467 % n == 0  # 3^19

def is_power_of_four(n):
    return n > 0 and (n & (n - 1)) == 0 and (n & 0xAAAAAAAA) == 0

# HARD: Sum of Two Integers (without +)
def get_sum(a, b):
    """Add two integers without + operator."""
    MASK = 0xFFFFFFFF
    MAX_INT = 0x7FFFFFFF

    while b != 0:
        carry = ((a & b) << 1) & MASK
        a = (a ^ b) & MASK
        b = carry

    return a if a <= MAX_INT else ~(a ^ MASK)

# Tests
tests = []

# Single Number III
tests.append(("single_iii", single_number_iii([1,2,1,3,2,5]), [3, 5]))

# Maximum XOR
tests.append(("max_xor", max_xor([3, 10, 5, 25, 2, 8]), 28))
tests.append(("max_xor2", max_xor([14, 70, 53, 83, 49, 91, 36, 80, 92, 51, 66, 70]), 127))

# Count Bits
tests.append(("count_bits", count_bits_range(5), 7))  # 0+1+1+2+1+2=7

# Divide
tests.append(("divide", divide(10, 3), 3))
tests.append(("divide2", divide(-7, 3), -2))

# Gray Code
tests.append(("gray", gray_code(2), [0, 1, 3, 2]))

# Min Flips
tests.append(("flips", min_flips(2, 6, 5), 3))
tests.append(("flips2", min_flips(4, 2, 7), 1))

# Range AND
tests.append(("range_and", range_bitwise_and(5, 7), 4))
tests.append(("range_and2", range_bitwise_and(1, 2147483647), 0))

# UTF-8
tests.append(("utf8", valid_utf8([197, 130, 1]), True))
tests.append(("utf8_bad", valid_utf8([235, 140, 4]), False))

# Concatenated Binary
tests.append(("concat", concatenated_binary(3), 27))  # "1" + "10" + "11" = 11011 = 27
tests.append(("concat2", concatenated_binary(12), 505379714))

# Reverse Bits
tests.append(("reverse", reverse_bits(43261596), 964176192))

# Hamming Weight
tests.append(("hamming", hamming_weight(11), 3))
tests.append(("hamming2", hamming_weight(128), 1))

# Power checks
tests.append(("pow2", is_power_of_two(16), True))
tests.append(("pow3", is_power_of_three(27), True))
tests.append(("pow4", is_power_of_four(16), True))
tests.append(("pow4_no", is_power_of_four(8), False))

# Sum
tests.append(("sum", get_sum(1, 2), 3))
tests.append(("sum_neg", get_sum(-1, 1), 0))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
