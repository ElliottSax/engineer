#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ULTRA TRAINING ITERATION 126                              ║
║                    Bit Manipulation Masterclass                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════════
# BIT TRICK 1: Count Set Bits (Population Count)
# ═══════════════════════════════════════════════════════════════════════════════
def count_bits(n):
    """Count number of 1 bits in n."""
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count

def count_bits_fast(n):
    """Brian Kernighan's algorithm - O(number of set bits)."""
    count = 0
    while n:
        n &= n - 1  # Clear lowest set bit
        count += 1
    return count

# ═══════════════════════════════════════════════════════════════════════════════
# BIT TRICK 2: Power of Two Check
# ═══════════════════════════════════════════════════════════════════════════════
def is_power_of_two(n):
    """Check if n is power of 2."""
    return n > 0 and (n & (n - 1)) == 0

def next_power_of_two(n):
    """Find smallest power of 2 >= n."""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1

# ═══════════════════════════════════════════════════════════════════════════════
# BIT TRICK 3: Single Number Problems
# ═══════════════════════════════════════════════════════════════════════════════
def single_number_xor(nums):
    """Find single number where all others appear twice."""
    result = 0
    for n in nums:
        result ^= n
    return result

def single_number_three(nums):
    """Find two numbers that appear once (others appear twice)."""
    xor_all = 0
    for n in nums:
        xor_all ^= n

    # Find rightmost set bit (differentiating bit)
    diff_bit = xor_all & (-xor_all)

    a = b = 0
    for n in nums:
        if n & diff_bit:
            a ^= n
        else:
            b ^= n

    return sorted([a, b])

# ═══════════════════════════════════════════════════════════════════════════════
# BIT TRICK 4: Bit Manipulation for Sets
# ═══════════════════════════════════════════════════════════════════════════════
def subset_sum_bitmask(nums, target):
    """Find if subset with given sum exists using bitmask DP."""
    n = len(nums)
    for mask in range(1 << n):
        total = sum(nums[i] for i in range(n) if mask & (1 << i))
        if total == target:
            return True
    return False

def count_subsets_with_sum(nums, target):
    """Count subsets with given sum."""
    n = len(nums)
    count = 0
    for mask in range(1 << n):
        total = sum(nums[i] for i in range(n) if mask & (1 << i))
        if total == target:
            count += 1
    return count

# ═══════════════════════════════════════════════════════════════════════════════
# BIT TRICK 5: Gray Code
# ═══════════════════════════════════════════════════════════════════════════════
def gray_code(n):
    """Generate n-bit Gray code sequence."""
    return [i ^ (i >> 1) for i in range(1 << n)]

def binary_to_gray(n):
    """Convert binary to Gray code."""
    return n ^ (n >> 1)

def gray_to_binary(gray):
    """Convert Gray code to binary."""
    binary = 0
    while gray:
        binary ^= gray
        gray >>= 1
    return binary

# ═══════════════════════════════════════════════════════════════════════════════
# BIT TRICK 6: Bit Reversal
# ═══════════════════════════════════════════════════════════════════════════════
def reverse_bits(n, bits=32):
    """Reverse bits of n."""
    result = 0
    for _ in range(bits):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result

def swap_bits(n, i, j):
    """Swap bits at positions i and j."""
    bit_i = (n >> i) & 1
    bit_j = (n >> j) & 1
    if bit_i != bit_j:
        n ^= (1 << i) | (1 << j)
    return n

# ═══════════════════════════════════════════════════════════════════════════════
# BIT TRICK 7: XOR Operations
# ═══════════════════════════════════════════════════════════════════════════════
def xor_range(a, b):
    """Compute XOR of all numbers from a to b."""
    def xor_to_n(n):
        # XOR from 0 to n
        if n < 0:
            return 0
        mod = n % 4
        if mod == 0:
            return n
        elif mod == 1:
            return 1
        elif mod == 2:
            return n + 1
        else:
            return 0

    return xor_to_n(b) ^ xor_to_n(a - 1)

def max_xor_pair(nums):
    """Find maximum XOR of any two numbers in array."""
    max_xor = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            max_xor = max(max_xor, nums[i] ^ nums[j])
    return max_xor

# ═══════════════════════════════════════════════════════════════════════════════
# BIT TRICK 8: Missing/Duplicate Number
# ═══════════════════════════════════════════════════════════════════════════════
def find_missing_xor(nums, n):
    """Find missing number in [0..n] using XOR."""
    xor_all = 0
    for i in range(n + 1):
        xor_all ^= i
    for num in nums:
        xor_all ^= num
    return xor_all

def find_duplicate_xor(nums):
    """Find duplicate when array has n+1 numbers in range [1..n]."""
    n = len(nums) - 1
    xor_all = 0
    for i in range(1, n + 1):
        xor_all ^= i
    for num in nums:
        xor_all ^= num
    return xor_all

# ═══════════════════════════════════════════════════════════════════════════════
#                              TEST SUITE
# ═══════════════════════════════════════════════════════════════════════════════
def run_tests():
    tests = []

    # Test 1: Count bits
    tests.append(("count_7", count_bits(7), 3))
    tests.append(("count_fast", count_bits_fast(255), 8))

    # Test 2: Power of two
    tests.append(("pow2_16", is_power_of_two(16), True))
    tests.append(("pow2_18", is_power_of_two(18), False))
    tests.append(("next_pow2", next_power_of_two(17), 32))

    # Test 3: Single number
    tests.append(("single_xor", single_number_xor([4, 1, 2, 1, 2]), 4))
    tests.append(("single_three", single_number_three([1, 2, 1, 3, 2, 5]), [3, 5]))

    # Test 4: Subset sum
    tests.append(("subset_sum", subset_sum_bitmask([1, 2, 3, 4], 6), True))
    tests.append(("count_subsets", count_subsets_with_sum([1, 2, 3], 3), 2))

    # Test 5: Gray code
    gc = gray_code(3)
    tests.append(("gray_len", len(gc), 8))
    tests.append(("gray_first", gc[0], 0))
    tests.append(("bin_to_gray", binary_to_gray(5), 7))
    tests.append(("gray_to_bin", gray_to_binary(7), 5))

    # Test 6: Bit reversal
    tests.append(("reverse_8", reverse_bits(1, 8), 128))
    tests.append(("swap_bits", swap_bits(0b1010, 1, 2), 0b1100))

    # Test 7: XOR range
    tests.append(("xor_range", xor_range(3, 5), 3 ^ 4 ^ 5))
    tests.append(("max_xor", max_xor_pair([3, 10, 5, 25]), 28))

    # Test 8: Missing/Duplicate
    tests.append(("missing", find_missing_xor([0, 1, 3], 3), 2))
    tests.append(("duplicate", find_duplicate_xor([1, 3, 4, 2, 2]), 2))

    # Run all tests
    passed = 0
    print("\n" + "─" * 60)
    for name, result, expected in tests:
        if result == expected:
            passed += 1
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name}: got {result}, expected {expected}")

    print("─" * 60)
    print(f"\n  📊 Results: {passed}/{len(tests)} tests passed")
    return passed, len(tests)

if __name__ == "__main__":
    print(__doc__)
    passed, total = run_tests()
    if passed == total:
        print("\n  🎯 PERFECT SCORE!")
