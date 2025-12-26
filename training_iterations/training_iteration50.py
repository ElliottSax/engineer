def single_number(nums):
    """Find element appearing once (others appear twice)."""
    result = 0
    for num in nums:
        result ^= num
    return result

def single_number_ii(nums):
    """Find element appearing once (others appear three times)."""
    ones = twos = 0
    for num in nums:
        ones = (ones ^ num) & ~twos
        twos = (twos ^ num) & ~ones
    return ones

def single_number_iii(nums):
    """Find two elements appearing once (others appear twice)."""
    xor = 0
    for num in nums:
        xor ^= num

    # Find rightmost set bit
    diff_bit = xor & (-xor)

    a = b = 0
    for num in nums:
        if num & diff_bit:
            a ^= num
        else:
            b ^= num
    return sorted([a, b])

def count_bits(n):
    """Count 1 bits for all numbers 0 to n."""
    result = [0] * (n + 1)
    for i in range(1, n + 1):
        result[i] = result[i >> 1] + (i & 1)
    return result

def hamming_distance(x, y):
    """Count differing bits between x and y."""
    xor = x ^ y
    count = 0
    while xor:
        count += xor & 1
        xor >>= 1
    return count

def total_hamming_distance(nums):
    """Sum of hamming distances between all pairs."""
    total = 0
    n = len(nums)
    for bit in range(32):
        ones = sum(1 for num in nums if num & (1 << bit))
        total += ones * (n - ones)
    return total

def missing_number(nums):
    """Find missing number in [0, n]."""
    n = len(nums)
    expected = n * (n + 1) // 2
    return expected - sum(nums)

def missing_number_xor(nums):
    """Find missing number using XOR."""
    result = len(nums)
    for i, num in enumerate(nums):
        result ^= i ^ num
    return result

def power_of_two(n):
    """Check if n is power of 2."""
    return n > 0 and (n & (n - 1)) == 0

def power_of_four(n):
    """Check if n is power of 4."""
    return n > 0 and (n & (n - 1)) == 0 and (n & 0x55555555) == n

def reverse_bits(n):
    """Reverse bits of 32-bit integer."""
    result = 0
    for _ in range(32):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result

def bitwise_and_of_range(left, right):
    """Bitwise AND of all numbers in [left, right]."""
    shift = 0
    while left < right:
        left >>= 1
        right >>= 1
        shift += 1
    return left << shift

def subsets_bitmask(nums):
    """Generate all subsets using bitmask."""
    n = len(nums)
    result = []
    for mask in range(1 << n):
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(nums[i])
        result.append(subset)
    return result

def max_xor_of_two(nums):
    """Maximum XOR of two numbers in array."""
    max_xor = 0
    mask = 0
    for i in range(31, -1, -1):
        mask |= (1 << i)
        prefixes = set(num & mask for num in nums)
        candidate = max_xor | (1 << i)
        for prefix in prefixes:
            if prefix ^ candidate in prefixes:
                max_xor = candidate
                break
    return max_xor

def gray_code(n):
    """Generate n-bit Gray code sequence."""
    return [i ^ (i >> 1) for i in range(1 << n)]

def decode_xored_array(encoded, first):
    """Decode XORed array given first element."""
    result = [first]
    for e in encoded:
        result.append(result[-1] ^ e)
    return result

def xor_queries(arr, queries):
    """XOR of subarray for each query."""
    n = len(arr)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] ^ arr[i]
    return [prefix[r + 1] ^ prefix[l] for l, r in queries]

# Tests
tests = [
    ("single", single_number([2,2,1]), 1),
    ("single_ii", single_number_ii([0,1,0,1,0,1,99]), 99),
    ("single_iii", single_number_iii([1,2,1,3,2,5]), [3, 5]),
    ("count_bits", count_bits(5), [0, 1, 1, 2, 1, 2]),
    ("hamming", hamming_distance(1, 4), 2),
    ("total_hamming", total_hamming_distance([4, 14, 2]), 6),
    ("missing", missing_number([3, 0, 1]), 2),
    ("missing_xor", missing_number_xor([9,6,4,2,3,5,7,0,1]), 8),
    ("pow2_true", power_of_two(16), True),
    ("pow2_false", power_of_two(6), False),
    ("pow4_true", power_of_four(16), True),
    ("pow4_false", power_of_four(8), False),
    ("reverse_bits", reverse_bits(43261596), 964176192),
    ("and_range", bitwise_and_of_range(5, 7), 4),
    ("subsets", len(subsets_bitmask([1, 2, 3])), 8),
    ("max_xor", max_xor_of_two([3, 10, 5, 25, 2, 8]), 28),
    ("gray_code", gray_code(2), [0, 1, 3, 2]),
    ("decode_xor", decode_xored_array([1, 2, 3], 1), [1, 0, 2, 1]),
    ("xor_queries", xor_queries([1, 3, 4, 8], [[0, 1], [1, 2], [0, 3], [3, 3]]), [2, 7, 14, 8]),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
