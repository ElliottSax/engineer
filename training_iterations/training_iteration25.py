def random_pick_with_weight(w):
    """Picks index proportional to weight."""
    import random
    prefix = []
    total = 0
    for weight in w:
        total += weight
        prefix.append(total)

    def pick():
        target = random.random() * total
        left, right = 0, len(prefix) - 1
        while left < right:
            mid = (left + right) // 2
            if prefix[mid] <= target:
                left = mid + 1
            else:
                right = mid
        return left
    return pick, total

def shuffle_array(nums):
    """Fisher-Yates shuffle."""
    import random
    original = nums[:]

    def shuffle():
        arr = original[:]
        for i in range(len(arr) - 1, 0, -1):
            j = random.randint(0, i)
            arr[i], arr[j] = arr[j], arr[i]
        return arr

    def reset():
        return original[:]

    return shuffle, reset

def reservoir_sampling(stream, k):
    """Selects k random items from stream."""
    import random
    reservoir = stream[:k]
    for i in range(k, len(stream)):
        j = random.randint(0, i)
        if j < k:
            reservoir[j] = stream[i]
    return reservoir

def random_point_in_circle(radius, x_center, y_center):
    """Returns random point uniformly in circle."""
    import random
    import math
    r = radius * math.sqrt(random.random())
    theta = random.random() * 2 * math.pi
    return [x_center + r * math.cos(theta), y_center + r * math.sin(theta)]

def implement_rand10_using_rand7(rand7):
    """Implements rand10 using rand7."""
    while True:
        row = rand7()
        col = rand7()
        idx = (row - 1) * 7 + col
        if idx <= 40:
            return 1 + (idx - 1) % 10

def sample_online(stream_iter, k):
    """Online reservoir sampling."""
    import random
    reservoir = []
    for i, item in enumerate(stream_iter):
        if i < k:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir

# Probability and Statistics
def is_valid_probability_distribution(probs):
    """Checks if probabilities sum to 1."""
    return abs(sum(probs) - 1.0) < 1e-9

def binomial_coefficient(n, k):
    """Computes C(n, k)."""
    if k > n or k < 0:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result

def pascal_triangle(n):
    """Generates first n rows of Pascal's triangle."""
    result = []
    for i in range(n):
        row = [1] * (i + 1)
        for j in range(1, i):
            row[j] = result[i-1][j-1] + result[i-1][j]
        result.append(row)
    return result

def pascal_triangle_row(n):
    """Gets nth row of Pascal's triangle (0-indexed)."""
    row = [1]
    for k in range(1, n + 1):
        row.append(row[-1] * (n - k + 1) // k)
    return row

# Bit manipulation
def single_number_ii(nums):
    """Finds number appearing once when others appear 3 times."""
    ones = twos = 0
    for num in nums:
        ones = (ones ^ num) & ~twos
        twos = (twos ^ num) & ~ones
    return ones

def single_number_iii(nums):
    """Finds two numbers appearing once when others appear twice."""
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

def total_hamming_distance(nums):
    """Sum of Hamming distances between all pairs."""
    total = 0
    for i in range(32):
        count = sum((num >> i) & 1 for num in nums)
        total += count * (len(nums) - count)
    return total

def maximum_xor_of_two(nums):
    """Maximum XOR of any two numbers."""
    max_xor = 0
    mask = 0
    for i in range(31, -1, -1):
        mask |= (1 << i)
        prefixes = {num & mask for num in nums}
        candidate = max_xor | (1 << i)
        for prefix in prefixes:
            if prefix ^ candidate in prefixes:
                max_xor = candidate
                break
    return max_xor

def range_bitwise_and(left, right):
    """Bitwise AND of all numbers in range."""
    shift = 0
    while left < right:
        left >>= 1
        right >>= 1
        shift += 1
    return left << shift

# Tests
import random
random.seed(42)

# Pascal triangle tests
tests = [
    ("pascal", pascal_triangle(5), [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]]),
    ("pascal_row", pascal_triangle_row(4), [1, 4, 6, 4, 1]),
    ("binomial", binomial_coefficient(5, 2), 10),
    ("binomial_edge", binomial_coefficient(10, 0), 1),
    ("prob_valid", is_valid_probability_distribution([0.2, 0.3, 0.5]), True),
    ("prob_invalid", is_valid_probability_distribution([0.2, 0.3, 0.4]), False),
    ("single_ii", single_number_ii([2, 2, 3, 2]), 3),
    ("single_iii", single_number_iii([1, 2, 1, 3, 2, 5]), [3, 5]),
    ("hamming_total", total_hamming_distance([4, 14, 2]), 6),
    ("max_xor", maximum_xor_of_two([3, 10, 5, 25, 2, 8]), 28),
    ("range_and", range_bitwise_and(5, 7), 4),
    ("range_and_2", range_bitwise_and(12, 15), 12),
]

# Weighted random pick test
pick, total = random_pick_with_weight([1, 3])
counts = [0, 0]
for _ in range(1000):
    counts[pick()] += 1
# Should be roughly 25% for 0, 75% for 1
tests.append(("weighted_pick", 0.15 < counts[0]/1000 < 0.35, True))

# Shuffle test
shuffle, reset = shuffle_array([1, 2, 3])
shuffled = shuffle()
tests.append(("shuffle_len", len(shuffled), 3))
tests.append(("shuffle_elements", sorted(shuffled), [1, 2, 3]))
tests.append(("reset", reset(), [1, 2, 3]))

# Reservoir sampling test
stream = list(range(100))
sample = reservoir_sampling(stream, 10)
tests.append(("reservoir_len", len(sample), 10))
tests.append(("reservoir_range", all(0 <= x < 100 for x in sample), True))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
