# EXTREME: Probabilistic Data Structures

import hashlib
import math
import random

# HARD: Bloom Filter
class BloomFilter:
    def __init__(self, n, fp_rate=0.01):
        """Create bloom filter for n elements with fp_rate false positive rate."""
        self.size = int(-n * math.log(fp_rate) / (math.log(2) ** 2))
        self.num_hashes = int(self.size / n * math.log(2))
        self.bit_array = [False] * self.size

    def _hashes(self, item):
        """Generate hash positions."""
        h1 = hash(item)
        h2 = hash(str(item) + "_2")
        for i in range(self.num_hashes):
            yield (h1 + i * h2) % self.size

    def add(self, item):
        """Add item to bloom filter."""
        for pos in self._hashes(item):
            self.bit_array[pos] = True

    def __contains__(self, item):
        """Check if item might be in the set."""
        return all(self.bit_array[pos] for pos in self._hashes(item))

# HARD: Count-Min Sketch
class CountMinSketch:
    def __init__(self, width, depth):
        """Create count-min sketch with given dimensions."""
        self.width = width
        self.depth = depth
        self.table = [[0] * width for _ in range(depth)]

    def _hash(self, item, i):
        """Generate hash for row i."""
        h = hash(str(item) + f"_{i}")
        return h % self.width

    def add(self, item, count=1):
        """Add item with given count."""
        for i in range(self.depth):
            self.table[i][self._hash(item, i)] += count

    def estimate(self, item):
        """Estimate count of item."""
        return min(self.table[i][self._hash(item, i)] for i in range(self.depth))

# HARD: HyperLogLog
class HyperLogLog:
    def __init__(self, p=14):
        """Create HyperLogLog with 2^p registers."""
        self.p = p
        self.m = 1 << p
        self.registers = [0] * self.m

        # Alpha constant
        if self.m == 16:
            self.alpha = 0.673
        elif self.m == 32:
            self.alpha = 0.697
        elif self.m == 64:
            self.alpha = 0.709
        else:
            self.alpha = 0.7213 / (1 + 1.079 / self.m)

    def _hash(self, item):
        """64-bit hash."""
        h = hashlib.sha256(str(item).encode()).hexdigest()
        return int(h[:16], 16)

    def _rho(self, w):
        """Position of leftmost 1-bit."""
        if w == 0:
            return 64 - self.p
        return (64 - self.p) - w.bit_length() + 1

    def add(self, item):
        """Add item."""
        x = self._hash(item)
        j = x & (self.m - 1)  # First p bits
        w = x >> self.p       # Remaining bits
        self.registers[j] = max(self.registers[j], self._rho(w) + 1)

    def count(self):
        """Estimate cardinality."""
        Z = sum(2 ** (-r) for r in self.registers)
        E = self.alpha * self.m * self.m / Z

        # Small range correction
        if E <= 2.5 * self.m:
            V = self.registers.count(0)
            if V > 0:
                return self.m * math.log(self.m / V)

        # Large range correction
        if E > (1 << 32) / 30:
            return -(1 << 32) * math.log(1 - E / (1 << 32))

        return E

# HARD: Skip List (probabilistic)
class SkipNode:
    def __init__(self, val, level):
        self.val = val
        self.forward = [None] * (level + 1)

class ProbSkipList:
    def __init__(self, max_level=16, p=0.5):
        self.max_level = max_level
        self.p = p
        self.level = 0
        self.head = SkipNode(float('-inf'), max_level)

    def _random_level(self):
        lvl = 0
        while random.random() < self.p and lvl < self.max_level:
            lvl += 1
        return lvl

    def insert(self, val):
        update = [None] * (self.max_level + 1)
        current = self.head

        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].val < val:
                current = current.forward[i]
            update[i] = current

        lvl = self._random_level()
        if lvl > self.level:
            for i in range(self.level + 1, lvl + 1):
                update[i] = self.head
            self.level = lvl

        new_node = SkipNode(val, lvl)
        for i in range(lvl + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node

    def search(self, val):
        current = self.head
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].val < val:
                current = current.forward[i]
        current = current.forward[0]
        return current and current.val == val

# HARD: Reservoir Sampling
def reservoir_sample(stream, k):
    """Sample k elements uniformly from stream."""
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir

# HARD: Weighted Reservoir Sampling
def weighted_reservoir_sample(stream, k):
    """Sample k elements with weights from stream."""
    import heapq
    heap = []

    for item, weight in stream:
        key = random.random() ** (1 / weight)
        if len(heap) < k:
            heapq.heappush(heap, (key, item))
        elif key > heap[0][0]:
            heapq.heapreplace(heap, (key, item))

    return [item for _, item in heap]

# HARD: Consistent Hashing
class ConsistentHash:
    def __init__(self, num_replicas=100):
        self.num_replicas = num_replicas
        self.ring = {}
        self.sorted_keys = []

    def _hash(self, key):
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_node(self, node):
        for i in range(self.num_replicas):
            key = self._hash(f"{node}:{i}")
            self.ring[key] = node
            self.sorted_keys.append(key)
        self.sorted_keys.sort()

    def remove_node(self, node):
        for i in range(self.num_replicas):
            key = self._hash(f"{node}:{i}")
            del self.ring[key]
            self.sorted_keys.remove(key)

    def get_node(self, key):
        if not self.ring:
            return None
        h = self._hash(key)
        # Binary search for first node >= h
        left, right = 0, len(self.sorted_keys)
        while left < right:
            mid = (left + right) // 2
            if self.sorted_keys[mid] < h:
                left = mid + 1
            else:
                right = mid
        if left == len(self.sorted_keys):
            left = 0
        return self.ring[self.sorted_keys[left]]

# Tests
tests = []

# Bloom Filter
bloom = BloomFilter(1000, 0.01)
for i in range(100):
    bloom.add(f"item_{i}")
tests.append(("bloom_positive", "item_50" in bloom, True))
tests.append(("bloom_negative", "item_999" in bloom, False))  # Should be false (with high probability)

# Count-Min Sketch
cms = CountMinSketch(100, 5)
for i in range(50):
    cms.add("apple")
for i in range(30):
    cms.add("banana")
tests.append(("cms_apple", cms.estimate("apple") >= 50, True))
tests.append(("cms_banana", cms.estimate("banana") >= 30, True))

# HyperLogLog
hll = HyperLogLog(10)
for i in range(10000):
    hll.add(f"item_{i}")
estimate = hll.count()
tests.append(("hll_accuracy", 8000 < estimate < 12000, True))  # Within 20% error

# Skip List
random.seed(42)
skip = ProbSkipList()
for val in [3, 1, 4, 1, 5, 9, 2, 6]:
    skip.insert(val)
tests.append(("skip_found", skip.search(5), True))
tests.append(("skip_not", skip.search(7), False))

# Reservoir Sampling
random.seed(42)
sample = reservoir_sample(range(1000), 10)
tests.append(("reservoir_size", len(sample), 10))

# Consistent Hashing
ch = ConsistentHash(10)
ch.add_node("server1")
ch.add_node("server2")
ch.add_node("server3")
tests.append(("consistent_hash", ch.get_node("key1") in ["server1", "server2", "server3"], True))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
