def design_hashmap():
    """Implements HashMap with separate chaining."""
    size = 1000
    buckets = [[] for _ in range(size)]

    def _hash(key):
        return key % size

    def put(key, value):
        h = _hash(key)
        for i, (k, v) in enumerate(buckets[h]):
            if k == key:
                buckets[h][i] = (key, value)
                return
        buckets[h].append((key, value))

    def get(key):
        h = _hash(key)
        for k, v in buckets[h]:
            if k == key:
                return v
        return -1

    def remove(key):
        h = _hash(key)
        for i, (k, v) in enumerate(buckets[h]):
            if k == key:
                buckets[h].pop(i)
                return

    return put, get, remove

def design_hashset():
    """Implements HashSet."""
    size = 1000
    buckets = [[] for _ in range(size)]

    def _hash(key):
        return key % size

    def add(key):
        h = _hash(key)
        if key not in buckets[h]:
            buckets[h].append(key)

    def contains(key):
        h = _hash(key)
        return key in buckets[h]

    def remove(key):
        h = _hash(key)
        if key in buckets[h]:
            buckets[h].remove(key)

    return add, contains, remove

def lru_cache_impl(capacity):
    """LRU Cache using OrderedDict."""
    from collections import OrderedDict
    cache = OrderedDict()

    def get(key):
        if key not in cache:
            return -1
        cache.move_to_end(key)
        return cache[key]

    def put(key, value):
        if key in cache:
            cache.move_to_end(key)
        cache[key] = value
        if len(cache) > capacity:
            cache.popitem(last=False)

    return get, put

def lfu_cache_impl(capacity):
    """LFU Cache implementation."""
    from collections import defaultdict, OrderedDict
    cache = {}  # key -> value
    freq = {}   # key -> frequency
    freq_to_keys = defaultdict(OrderedDict)  # freq -> OrderedDict of keys
    min_freq = 0

    def get(key):
        nonlocal min_freq
        if key not in cache:
            return -1
        f = freq[key]
        del freq_to_keys[f][key]
        if not freq_to_keys[f]:
            del freq_to_keys[f]
            if min_freq == f:
                min_freq += 1
        freq[key] = f + 1
        freq_to_keys[f + 1][key] = None
        return cache[key]

    def put(key, value):
        nonlocal min_freq
        if capacity <= 0:
            return
        if key in cache:
            cache[key] = value
            get(key)
            return
        if len(cache) >= capacity:
            evict_key, _ = freq_to_keys[min_freq].popitem(last=False)
            if not freq_to_keys[min_freq]:
                del freq_to_keys[min_freq]
            del cache[evict_key]
            del freq[evict_key]
        cache[key] = value
        freq[key] = 1
        freq_to_keys[1][key] = None
        min_freq = 1

    return get, put

def time_based_key_value():
    """Time-based key-value store."""
    from collections import defaultdict
    import bisect
    store = defaultdict(list)  # key -> [(timestamp, value), ...]

    def set_val(key, value, timestamp):
        store[key].append((timestamp, value))

    def get_val(key, timestamp):
        if key not in store:
            return ""
        arr = store[key]
        idx = bisect.bisect_right(arr, (timestamp, chr(127))) - 1
        return arr[idx][1] if idx >= 0 else ""

    return set_val, get_val

def random_set():
    """Insert, delete, getRandom in O(1)."""
    import random
    val_to_idx = {}
    vals = []

    def insert(val):
        if val in val_to_idx:
            return False
        val_to_idx[val] = len(vals)
        vals.append(val)
        return True

    def remove(val):
        if val not in val_to_idx:
            return False
        idx = val_to_idx[val]
        last = vals[-1]
        vals[idx] = last
        val_to_idx[last] = idx
        vals.pop()
        del val_to_idx[val]
        return True

    def get_random():
        return random.choice(vals)

    return insert, remove, get_random

def trie_with_prefix_count():
    """Trie with prefix count."""
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.count = 0
            self.end_count = 0

    root = TrieNode()

    def insert(word):
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.count += 1
        node.end_count += 1

    def count_prefix(prefix):
        node = root
        for char in prefix:
            if char not in node.children:
                return 0
            node = node.children[char]
        return node.count

    def count_word(word):
        node = root
        for char in word:
            if char not in node.children:
                return 0
            node = node.children[char]
        return node.end_count

    return insert, count_prefix, count_word

def word_dictionary():
    """Word dictionary with wildcard search."""
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_end = False

    root = TrieNode()

    def add_word(word):
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(word):
        def dfs(node, i):
            if i == len(word):
                return node.is_end
            if word[i] == '.':
                for child in node.children.values():
                    if dfs(child, i + 1):
                        return True
                return False
            if word[i] not in node.children:
                return False
            return dfs(node.children[word[i]], i + 1)
        return dfs(root, 0)

    return add_word, search

# Tests
# HashMap tests
put, get, remove = design_hashmap()
put(1, 1)
put(2, 2)
test1 = get(1)
test2 = get(3)
put(2, 1)
test3 = get(2)
remove(2)
test4 = get(2)

tests = [
    ("hashmap_get", test1, 1),
    ("hashmap_miss", test2, -1),
    ("hashmap_update", test3, 1),
    ("hashmap_remove", test4, -1),
]

# HashSet tests
add, contains, remove_set = design_hashset()
add(1)
add(2)
tests.append(("hashset_contains", contains(1), True))
tests.append(("hashset_miss", contains(3), False))
remove_set(2)
tests.append(("hashset_remove", contains(2), False))

# LRU Cache tests
lru_get, lru_put = lru_cache_impl(2)
lru_put(1, 1)
lru_put(2, 2)
tests.append(("lru_get", lru_get(1), 1))
lru_put(3, 3)  # evicts key 2
tests.append(("lru_evict", lru_get(2), -1))
tests.append(("lru_keep", lru_get(3), 3))

# LFU Cache tests
lfu_get, lfu_put = lfu_cache_impl(2)
lfu_put(1, 1)
lfu_put(2, 2)
tests.append(("lfu_get", lfu_get(1), 1))
lfu_put(3, 3)  # evicts key 2 (least frequently used)
tests.append(("lfu_evict", lfu_get(2), -1))

# Time-based store
set_val, get_val = time_based_key_value()
set_val("foo", "bar", 1)
tests.append(("time_get", get_val("foo", 1), "bar"))
tests.append(("time_get_future", get_val("foo", 3), "bar"))
set_val("foo", "bar2", 4)
tests.append(("time_get_latest", get_val("foo", 4), "bar2"))
tests.append(("time_get_old", get_val("foo", 3), "bar"))

# RandomSet tests
insert, remove_rand, get_random = random_set()
tests.append(("rand_insert", insert(1), True))
tests.append(("rand_dup", insert(1), False))
tests.append(("rand_remove", remove_rand(1), True))
tests.append(("rand_remove_miss", remove_rand(1), False))

# Trie with prefix count
trie_insert, count_prefix, count_word = trie_with_prefix_count()
trie_insert("apple")
trie_insert("app")
trie_insert("application")
tests.append(("trie_prefix", count_prefix("app"), 3))
tests.append(("trie_word", count_word("app"), 1))

# Word dictionary
add_word, search_word = word_dictionary()
add_word("bad")
add_word("dad")
add_word("mad")
tests.append(("dict_search", search_word("pad"), False))
tests.append(("dict_wildcard", search_word(".ad"), True))
tests.append(("dict_exact", search_word("bad"), True))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
