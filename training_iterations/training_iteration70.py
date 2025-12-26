from collections import OrderedDict, defaultdict
import heapq

def lru_cache_design(capacity):
    """LRU Cache implementation."""
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

def lfu_cache_design(capacity):
    """LFU Cache implementation."""
    cache = {}
    freq = {}
    freq_lists = defaultdict(OrderedDict)
    min_freq = [0]

    def get(key):
        if key not in cache:
            return -1
        _update_freq(key)
        return cache[key]

    def put(key, value):
        if capacity <= 0:
            return
        if key in cache:
            cache[key] = value
            _update_freq(key)
            return
        if len(cache) >= capacity:
            _evict()
        cache[key] = value
        freq[key] = 1
        freq_lists[1][key] = None
        min_freq[0] = 1

    def _update_freq(key):
        f = freq[key]
        freq[key] = f + 1
        del freq_lists[f][key]
        if len(freq_lists[f]) == 0 and min_freq[0] == f:
            min_freq[0] += 1
        freq_lists[f + 1][key] = None

    def _evict():
        key_to_evict = next(iter(freq_lists[min_freq[0]]))
        del freq_lists[min_freq[0]][key_to_evict]
        del cache[key_to_evict]
        del freq[key_to_evict]

    return get, put

def median_finder():
    """Running median from data stream."""
    small = []  # max heap (inverted)
    large = []  # min heap

    def add_num(num):
        heapq.heappush(small, -num)
        heapq.heappush(large, -heapq.heappop(small))
        if len(large) > len(small):
            heapq.heappush(small, -heapq.heappop(large))

    def find_median():
        if len(small) > len(large):
            return -small[0]
        return (-small[0] + large[0]) / 2

    return add_num, find_median

def random_pick_with_weight(weights):
    """Random pick with weights."""
    import random
    prefix = []
    total = 0
    for w in weights:
        total += w
        prefix.append(total)

    def pick():
        target = random.random() * prefix[-1]
        lo, hi = 0, len(prefix) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if prefix[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        return lo

    return pick

def trie_design():
    """Trie (prefix tree) implementation."""
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_end = False

    root = TrieNode()

    def insert(word):
        node = root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_end = True

    def search(word):
        node = root
        for c in word:
            if c not in node.children:
                return False
            node = node.children[c]
        return node.is_end

    def starts_with(prefix):
        node = root
        for c in prefix:
            if c not in node.children:
                return False
            node = node.children[c]
        return True

    return insert, search, starts_with

def skip_iterator_design(it):
    """Skip iterator."""
    skip_map = defaultdict(int)
    next_elem = [None]
    has_next_elem = [False]

    def advance():
        while True:
            if not it:
                has_next_elem[0] = False
                return
            elem = it.pop(0)
            if skip_map[elem] > 0:
                skip_map[elem] -= 1
            else:
                next_elem[0] = elem
                has_next_elem[0] = True
                return

    def has_next():
        return has_next_elem[0]

    def next_val():
        if not has_next_elem[0]:
            return None
        result = next_elem[0]
        advance()
        return result

    def skip(num):
        if has_next_elem[0] and next_elem[0] == num:
            advance()
        else:
            skip_map[num] += 1

    advance()
    return has_next, next_val, skip

def snapshot_array_design(length):
    """Snapshot array."""
    data = defaultdict(list)  # index -> [(snap_id, val)]
    snap_id = [0]

    def set_val(index, val):
        if data[index] and data[index][-1][0] == snap_id[0]:
            data[index][-1] = (snap_id[0], val)
        else:
            data[index].append((snap_id[0], val))

    def snap():
        snap_id[0] += 1
        return snap_id[0] - 1

    def get_val(index, snap):
        history = data[index]
        if not history:
            return 0
        lo, hi = 0, len(history) - 1
        result = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if history[mid][0] <= snap:
                result = history[mid][1]
                lo = mid + 1
            else:
                hi = mid - 1
        return result

    return set_val, snap, get_val

def time_map_design():
    """Time-based key-value store."""
    from collections import defaultdict
    data = defaultdict(list)

    def set_val(key, value, timestamp):
        data[key].append((timestamp, value))

    def get_val(key, timestamp):
        if key not in data:
            return ""
        vals = data[key]
        lo, hi = 0, len(vals) - 1
        result = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            if vals[mid][0] <= timestamp:
                result = vals[mid][1]
                lo = mid + 1
            else:
                hi = mid - 1
        return result

    return set_val, get_val

def stock_price_design():
    """Stock price tracker."""
    prices = {}
    max_heap = []
    min_heap = []
    current_timestamp = [0]

    def update(timestamp, price):
        prices[timestamp] = price
        current_timestamp[0] = max(current_timestamp[0], timestamp)
        heapq.heappush(max_heap, (-price, timestamp))
        heapq.heappush(min_heap, (price, timestamp))

    def current():
        return prices[current_timestamp[0]]

    def maximum():
        while max_heap:
            price, ts = max_heap[0]
            if prices[ts] == -price:
                return -price
            heapq.heappop(max_heap)
        return -1

    def minimum():
        while min_heap:
            price, ts = min_heap[0]
            if prices[ts] == price:
                return price
            heapq.heappop(min_heap)
        return -1

    return update, current, maximum, minimum

# Tests
tests = []

# LRU Cache
get_lru, put_lru = lru_cache_design(2)
put_lru(1, 1)
put_lru(2, 2)
tests.append(("lru_get1", get_lru(1), 1))
put_lru(3, 3)
tests.append(("lru_get2", get_lru(2), -1))

# LFU Cache
get_lfu, put_lfu = lfu_cache_design(2)
put_lfu(1, 1)
put_lfu(2, 2)
tests.append(("lfu_get1", get_lfu(1), 1))
put_lfu(3, 3)
tests.append(("lfu_get2", get_lfu(2), -1))

# Median Finder
add_num, find_med = median_finder()
add_num(1)
add_num(2)
tests.append(("median_1", find_med(), 1.5))
add_num(3)
tests.append(("median_2", find_med(), 2))

# Trie
insert_t, search_t, starts_t = trie_design()
insert_t("apple")
tests.append(("trie_search", search_t("apple"), True))
tests.append(("trie_search_no", search_t("app"), False))
tests.append(("trie_starts", starts_t("app"), True))

# Snapshot Array
set_sa, snap_sa, get_sa = snapshot_array_design(3)
set_sa(0, 5)
tests.append(("snap_0", snap_sa(), 0))
set_sa(0, 6)
tests.append(("get_snap", get_sa(0, 0), 5))

# Time Map
set_tm, get_tm = time_map_design()
set_tm("foo", "bar", 1)
tests.append(("time_get1", get_tm("foo", 1), "bar"))
tests.append(("time_get3", get_tm("foo", 3), "bar"))
set_tm("foo", "bar2", 4)
tests.append(("time_get4", get_tm("foo", 4), "bar2"))

# Stock Price
update_sp, current_sp, max_sp, min_sp = stock_price_design()
update_sp(1, 10)
update_sp(2, 5)
tests.append(("stock_current", current_sp(), 5))
tests.append(("stock_max", max_sp(), 10))
update_sp(1, 3)
tests.append(("stock_max_2", max_sp(), 5))
update_sp(4, 2)
tests.append(("stock_min", min_sp(), 2))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
