# EXTREME: Advanced Data Structures

from collections import defaultdict

# HARD: Segment Tree with Lazy Propagation
class SegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        self._build(arr, 0, 0, self.n - 1)

    def _build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self._build(arr, 2*node+1, start, mid)
            self._build(arr, 2*node+2, mid+1, end)
            self.tree[node] = self.tree[2*node+1] + self.tree[2*node+2]

    def _propagate(self, node, start, end):
        if self.lazy[node] != 0:
            self.tree[node] += self.lazy[node] * (end - start + 1)
            if start != end:
                self.lazy[2*node+1] += self.lazy[node]
                self.lazy[2*node+2] += self.lazy[node]
            self.lazy[node] = 0

    def update_range(self, l, r, val):
        self._update(0, 0, self.n-1, l, r, val)

    def _update(self, node, start, end, l, r, val):
        self._propagate(node, start, end)
        if r < start or end < l:
            return
        if l <= start and end <= r:
            self.lazy[node] += val
            self._propagate(node, start, end)
            return
        mid = (start + end) // 2
        self._update(2*node+1, start, mid, l, r, val)
        self._update(2*node+2, mid+1, end, l, r, val)
        self.tree[node] = self.tree[2*node+1] + self.tree[2*node+2]

    def query(self, l, r):
        return self._query(0, 0, self.n-1, l, r)

    def _query(self, node, start, end, l, r):
        self._propagate(node, start, end)
        if r < start or end < l:
            return 0
        if l <= start and end <= r:
            return self.tree[node]
        mid = (start + end) // 2
        return self._query(2*node+1, start, mid, l, r) + \
               self._query(2*node+2, mid+1, end, l, r)

# HARD: Fenwick Tree (Binary Indexed Tree)
class FenwickTree:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)

    def update(self, i, delta):
        i += 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def prefix_sum(self, i):
        i += 1
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s

    def range_sum(self, l, r):
        return self.prefix_sum(r) - (self.prefix_sum(l-1) if l > 0 else 0)

# HARD: Trie with Autocomplete
class Trie:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = 0

    def insert(self, word):
        node = self
        for c in word:
            if c not in node.children:
                node.children[c] = Trie()
            node = node.children[c]
            node.count += 1
        node.is_end = True

    def search(self, word):
        node = self._find_node(word)
        return node is not None and node.is_end

    def starts_with(self, prefix):
        return self._find_node(prefix) is not None

    def count_prefix(self, prefix):
        node = self._find_node(prefix)
        return node.count if node else 0

    def _find_node(self, prefix):
        node = self
        for c in prefix:
            if c not in node.children:
                return None
            node = node.children[c]
        return node

    def autocomplete(self, prefix, limit=10):
        node = self._find_node(prefix)
        if not node:
            return []
        results = []
        self._dfs(node, prefix, results, limit)
        return results

    def _dfs(self, node, current, results, limit):
        if len(results) >= limit:
            return
        if node.is_end:
            results.append(current)
        for c, child in sorted(node.children.items()):
            self._dfs(child, current + c, results, limit)

# HARD: LRU Cache
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.head = {'prev': None, 'next': None}
        self.tail = {'prev': self.head, 'next': None}
        self.head['next'] = self.tail

    def _remove(self, node):
        node['prev']['next'] = node['next']
        node['next']['prev'] = node['prev']

    def _add_to_front(self, node):
        node['next'] = self.head['next']
        node['prev'] = self.head
        self.head['next']['prev'] = node
        self.head['next'] = node

    def get(self, key):
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)
        self._add_to_front(node)
        return node['val']

    def put(self, key, value):
        if key in self.cache:
            self._remove(self.cache[key])
        node = {'key': key, 'val': value, 'prev': None, 'next': None}
        self.cache[key] = node
        self._add_to_front(node)
        if len(self.cache) > self.capacity:
            lru = self.tail['prev']
            self._remove(lru)
            del self.cache[lru['key']]

# HARD: Skip List
import random

class SkipListNode:
    def __init__(self, val, level):
        self.val = val
        self.forward = [None] * (level + 1)

class SkipList:
    def __init__(self, max_level=16, p=0.5):
        self.max_level = max_level
        self.p = p
        self.level = 0
        self.head = SkipListNode(float('-inf'), max_level)

    def _random_level(self):
        lvl = 0
        while random.random() < self.p and lvl < self.max_level:
            lvl += 1
        return lvl

    def search(self, target):
        current = self.head
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].val < target:
                current = current.forward[i]
        current = current.forward[0]
        return current and current.val == target

    def insert(self, num):
        update = [None] * (self.max_level + 1)
        current = self.head
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].val < num:
                current = current.forward[i]
            update[i] = current
        lvl = self._random_level()
        if lvl > self.level:
            for i in range(self.level + 1, lvl + 1):
                update[i] = self.head
            self.level = lvl
        node = SkipListNode(num, lvl)
        for i in range(lvl + 1):
            node.forward[i] = update[i].forward[i]
            update[i].forward[i] = node

    def delete(self, num):
        update = [None] * (self.max_level + 1)
        current = self.head
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].val < num:
                current = current.forward[i]
            update[i] = current
        current = current.forward[0]
        if current and current.val == num:
            for i in range(self.level + 1):
                if update[i].forward[i] != current:
                    break
                update[i].forward[i] = current.forward[i]
            while self.level > 0 and self.head.forward[self.level] is None:
                self.level -= 1
            return True
        return False

# HARD: Disjoint Set Union with Path Compression and Union by Rank
class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        self.size[px] += self.size[py]
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)

    def get_size(self, x):
        return self.size[self.find(x)]

# Tests
tests = []

# Segment Tree
st = SegmentTree([1, 3, 5, 7, 9, 11])
tests.append(("seg_query", st.query(1, 3), 15))
st.update_range(1, 3, 10)
tests.append(("seg_update", st.query(1, 3), 45))

# Fenwick Tree
ft = FenwickTree(6)
for i, v in enumerate([1, 3, 5, 7, 9, 11]):
    ft.update(i, v)
tests.append(("fenwick", ft.range_sum(1, 3), 15))
ft.update(2, 5)
tests.append(("fenwick_up", ft.range_sum(1, 3), 20))

# Trie
trie = Trie()
for word in ["apple", "app", "application", "apply", "banana"]:
    trie.insert(word)
tests.append(("trie_search", trie.search("app"), True))
tests.append(("trie_prefix", trie.starts_with("app"), True))
tests.append(("trie_count", trie.count_prefix("app"), 4))
tests.append(("trie_auto", trie.autocomplete("app", 3), ["app", "apple", "application"]))

# LRU Cache
lru = LRUCache(2)
lru.put(1, 1)
lru.put(2, 2)
tests.append(("lru_get", lru.get(1), 1))
lru.put(3, 3)
tests.append(("lru_evict", lru.get(2), -1))

# Skip List
random.seed(42)
sl = SkipList()
sl.insert(3)
sl.insert(1)
sl.insert(2)
tests.append(("skip_search", sl.search(2), True))
tests.append(("skip_not", sl.search(4), False))
sl.delete(2)
tests.append(("skip_del", sl.search(2), False))

# DSU
dsu = DSU(5)
dsu.union(0, 1)
dsu.union(2, 3)
tests.append(("dsu_conn", dsu.connected(0, 1), True))
tests.append(("dsu_not", dsu.connected(0, 2), False))
dsu.union(1, 3)
tests.append(("dsu_merge", dsu.connected(0, 2), True))
tests.append(("dsu_size", dsu.get_size(0), 4))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
