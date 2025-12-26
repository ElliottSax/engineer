# EXTREME: Treaps, AVL Trees, and Self-Balancing Structures

import random

# HARD: AVL Tree
class AVLNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.height = 1

class AVLTree:
    def __init__(self):
        self.root = None

    def height(self, node):
        return node.height if node else 0

    def balance(self, node):
        return self.height(node.left) - self.height(node.right) if node else 0

    def update_height(self, node):
        node.height = 1 + max(self.height(node.left), self.height(node.right))

    def rotate_right(self, y):
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        self.update_height(y)
        self.update_height(x)
        return x

    def rotate_left(self, x):
        y = x.right
        T2 = y.left
        y.left = x
        x.right = T2
        self.update_height(x)
        self.update_height(y)
        return y

    def insert(self, val):
        self.root = self._insert(self.root, val)

    def _insert(self, node, val):
        if not node:
            return AVLNode(val)

        if val < node.val:
            node.left = self._insert(node.left, val)
        else:
            node.right = self._insert(node.right, val)

        self.update_height(node)
        balance = self.balance(node)

        # Left Left
        if balance > 1 and val < node.left.val:
            return self.rotate_right(node)
        # Right Right
        if balance < -1 and val > node.right.val:
            return self.rotate_left(node)
        # Left Right
        if balance > 1 and val > node.left.val:
            node.left = self.rotate_left(node.left)
            return self.rotate_right(node)
        # Right Left
        if balance < -1 and val < node.right.val:
            node.right = self.rotate_right(node.right)
            return self.rotate_left(node)

        return node

    def inorder(self):
        result = []
        self._inorder(self.root, result)
        return result

    def _inorder(self, node, result):
        if node:
            self._inorder(node.left, result)
            result.append(node.val)
            self._inorder(node.right, result)

    def get_height(self):
        return self.height(self.root)

# HARD: Treap (Tree + Heap)
class TreapNode:
    def __init__(self, val, priority=None):
        self.val = val
        self.priority = priority if priority is not None else random.random()
        self.left = None
        self.right = None
        self.size = 1

class Treap:
    def __init__(self):
        self.root = None

    def _size(self, node):
        return node.size if node else 0

    def _update(self, node):
        if node:
            node.size = 1 + self._size(node.left) + self._size(node.right)

    def _split(self, node, val):
        if not node:
            return None, None
        if node.val <= val:
            left, right = self._split(node.right, val)
            node.right = left
            self._update(node)
            return node, right
        else:
            left, right = self._split(node.left, val)
            node.left = right
            self._update(node)
            return left, node

    def _merge(self, left, right):
        if not left or not right:
            return left or right
        if left.priority > right.priority:
            left.right = self._merge(left.right, right)
            self._update(left)
            return left
        else:
            right.left = self._merge(left, right.left)
            self._update(right)
            return right

    def insert(self, val):
        left, right = self._split(self.root, val)
        self.root = self._merge(self._merge(left, TreapNode(val)), right)

    def delete(self, val):
        left, temp = self._split(self.root, val - 1)
        _, right = self._split(temp, val)
        self.root = self._merge(left, right)

    def contains(self, val):
        node = self.root
        while node:
            if val == node.val:
                return True
            elif val < node.val:
                node = node.left
            else:
                node = node.right
        return False

    def kth_element(self, k):
        """Find k-th smallest element (1-indexed)."""
        node = self.root
        while node:
            left_size = self._size(node.left)
            if k == left_size + 1:
                return node.val
            elif k <= left_size:
                node = node.left
            else:
                k -= left_size + 1
                node = node.right
        return None

    def size(self):
        return self._size(self.root)

# HARD: Order Statistics Tree (using Treap)
class OrderStatisticsTree:
    def __init__(self):
        self.treap = Treap()

    def insert(self, val):
        self.treap.insert(val)

    def delete(self, val):
        self.treap.delete(val)

    def rank(self, val):
        """Number of elements less than val."""
        count = 0
        node = self.treap.root
        while node:
            if val <= node.val:
                node = node.left
            else:
                count += 1 + (node.left.size if node.left else 0)
                node = node.right
        return count

    def select(self, k):
        """Select k-th smallest element."""
        return self.treap.kth_element(k)

# HARD: Interval Tree
class IntervalNode:
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.max = high
        self.left = None
        self.right = None

class IntervalTree:
    def __init__(self):
        self.root = None

    def insert(self, low, high):
        self.root = self._insert(self.root, low, high)

    def _insert(self, node, low, high):
        if not node:
            return IntervalNode(low, high)

        if low < node.low:
            node.left = self._insert(node.left, low, high)
        else:
            node.right = self._insert(node.right, low, high)

        node.max = max(node.max, high)
        return node

    def overlap(self, low, high):
        """Find any overlapping interval."""
        return self._overlap(self.root, low, high)

    def _overlap(self, node, low, high):
        if not node:
            return None

        if node.low <= high and low <= node.high:
            return (node.low, node.high)

        if node.left and node.left.max >= low:
            result = self._overlap(node.left, low, high)
            if result:
                return result

        return self._overlap(node.right, low, high)

    def all_overlaps(self, low, high):
        """Find all overlapping intervals."""
        result = []
        self._all_overlaps(self.root, low, high, result)
        return result

    def _all_overlaps(self, node, low, high, result):
        if not node:
            return

        if node.low <= high and low <= node.high:
            result.append((node.low, node.high))

        if node.left and node.left.max >= low:
            self._all_overlaps(node.left, low, high, result)

        self._all_overlaps(node.right, low, high, result)

# Tests
tests = []

# AVL Tree
random.seed(42)
avl = AVLTree()
for val in [10, 20, 30, 40, 50, 25]:
    avl.insert(val)
tests.append(("avl_sorted", avl.inorder(), [10, 20, 25, 30, 40, 50]))
tests.append(("avl_balanced", avl.get_height() <= 3, True))

# Treap
random.seed(42)
treap = Treap()
for val in [5, 3, 7, 1, 4, 6, 8]:
    treap.insert(val)
tests.append(("treap_contains", treap.contains(4), True))
tests.append(("treap_kth", treap.kth_element(3), 4))  # 3rd smallest is 4
tests.append(("treap_size", treap.size(), 7))
treap.delete(4)
tests.append(("treap_deleted", treap.contains(4), False))

# Order Statistics Tree
random.seed(42)
ost = OrderStatisticsTree()
for val in [5, 3, 7, 1, 4, 6, 8]:
    ost.insert(val)
tests.append(("ost_rank", ost.rank(5), 3))  # 1, 3, 4 are less than 5
tests.append(("ost_select", ost.select(4), 5))  # 4th smallest is 5

# Interval Tree
it = IntervalTree()
intervals = [(15, 20), (10, 30), (17, 19), (5, 20), (12, 15), (30, 40)]
for low, high in intervals:
    it.insert(low, high)
tests.append(("interval_overlap", it.overlap(14, 16) is not None, True))
tests.append(("interval_no_overlap", it.overlap(21, 23) is not None, True))  # (10,30) overlaps
overlaps = it.all_overlaps(14, 16)
tests.append(("interval_count", len(overlaps) >= 3, True))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
