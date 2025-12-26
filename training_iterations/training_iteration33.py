def segment_tree_sum():
    """Segment tree for range sum queries with point updates."""
    tree = []
    n = 0

    def build(arr):
        nonlocal tree, n
        n = len(arr)
        tree = [0] * (2 * n)
        for i in range(n):
            tree[n + i] = arr[i]
        for i in range(n - 1, 0, -1):
            tree[i] = tree[2 * i] + tree[2 * i + 1]

    def update(idx, val):
        idx += n
        tree[idx] = val
        while idx > 1:
            idx //= 2
            tree[idx] = tree[2 * idx] + tree[2 * idx + 1]

    def query(left, right):
        result = 0
        left += n
        right += n + 1
        while left < right:
            if left % 2 == 1:
                result += tree[left]
                left += 1
            if right % 2 == 1:
                right -= 1
                result += tree[right]
            left //= 2
            right //= 2
        return result

    return build, update, query

def fenwick_tree():
    """Binary Indexed Tree for prefix sums."""
    tree = []

    def build(arr):
        nonlocal tree
        n = len(arr)
        tree = [0] * (n + 1)
        for i, val in enumerate(arr):
            add(i, val)

    def add(idx, delta):
        idx += 1
        while idx < len(tree):
            tree[idx] += delta
            idx += idx & (-idx)

    def prefix_sum(idx):
        idx += 1
        result = 0
        while idx > 0:
            result += tree[idx]
            idx -= idx & (-idx)
        return result

    def range_sum(left, right):
        return prefix_sum(right) - (prefix_sum(left - 1) if left > 0 else 0)

    return build, add, prefix_sum, range_sum

def union_find():
    """Union-Find with path compression and union by rank."""
    parent = {}
    rank = {}

    def find(x):
        if x not in parent:
            parent[x] = x
            rank[x] = 0
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True

    def connected(x, y):
        return find(x) == find(y)

    return find, union, connected

def count_range_sum(nums, lower, upper):
    """Counts range sums within [lower, upper] using merge sort."""
    n = len(nums)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + nums[i]

    count = [0]

    def merge_count(arr, left, right):
        if left >= right:
            return
        mid = (left + right) // 2
        merge_count(arr, left, mid)
        merge_count(arr, mid + 1, right)

        # Count valid pairs
        j = k = mid + 1
        for i in range(left, mid + 1):
            while j <= right and arr[j] - arr[i] < lower:
                j += 1
            while k <= right and arr[k] - arr[i] <= upper:
                k += 1
            count[0] += k - j

        # Merge
        arr[left:right + 1] = sorted(arr[left:right + 1])

    merge_count(prefix, 0, n)
    return count[0]

def reverse_pairs(nums):
    """Counts reverse pairs where nums[i] > 2*nums[j] and i < j."""
    count = [0]

    def merge_count(arr):
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = merge_count(arr[:mid])
        right = merge_count(arr[mid:])

        # Count pairs
        j = 0
        for i in range(len(left)):
            while j < len(right) and left[i] > 2 * right[j]:
                j += 1
            count[0] += j

        # Merge
        return sorted(left + right)

    merge_count(nums[:])
    return count[0]

def skyline_divide_conquer(buildings):
    """Skyline using divide and conquer."""
    def divide(buildings):
        if not buildings:
            return []
        if len(buildings) == 1:
            left, right, height = buildings[0]
            return [[left, height], [right, 0]]
        mid = len(buildings) // 2
        left = divide(buildings[:mid])
        right = divide(buildings[mid:])
        return merge(left, right)

    def merge(left, right):
        result = []
        i = j = 0
        h1 = h2 = 0
        while i < len(left) and j < len(right):
            if left[i][0] < right[j][0]:
                x = left[i][0]
                h1 = left[i][1]
                i += 1
            elif left[i][0] > right[j][0]:
                x = right[j][0]
                h2 = right[j][1]
                j += 1
            else:
                x = left[i][0]
                h1 = left[i][1]
                h2 = right[j][1]
                i += 1
                j += 1
            max_h = max(h1, h2)
            if not result or result[-1][1] != max_h:
                result.append([x, max_h])
        while i < len(left):
            if not result or result[-1][1] != left[i][1]:
                result.append(left[i])
            i += 1
        while j < len(right):
            if not result or result[-1][1] != right[j][1]:
                result.append(right[j])
            j += 1
        return result

    return divide(buildings)

def count_of_smaller_after_self(nums):
    """Counts smaller elements to the right using BST."""
    from sortedcontainers import SortedList
    result = []
    sorted_list = SortedList()
    for num in reversed(nums):
        result.append(sorted_list.bisect_left(num))
        sorted_list.add(num)
    return result[::-1]

# Tests
# Segment tree tests
build_st, update_st, query_st = segment_tree_sum()
build_st([1, 3, 5, 7, 9, 11])

tests = [
    ("seg_query", query_st(1, 3), 15),  # 3+5+7
]

update_st(1, 2)  # Change 3 to 2
tests.append(("seg_update", query_st(1, 3), 14))  # 2+5+7

# Fenwick tree tests
build_ft, add_ft, prefix_ft, range_ft = fenwick_tree()
build_ft([1, 3, 5, 7, 9, 11])
tests.append(("fenwick_prefix", prefix_ft(3), 16))  # 1+3+5+7
tests.append(("fenwick_range", range_ft(1, 3), 15))  # 3+5+7

# Union-Find tests
find, union, connected = union_find()
union(1, 2)
union(2, 3)
tests.append(("uf_connected", connected(1, 3), True))
tests.append(("uf_not_connected", connected(1, 4), False))

# Advanced counting tests
tests.append(("count_range", count_range_sum([-2, 5, -1], -2, 2), 3))
tests.append(("reverse_pairs", reverse_pairs([1, 3, 2, 3, 1]), 2))

# Skyline divide and conquer
tests.append(("skyline_dc", skyline_divide_conquer([[2,9,10],[3,7,15],[5,12,12]]),
              [[2,10],[3,15],[7,12],[12,0]]))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
