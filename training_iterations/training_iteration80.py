from collections import deque, defaultdict
import heapq

def serialize_deserialize_tree():
    """Serialize and deserialize binary tree."""
    def serialize(root):
        if not root:
            return "null"
        return f"{root['val']},{serialize(root.get('left'))},{serialize(root.get('right'))}"

    def deserialize(data):
        def helper(nodes):
            val = next(nodes)
            if val == "null":
                return None
            node = {'val': int(val), 'left': None, 'right': None}
            node['left'] = helper(nodes)
            node['right'] = helper(nodes)
            return node
        return helper(iter(data.split(',')))

    return serialize, deserialize

def serialize_n_ary_tree():
    """Serialize and deserialize N-ary tree."""
    def serialize(root):
        if not root:
            return ""
        result = []
        def dfs(node):
            result.append(str(node['val']))
            result.append(str(len(node.get('children', []))))
            for child in node.get('children', []):
                dfs(child)
        dfs(root)
        return ','.join(result)

    def deserialize(data):
        if not data:
            return None
        nodes = iter(data.split(','))
        def dfs():
            val = int(next(nodes))
            num_children = int(next(nodes))
            node = {'val': val, 'children': []}
            for _ in range(num_children):
                node['children'].append(dfs())
            return node
        return dfs()

    return serialize, deserialize

def count_complete_tree_nodes(root):
    """Count nodes in complete binary tree in O(log^2 n)."""
    if not root:
        return 0

    def get_depth(node):
        depth = 0
        while node.get('left'):
            depth += 1
            node = node['left']
        return depth

    def exists(idx, depth, node):
        left, right = 0, 2**depth - 1
        for _ in range(depth):
            mid = (left + right) // 2
            if idx <= mid:
                node = node.get('left')
                right = mid
            else:
                node = node.get('right')
                left = mid + 1
        return node is not None

    depth = get_depth(root)
    if depth == 0:
        return 1

    left, right = 0, 2**depth - 1
    while left < right:
        mid = (left + right + 1) // 2
        if exists(mid, depth, root):
            left = mid
        else:
            right = mid - 1

    return 2**depth + left

def boundary_of_binary_tree(root):
    """Get boundary of binary tree."""
    if not root:
        return []

    def is_leaf(node):
        return not node.get('left') and not node.get('right')

    result = []
    if not is_leaf(root):
        result.append(root['val'])

    # Left boundary
    node = root.get('left')
    while node and not is_leaf(node):
        result.append(node['val'])
        node = node.get('left') or node.get('right')

    # Leaves
    def add_leaves(node):
        if not node:
            return
        if is_leaf(node):
            result.append(node['val'])
        add_leaves(node.get('left'))
        add_leaves(node.get('right'))

    add_leaves(root)

    # Right boundary (reversed)
    right_boundary = []
    node = root.get('right')
    while node and not is_leaf(node):
        right_boundary.append(node['val'])
        node = node.get('right') or node.get('left')

    result.extend(reversed(right_boundary))
    return result

def find_median_from_stream():
    """Median from data stream."""
    small = []  # max heap
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

def sliding_window_median(nums, k):
    """Median of each sliding window."""
    import bisect
    window = sorted(nums[:k])
    result = []

    def get_median():
        if k % 2:
            return float(window[k // 2])
        return (window[k // 2 - 1] + window[k // 2]) / 2

    result.append(get_median())

    for i in range(k, len(nums)):
        window.pop(bisect.bisect_left(window, nums[i - k]))
        bisect.insort(window, nums[i])
        result.append(get_median())

    return result

def range_sum_query_2d_mutable():
    """2D range sum with updates."""
    matrix = None
    m = n = 0
    bit = None

    def init(mat):
        nonlocal matrix, m, n, bit
        if not mat or not mat[0]:
            return
        m, n = len(mat), len(mat[0])
        matrix = [[0] * n for _ in range(m)]
        bit = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                update(i, j, mat[i][j])

    def update(row, col, val):
        diff = val - matrix[row][col]
        matrix[row][col] = val
        i = row + 1
        while i <= m:
            j = col + 1
            while j <= n:
                bit[i][j] += diff
                j += j & (-j)
            i += i & (-i)

    def query(row, col):
        result = 0
        i = row + 1
        while i > 0:
            j = col + 1
            while j > 0:
                result += bit[i][j]
                j -= j & (-j)
            i -= i & (-i)
        return result

    def sum_region(r1, c1, r2, c2):
        return query(r2, c2) - query(r1 - 1, c2) - query(r2, c1 - 1) + query(r1 - 1, c1 - 1)

    return init, update, sum_region

def text_justification(words, max_width):
    """Full justify text."""
    result = []
    line = []
    line_len = 0

    for word in words:
        if line_len + len(word) + len(line) > max_width:
            # Justify current line
            spaces = max_width - line_len
            gaps = len(line) - 1
            if gaps == 0:
                result.append(line[0] + ' ' * spaces)
            else:
                space_per_gap = spaces // gaps
                extra = spaces % gaps
                justified = ''
                for i, w in enumerate(line[:-1]):
                    justified += w + ' ' * (space_per_gap + (1 if i < extra else 0))
                justified += line[-1]
                result.append(justified)
            line = []
            line_len = 0

        line.append(word)
        line_len += len(word)

    # Last line - left justify
    last_line = ' '.join(line)
    result.append(last_line + ' ' * (max_width - len(last_line)))

    return result

# Tests
tests = []

# Serialize/deserialize binary tree
ser, deser = serialize_deserialize_tree()
tree = {'val': 1, 'left': {'val': 2}, 'right': {'val': 3, 'left': {'val': 4}, 'right': {'val': 5}}}
serialized = ser(tree)
deserialized = deser(serialized)
tests.append(("serialize_bt", deserialized['val'], 1))

# Serialize/deserialize N-ary tree
ser_n, deser_n = serialize_n_ary_tree()
n_tree = {'val': 1, 'children': [{'val': 3, 'children': [{'val': 5, 'children': []}, {'val': 6, 'children': []}]},
                                  {'val': 2, 'children': []}, {'val': 4, 'children': []}]}
serialized_n = ser_n(n_tree)
deserialized_n = deser_n(serialized_n)
tests.append(("serialize_nt", deserialized_n['val'], 1))

# Median from stream
add_num, find_med = find_median_from_stream()
add_num(1)
add_num(2)
tests.append(("median_1", find_med(), 1.5))
add_num(3)
tests.append(("median_2", find_med(), 2))

# Sliding window median
tests.append(("sliding_med", sliding_window_median([1,3,-1,-3,5,3,6,7], 3), [1.0,-1.0,-1.0,3.0,5.0,6.0]))

# 2D range sum
init_rs, update_rs, sum_rs = range_sum_query_2d_mutable()
init_rs([[3,0,1,4,2],[5,6,3,2,1],[1,2,0,1,5],[4,1,0,1,7],[1,0,3,0,5]])
tests.append(("range_sum", sum_rs(2, 1, 4, 3), 8))
update_rs(3, 2, 2)
tests.append(("range_sum_up", sum_rs(2, 1, 4, 3), 10))

# Text justification
tests.append(("justify", text_justification(["This","is","an","example","of","text","justification."], 16),
              ["This    is    an", "example  of text", "justification.  "]))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
