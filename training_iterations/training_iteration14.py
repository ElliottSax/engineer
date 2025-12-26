def find_median_data_stream():
    """MedianFinder using two heaps."""
    import heapq
    small = []  # max heap (negated)
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

def count_smaller_right(nums):
    """Counts smaller elements to the right of each element."""
    def merge_count(arr):
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = merge_count(arr[:mid])
        right = merge_count(arr[mid:])
        result = []
        i = j = 0
        right_count = 0
        while i < len(left) or j < len(right):
            if j >= len(right) or (i < len(left) and left[i][1] <= right[j][1]):
                counts[left[i][0]] += right_count
                result.append(left[i])
                i += 1
            else:
                right_count += 1
                result.append(right[j])
                j += 1
        return result

    n = len(nums)
    counts = [0] * n
    indexed = [(i, nums[i]) for i in range(n)]
    merge_count(indexed)
    return counts

def skyline(buildings):
    """Returns skyline formed by buildings."""
    import heapq
    events = []
    for left, right, height in buildings:
        events.append((left, -height, right))
        events.append((right, 0, 0))
    events.sort()
    result = []
    heap = [(0, float('inf'))]
    for x, neg_height, right in events:
        while heap[0][1] <= x:
            heapq.heappop(heap)
        if neg_height:
            heapq.heappush(heap, (neg_height, right))
        max_height = -heap[0][0]
        if not result or result[-1][1] != max_height:
            result.append([x, max_height])
    return result

def reverse_nodes_k_group(head, k):
    """Reverses nodes in k-group."""
    def reverse_k(head, k):
        count = 0
        node = head
        while node and count < k:
            node = node.get('next')
            count += 1
        if count < k:
            return head
        prev = reverse_k(node, k)
        while count > 0:
            next_node = head.get('next')
            head['next'] = prev
            prev = head
            head = next_node
            count -= 1
        return prev
    return reverse_k(head, k)

def first_missing_positive(nums):
    """Finds first missing positive integer in O(n) time, O(1) space."""
    n = len(nums)
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    return n + 1

def basic_calculator(s):
    """Evaluates expression with +, -, (, )."""
    stack = []
    num = 0
    sign = 1
    result = 0
    for char in s:
        if char.isdigit():
            num = num * 10 + int(char)
        elif char == '+':
            result += sign * num
            num = 0
            sign = 1
        elif char == '-':
            result += sign * num
            num = 0
            sign = -1
        elif char == '(':
            stack.append(result)
            stack.append(sign)
            result = 0
            sign = 1
        elif char == ')':
            result += sign * num
            num = 0
            result *= stack.pop()
            result += stack.pop()
    result += sign * num
    return result

def find_duplicate_number(nums):
    """Finds duplicate in [1,n] array using Floyd's cycle detection."""
    slow = fast = nums[0]
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    return slow

def serialize_deserialize_bst(root):
    """Serializes and deserializes BST."""
    def serialize(node):
        if not node:
            return []
        return [node['val']] + serialize(node.get('left')) + serialize(node.get('right'))

    def deserialize(data, min_val=float('-inf'), max_val=float('inf')):
        if not data or data[0] < min_val or data[0] > max_val:
            return None
        val = data.pop(0)
        node = {'val': val, 'left': None, 'right': None}
        node['left'] = deserialize(data, min_val, val)
        node['right'] = deserialize(data, val, max_val)
        return node

    data = serialize(root)
    return deserialize(data)

def max_points_on_line(points):
    """Maximum points on same line."""
    from collections import defaultdict
    from math import gcd
    if len(points) <= 2:
        return len(points)

    def get_slope(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if dx == 0:
            return (0, 1)
        if dy == 0:
            return (1, 0)
        g = gcd(abs(dx), abs(dy))
        dx, dy = dx // g, dy // g
        if dx < 0:
            dx, dy = -dx, -dy
        return (dx, dy)

    max_points = 0
    for i in range(len(points)):
        slopes = defaultdict(int)
        same = 1
        for j in range(i + 1, len(points)):
            if points[i] == points[j]:
                same += 1
            else:
                slope = get_slope(points[i], points[j])
                slopes[slope] += 1
        local_max = same + max(slopes.values(), default=0)
        max_points = max(max_points, local_max)
    return max_points

# Tests
add_num, find_median = find_median_data_stream()
add_num(1); add_num(2)
median_test_1 = find_median()
add_num(3)
median_test_2 = find_median()

tests = [
    ("median_stream_1", median_test_1, 1.5),
    ("median_stream_2", median_test_2, 2),
    ("count_smaller", count_smaller_right([5,2,6,1]), [2,1,1,0]),
    ("skyline", skyline([[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]]),
     [[2,10],[3,15],[7,12],[12,0],[15,10],[20,8],[24,0]]),
    ("first_missing_pos", first_missing_positive([3,4,-1,1]), 2),
    ("first_missing_pos_2", first_missing_positive([1,2,0]), 3),
    ("calculator", basic_calculator("(1+(4+5+2)-3)+(6+8)"), 23),
    ("calculator_2", basic_calculator(" 2-1 + 2 "), 3),
    ("find_dup", find_duplicate_number([1,3,4,2,2]), 2),
    ("max_points", max_points_on_line([[1,1],[2,2],[3,3]]), 3),
    ("max_points_2", max_points_on_line([[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]), 4),
]

# Reverse k-group test
head = {'val': 1, 'next': {'val': 2, 'next': {'val': 3, 'next': {'val': 4, 'next': {'val': 5, 'next': None}}}}}
reversed_head = reverse_nodes_k_group(head, 2)
vals = []
while reversed_head:
    vals.append(reversed_head['val'])
    reversed_head = reversed_head.get('next')
tests.append(("reverse_k_group", vals, [2,1,4,3,5]))

# BST serialize/deserialize test
bst = {'val': 5, 'left': {'val': 3, 'left': None, 'right': None}, 'right': {'val': 7, 'left': None, 'right': None}}
restored = serialize_deserialize_bst(bst)
tests.append(("bst_serialize", restored['val'] == 5 and restored['left']['val'] == 3, True))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
