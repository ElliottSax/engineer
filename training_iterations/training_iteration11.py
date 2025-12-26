def trap_rain_water(height):
    """Calculates trapped rainwater between bars."""
    if not height:
        return 0
    left, right = 0, len(height) - 1
    left_max = right_max = water = 0
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
    return water

def largest_rectangle_histogram(heights):
    """Largest rectangle in histogram."""
    stack = []
    max_area = 0
    for i, h in enumerate(heights + [0]):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    return max_area

def median_two_sorted_arrays(nums1, nums2):
    """Finds median of two sorted arrays in O(log(min(m,n)))."""
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    m, n = len(nums1), len(nums2)
    low, high = 0, m
    while low <= high:
        i = (low + high) // 2
        j = (m + n + 1) // 2 - i
        max_left1 = float('-inf') if i == 0 else nums1[i - 1]
        min_right1 = float('inf') if i == m else nums1[i]
        max_left2 = float('-inf') if j == 0 else nums2[j - 1]
        min_right2 = float('inf') if j == n else nums2[j]
        if max_left1 <= min_right2 and max_left2 <= min_right1:
            if (m + n) % 2 == 0:
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2
            return max(max_left1, max_left2)
        elif max_left1 > min_right2:
            high = i - 1
        else:
            low = i + 1
    return 0

def merge_k_sorted_lists(lists):
    """Merges k sorted linked lists into one."""
    import heapq
    heap = []
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst['val'], i, lst))
    dummy = {'val': 0, 'next': None}
    current = dummy
    while heap:
        val, i, node = heapq.heappop(heap)
        current['next'] = node
        current = current['next']
        if node.get('next'):
            heapq.heappush(heap, (node['next']['val'], i, node['next']))
    return dummy['next']

def decode_ways(s):
    """Number of ways to decode string (1=A, 2=B, ..., 26=Z)."""
    if not s or s[0] == '0':
        return 0
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1
    for i in range(2, n + 1):
        if s[i-1] != '0':
            dp[i] += dp[i-1]
        two_digit = int(s[i-2:i])
        if 10 <= two_digit <= 26:
            dp[i] += dp[i-2]
    return dp[n]

def max_subarray_product(nums):
    """Maximum product of contiguous subarray."""
    if not nums:
        return 0
    max_prod = min_prod = result = nums[0]
    for num in nums[1:]:
        if num < 0:
            max_prod, min_prod = min_prod, max_prod
        max_prod = max(num, max_prod * num)
        min_prod = min(num, min_prod * num)
        result = max(result, max_prod)
    return result

def spiral_matrix(matrix):
    """Returns elements in spiral order."""
    if not matrix:
        return []
    result = []
    top, bottom, left, right = 0, len(matrix) - 1, 0, len(matrix[0]) - 1
    while top <= bottom and left <= right:
        for i in range(left, right + 1):
            result.append(matrix[top][i])
        top += 1
        for i in range(top, bottom + 1):
            result.append(matrix[i][right])
        right -= 1
        if top <= bottom:
            for i in range(right, left - 1, -1):
                result.append(matrix[bottom][i])
            bottom -= 1
        if left <= right:
            for i in range(bottom, top - 1, -1):
                result.append(matrix[i][left])
            left += 1
    return result

def set_matrix_zeroes(matrix):
    """Sets entire row and column to zero if element is zero."""
    m, n = len(matrix), len(matrix[0])
    first_row_zero = any(matrix[0][j] == 0 for j in range(n))
    first_col_zero = any(matrix[i][0] == 0 for i in range(m))
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] == 0:
                matrix[i][0] = matrix[0][j] = 0
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0
    if first_row_zero:
        for j in range(n):
            matrix[0][j] = 0
    if first_col_zero:
        for i in range(m):
            matrix[i][0] = 0
    return matrix

def word_break(s, word_dict):
    """Can string be segmented into dictionary words?"""
    word_set = set(word_dict)
    dp = [False] * (len(s) + 1)
    dp[0] = True
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    return dp[len(s)]

def clone_graph(node, visited=None):
    """Deep clones an undirected graph."""
    if not node:
        return None
    if visited is None:
        visited = {}
    if node['val'] in visited:
        return visited[node['val']]
    clone = {'val': node['val'], 'neighbors': []}
    visited[node['val']] = clone
    for neighbor in node.get('neighbors', []):
        clone['neighbors'].append(clone_graph(neighbor, visited))
    return clone

# Tests
tests = [
    ("trap_water", trap_rain_water([0,1,0,2,1,0,1,3,2,1,2,1]), 6),
    ("largest_rect", largest_rectangle_histogram([2,1,5,6,2,3]), 10),
    ("median_sorted", median_two_sorted_arrays([1,3], [2]), 2.0),
    ("median_even", median_two_sorted_arrays([1,2], [3,4]), 2.5),
    ("decode_ways", decode_ways("226"), 3),
    ("decode_ways_0", decode_ways("06"), 0),
    ("max_product", max_subarray_product([2,3,-2,4]), 6),
    ("max_product_neg", max_subarray_product([-2,0,-1]), 0),
    ("spiral", spiral_matrix([[1,2,3],[4,5,6],[7,8,9]]), [1,2,3,6,9,8,7,4,5]),
    ("word_break_yes", word_break("leetcode", ["leet","code"]), True),
    ("word_break_no", word_break("catsandog", ["cats","dog","sand","and","cat"]), False),
]

# Merge k lists test
l1 = {'val': 1, 'next': {'val': 4, 'next': {'val': 5, 'next': None}}}
l2 = {'val': 1, 'next': {'val': 3, 'next': {'val': 4, 'next': None}}}
l3 = {'val': 2, 'next': {'val': 6, 'next': None}}
merged = merge_k_sorted_lists([l1, l2, l3])
merged_vals = []
while merged:
    merged_vals.append(merged['val'])
    merged = merged.get('next')
tests.append(("merge_k_lists", merged_vals, [1,1,2,3,4,4,5,6]))

# Set matrix zeroes test
mat = [[1,1,1],[1,0,1],[1,1,1]]
set_matrix_zeroes(mat)
tests.append(("set_zeroes", mat[0][1], 0))

# Clone graph test
g = {'val': 1, 'neighbors': []}
g2 = {'val': 2, 'neighbors': [g]}
g['neighbors'].append(g2)
cloned = clone_graph(g)
tests.append(("clone_graph", cloned['val'] == 1 and cloned is not g, True))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
