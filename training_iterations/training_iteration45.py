def find_132_pattern(nums):
    """Finds 132 pattern: i < j < k and nums[i] < nums[k] < nums[j]."""
    n = len(nums)
    if n < 3:
        return False
    stack = []
    third = float('-inf')  # nums[k] candidate
    for i in range(n - 1, -1, -1):
        if nums[i] < third:
            return True
        while stack and stack[-1] < nums[i]:
            third = stack.pop()
        stack.append(nums[i])
    return False

def next_greater_element_ii(nums):
    """Next greater element in circular array."""
    n = len(nums)
    result = [-1] * n
    stack = []
    for i in range(2 * n):
        while stack and nums[stack[-1]] < nums[i % n]:
            result[stack.pop()] = nums[i % n]
        if i < n:
            stack.append(i)
    return result

def next_greater_node_linked_list(head):
    """Next greater value for each node in linked list."""
    values = []
    while head:
        values.append(head['val'])
        head = head.get('next')

    result = [0] * len(values)
    stack = []
    for i, val in enumerate(values):
        while stack and values[stack[-1]] < val:
            result[stack.pop()] = val
        stack.append(i)
    return result

def trapping_rain_water_stack(height):
    """Trapping rain water using monotonic stack."""
    stack = []
    water = 0
    for i, h in enumerate(height):
        while stack and height[stack[-1]] < h:
            bottom = stack.pop()
            if not stack:
                break
            width = i - stack[-1] - 1
            bounded_height = min(h, height[stack[-1]]) - height[bottom]
            water += width * bounded_height
        stack.append(i)
    return water

def shortest_unsorted_subarray(nums):
    """Length of shortest subarray to sort to make whole array sorted."""
    n = len(nums)
    stack = []
    left = n
    for i in range(n):
        while stack and nums[stack[-1]] > nums[i]:
            left = min(left, stack.pop())
        stack.append(i)

    stack = []
    right = 0
    for i in range(n - 1, -1, -1):
        while stack and nums[stack[-1]] < nums[i]:
            right = max(right, stack.pop())
        stack.append(i)

    return right - left + 1 if right > left else 0

def max_width_ramp(nums):
    """Maximum j - i where nums[i] <= nums[j]."""
    n = len(nums)
    stack = []
    for i in range(n):
        if not stack or nums[i] < nums[stack[-1]]:
            stack.append(i)

    max_width = 0
    for j in range(n - 1, -1, -1):
        while stack and nums[j] >= nums[stack[-1]]:
            max_width = max(max_width, j - stack.pop())
    return max_width

def longest_well_performing_interval(hours):
    """Longest subarray where tiring days > non-tiring days."""
    n = len(hours)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + (1 if hours[i] > 8 else -1)

    stack = []
    for i in range(n + 1):
        if not stack or prefix[i] < prefix[stack[-1]]:
            stack.append(i)

    max_len = 0
    for j in range(n, -1, -1):
        while stack and prefix[j] > prefix[stack[-1]]:
            max_len = max(max_len, j - stack.pop())
    return max_len

def online_stock_span_stack():
    """Stock span using stack."""
    stack = []

    def next_price(price):
        span = 1
        while stack and stack[-1][0] <= price:
            span += stack.pop()[1]
        stack.append((price, span))
        return span

    return next_price

def visible_people_in_queue(heights):
    """Number of people each person can see to their right."""
    n = len(heights)
    result = [0] * n
    stack = []

    for i in range(n - 1, -1, -1):
        count = 0
        while stack and heights[i] > stack[-1]:
            stack.pop()
            count += 1
        if stack:
            count += 1
        result[i] = count
        stack.append(heights[i])
    return result

def buildings_with_ocean_view(heights):
    """Buildings with ocean view (to the right)."""
    stack = []
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] <= h:
            stack.pop()
        stack.append(i)
    return stack

# Tests
tests = [
    ("132_pattern", find_132_pattern([3,1,4,2]), True),
    ("132_pattern_no", find_132_pattern([1,2,3,4]), False),
    ("next_greater_ii", next_greater_element_ii([1,2,1]), [2,-1,2]),
    ("trap_water_stack", trapping_rain_water_stack([0,1,0,2,1,0,1,3,2,1,2,1]), 6),
    ("shortest_unsorted", shortest_unsorted_subarray([2,6,4,8,10,9,15]), 5),
    ("shortest_unsorted_sorted", shortest_unsorted_subarray([1,2,3,4]), 0),
    ("max_ramp", max_width_ramp([6,0,8,2,1,5]), 4),
    ("max_ramp_2", max_width_ramp([9,8,1,0,1,9,4,0,4,1]), 7),
    ("well_performing", longest_well_performing_interval([9,9,6,0,6,6,9]), 3),
    ("visible_queue", visible_people_in_queue([10,6,8,5,11,9]), [3,1,2,1,1,0]),
    ("ocean_view", buildings_with_ocean_view([4,2,3,1]), [0, 2, 3]),
]

# Stock span test
next_stock = online_stock_span_stack()
spans = [next_stock(p) for p in [100, 80, 60, 70, 60, 75, 85]]
tests.append(("stock_span", spans, [1, 1, 1, 2, 1, 4, 6]))

# Next greater linked list test
ll = {'val': 2, 'next': {'val': 1, 'next': {'val': 5, 'next': None}}}
tests.append(("next_greater_ll", next_greater_node_linked_list(ll), [5, 5, 0]))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
