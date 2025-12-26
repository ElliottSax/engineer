def min_stack():
    """Stack with O(1) getMin operation."""
    stack = []
    min_stack = []

    def push(val):
        stack.append(val)
        if not min_stack or val <= min_stack[-1]:
            min_stack.append(val)

    def pop():
        val = stack.pop()
        if val == min_stack[-1]:
            min_stack.pop()
        return val

    def top():
        return stack[-1]

    def get_min():
        return min_stack[-1]

    return push, pop, top, get_min

def daily_temperatures(temperatures):
    """Days until warmer temperature."""
    n = len(temperatures)
    result = [0] * n
    stack = []
    for i in range(n):
        while stack and temperatures[i] > temperatures[stack[-1]]:
            idx = stack.pop()
            result[idx] = i - idx
        stack.append(i)
    return result

def next_greater_element(nums1, nums2):
    """Next greater element for each in nums1 from nums2."""
    next_greater = {}
    stack = []
    for num in nums2:
        while stack and stack[-1] < num:
            next_greater[stack.pop()] = num
        stack.append(num)
    return [next_greater.get(num, -1) for num in nums1]

def validate_stack_sequence(pushed, popped):
    """Validates if pushed/popped sequence is valid."""
    stack = []
    j = 0
    for num in pushed:
        stack.append(num)
        while stack and stack[-1] == popped[j]:
            stack.pop()
            j += 1
    return j == len(popped)

def remove_k_digits(num, k):
    """Removes k digits to get smallest number."""
    stack = []
    for digit in num:
        while k and stack and stack[-1] > digit:
            stack.pop()
            k -= 1
        stack.append(digit)
    stack = stack[:-k] if k else stack
    return ''.join(stack).lstrip('0') or '0'

def asteroid_collision(asteroids):
    """Simulates asteroid collision."""
    stack = []
    for a in asteroids:
        while stack and a < 0 < stack[-1]:
            if stack[-1] < -a:
                stack.pop()
                continue
            elif stack[-1] == -a:
                stack.pop()
            break
        else:
            stack.append(a)
    return stack

def remove_duplicate_letters(s):
    """Removes duplicate letters to get smallest lexicographical result."""
    last_idx = {c: i for i, c in enumerate(s)}
    stack = []
    seen = set()
    for i, c in enumerate(s):
        if c in seen:
            continue
        while stack and c < stack[-1] and i < last_idx[stack[-1]]:
            seen.remove(stack.pop())
        stack.append(c)
        seen.add(c)
    return ''.join(stack)

def online_stock_span():
    """Stock span - consecutive days price <= today."""
    stack = []

    def next_price(price):
        span = 1
        while stack and stack[-1][0] <= price:
            span += stack.pop()[1]
        stack.append((price, span))
        return span

    return next_price

def car_fleet(target, position, speed):
    """Number of car fleets reaching target."""
    cars = sorted(zip(position, speed), reverse=True)
    times = [(target - p) / s for p, s in cars]
    fleets = 0
    current_time = 0
    for time in times:
        if time > current_time:
            fleets += 1
            current_time = time
    return fleets

def sum_of_subarray_minimums(arr):
    """Sum of minimums of all subarrays."""
    MOD = 10**9 + 7
    n = len(arr)
    left = [0] * n  # distance to previous smaller
    right = [0] * n  # distance to next smaller or equal
    stack = []

    for i in range(n):
        while stack and arr[stack[-1]] >= arr[i]:
            stack.pop()
        left[i] = i - stack[-1] if stack else i + 1
        stack.append(i)

    stack = []
    for i in range(n - 1, -1, -1):
        while stack and arr[stack[-1]] > arr[i]:
            stack.pop()
        right[i] = stack[-1] - i if stack else n - i
        stack.append(i)

    return sum(arr[i] * left[i] * right[i] for i in range(n)) % MOD

def longest_increasing_path(matrix):
    """Longest increasing path in matrix."""
    if not matrix:
        return 0
    m, n = len(matrix), len(matrix[0])
    memo = {}

    def dfs(r, c):
        if (r, c) in memo:
            return memo[(r, c)]
        length = 1
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and matrix[nr][nc] > matrix[r][c]:
                length = max(length, 1 + dfs(nr, nc))
        memo[(r, c)] = length
        return length

    return max(dfs(i, j) for i in range(m) for j in range(n))

# Tests
push, pop, top, get_min = min_stack()
push(-2); push(0); push(-3)
min_test_1 = get_min()
pop()
min_test_2 = get_min()

next_price = online_stock_span()
span_results = [next_price(p) for p in [100, 80, 60, 70, 60, 75, 85]]

tests = [
    ("min_stack_1", min_test_1, -3),
    ("min_stack_2", min_test_2, -2),
    ("daily_temps", daily_temperatures([73,74,75,71,69,72,76,73]), [1,1,4,2,1,1,0,0]),
    ("next_greater", next_greater_element([4,1,2], [1,3,4,2]), [-1,3,-1]),
    ("validate_stack", validate_stack_sequence([1,2,3,4,5], [4,5,3,2,1]), True),
    ("validate_stack_no", validate_stack_sequence([1,2,3,4,5], [4,3,5,1,2]), False),
    ("remove_k", remove_k_digits("1432219", 3), "1219"),
    ("remove_k_2", remove_k_digits("10200", 1), "200"),
    ("asteroids", asteroid_collision([5,10,-5]), [5,10]),
    ("asteroids_2", asteroid_collision([8,-8]), []),
    ("remove_dup_letters", remove_duplicate_letters("bcabc"), "abc"),
    ("remove_dup_2", remove_duplicate_letters("cbacdcbc"), "acdb"),
    ("stock_span", span_results, [1, 1, 1, 2, 1, 4, 6]),
    ("car_fleet", car_fleet(12, [10,8,0,5,3], [2,4,1,1,3]), 3),
    ("subarray_mins", sum_of_subarray_minimums([3,1,2,4]), 17),
    ("increasing_path", longest_increasing_path([[9,9,4],[6,6,8],[2,1,1]]), 4),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
