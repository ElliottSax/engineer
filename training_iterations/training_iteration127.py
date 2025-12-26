#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ULTRA TRAINING ITERATION 127                              ║
║               Stack & Queue - Advanced Applications                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from collections import deque

# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION 1: Monotonic Stack - Next Greater Element
# ═══════════════════════════════════════════════════════════════════════════════
def next_greater_element(nums):
    """For each element, find next greater element to its right."""
    n = len(nums)
    result = [-1] * n
    stack = []  # Stack of indices

    for i in range(n):
        while stack and nums[i] > nums[stack[-1]]:
            result[stack.pop()] = nums[i]
        stack.append(i)

    return result

def next_smaller_element(nums):
    """For each element, find next smaller element to its right."""
    n = len(nums)
    result = [-1] * n
    stack = []

    for i in range(n):
        while stack and nums[i] < nums[stack[-1]]:
            result[stack.pop()] = nums[i]
        stack.append(i)

    return result

# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION 2: Largest Rectangle in Histogram
# ═══════════════════════════════════════════════════════════════════════════════
def largest_rectangle_histogram(heights):
    """Find largest rectangular area in histogram."""
    stack = []
    max_area = 0
    index = 0

    while index < len(heights):
        if not stack or heights[index] >= heights[stack[-1]]:
            stack.append(index)
            index += 1
        else:
            top = stack.pop()
            width = index if not stack else index - stack[-1] - 1
            max_area = max(max_area, heights[top] * width)

    while stack:
        top = stack.pop()
        width = index if not stack else index - stack[-1] - 1
        max_area = max(max_area, heights[top] * width)

    return max_area

# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION 3: Trapping Rain Water
# ═══════════════════════════════════════════════════════════════════════════════
def trap_rain_water(heights):
    """Calculate trapped rainwater using stack."""
    stack = []
    water = 0

    for i, h in enumerate(heights):
        while stack and h > heights[stack[-1]]:
            bottom = stack.pop()
            if not stack:
                break
            width = i - stack[-1] - 1
            bounded_height = min(h, heights[stack[-1]]) - heights[bottom]
            water += width * bounded_height
        stack.append(i)

    return water

# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION 4: Valid Parentheses & Min Add to Make Valid
# ═══════════════════════════════════════════════════════════════════════════════
def is_valid_parentheses(s):
    """Check if parentheses are balanced."""
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}

    for char in s:
        if char in '([{':
            stack.append(char)
        elif char in ')]}':
            if not stack or stack.pop() != pairs[char]:
                return False

    return len(stack) == 0

def min_add_to_make_valid(s):
    """Minimum additions to make parentheses valid."""
    open_needed = 0
    close_needed = 0

    for char in s:
        if char == '(':
            close_needed += 1
        elif char == ')':
            if close_needed > 0:
                close_needed -= 1
            else:
                open_needed += 1

    return open_needed + close_needed

# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION 5: Sliding Window Maximum
# ═══════════════════════════════════════════════════════════════════════════════
def sliding_window_max(nums, k):
    """Find maximum in each sliding window of size k."""
    if not nums or k == 0:
        return []

    result = []
    dq = deque()  # Store indices, front is max

    for i in range(len(nums)):
        # Remove elements outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # Remove smaller elements (they can never be max)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        dq.append(i)

        if i >= k - 1:
            result.append(nums[dq[0]])

    return result

# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION 6: Min Stack
# ═══════════════════════════════════════════════════════════════════════════════
class MinStack:
    """Stack that supports O(1) getMin operation."""

    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self):
        if self.stack:
            val = self.stack.pop()
            if val == self.min_stack[-1]:
                self.min_stack.pop()
            return val
        return None

    def top(self):
        return self.stack[-1] if self.stack else None

    def get_min(self):
        return self.min_stack[-1] if self.min_stack else None

# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION 7: Daily Temperatures
# ═══════════════════════════════════════════════════════════════════════════════
def daily_temperatures(temps):
    """Days until warmer temperature."""
    n = len(temps)
    result = [0] * n
    stack = []  # Stack of indices

    for i in range(n):
        while stack and temps[i] > temps[stack[-1]]:
            prev = stack.pop()
            result[prev] = i - prev
        stack.append(i)

    return result

# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION 8: Remove K Digits
# ═══════════════════════════════════════════════════════════════════════════════
def remove_k_digits(num, k):
    """Remove k digits to make smallest number."""
    stack = []

    for digit in num:
        while k > 0 and stack and stack[-1] > digit:
            stack.pop()
            k -= 1
        stack.append(digit)

    # Remove remaining k from end
    stack = stack[:-k] if k else stack

    # Remove leading zeros and build result
    result = ''.join(stack).lstrip('0')
    return result if result else '0'

# ═══════════════════════════════════════════════════════════════════════════════
#                              TEST SUITE
# ═══════════════════════════════════════════════════════════════════════════════
def run_tests():
    tests = []

    # Test 1: Next greater element
    tests.append(("nge", next_greater_element([4, 5, 2, 25]), [5, 25, 25, -1]))
    tests.append(("nse", next_smaller_element([4, 5, 2, 25]), [2, 2, -1, -1]))

    # Test 2: Largest rectangle
    tests.append(("histogram", largest_rectangle_histogram([2, 1, 5, 6, 2, 3]), 10))

    # Test 3: Trap water
    tests.append(("trap", trap_rain_water([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]), 6))

    # Test 4: Parentheses
    tests.append(("valid_paren", is_valid_parentheses("()[]{}"), True))
    tests.append(("invalid_paren", is_valid_parentheses("([)]"), False))
    tests.append(("min_add", min_add_to_make_valid("())"), 1))

    # Test 5: Sliding window max
    tests.append(("window_max", sliding_window_max([1, 3, -1, -3, 5, 3, 6, 7], 3),
                  [3, 3, 5, 5, 6, 7]))

    # Test 6: Min Stack
    ms = MinStack()
    ms.push(-2)
    ms.push(0)
    ms.push(-3)
    tests.append(("min_stack", ms.get_min(), -3))
    ms.pop()
    tests.append(("min_after_pop", ms.get_min(), -2))

    # Test 7: Daily temperatures
    tests.append(("daily_temp", daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]),
                  [1, 1, 4, 2, 1, 1, 0, 0]))

    # Test 8: Remove K digits
    tests.append(("remove_k", remove_k_digits("1432219", 3), "1219"))
    tests.append(("remove_k2", remove_k_digits("10200", 1), "200"))

    # Run all tests
    passed = 0
    print("\n" + "─" * 60)
    for name, result, expected in tests:
        if result == expected:
            passed += 1
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name}: got {result}, expected {expected}")

    print("─" * 60)
    print(f"\n  📊 Results: {passed}/{len(tests)} tests passed")
    return passed, len(tests)

if __name__ == "__main__":
    print(__doc__)
    passed, total = run_tests()
    if passed == total:
        print("\n  🎯 PERFECT SCORE!")
