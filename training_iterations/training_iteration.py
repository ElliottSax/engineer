def find_duplicates(lst):
    """Finds duplicate elements in a list and returns them sorted."""
    seen = set()
    duplicates = set()
    for item in lst:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return sorted(list(duplicates))

def is_palindrome(s):
    """Checks if a string is a palindrome."""
    s = s.lower().replace(" ", "")
    return s == s[::-1]

def fibonacci(n):
    """Returns the first n fibonacci numbers."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    result = [0, 1]
    for _ in range(2, n):
        result.append(result[-1] + result[-2])
    return result[:n]

def is_prime(n):
    """Checks if a number is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def max_subarray_sum(arr):
    """Finds maximum sum of contiguous subarray (Kadane's algorithm)."""
    if not arr:
        return 0
    max_sum = current_sum = arr[0]
    for num in arr[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum

def valid_parentheses(s):
    """Checks if brackets are properly matched."""
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    for char in s:
        if char in '([{':
            stack.append(char)
        elif char in ')]}':
            if not stack or stack.pop() != pairs[char]:
                return False
    return len(stack) == 0

def coin_change(coins, amount):
    """Finds minimum coins needed for amount, -1 if impossible."""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1
    return dp[amount] if dp[amount] != float('inf') else -1

def merge_intervals(intervals):
    """Merges overlapping intervals."""
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged

# Run tests
tests = [
    ("find_duplicates", find_duplicates([1,2,3,2,4,3]), [2,3]),
    ("is_palindrome racecar", is_palindrome("racecar"), True),
    ("is_palindrome hello", is_palindrome("hello"), False),
    ("fibonacci(5)", fibonacci(5), [0,1,1,2,3]),
    ("is_prime(7)", is_prime(7), True),
    ("is_prime(4)", is_prime(4), False),
    ("max_subarray", max_subarray_sum([-2,1,-3,4,-1,2,1,-5,4]), 6),
    ("valid_parens ()", valid_parentheses("()[]{}"), True),
    ("valid_parens ([)]", valid_parentheses("([)]"), False),
    ("coin_change", coin_change([1,2,5], 11), 3),
    ("merge_intervals", merge_intervals([[1,3],[2,6],[8,10]]), [[1,6],[8,10]]),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}: {result}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
