def find_missing_number(nums):
    """Finds missing number in [0, n] using XOR."""
    n = len(nums)
    result = n
    for i, num in enumerate(nums):
        result ^= i ^ num
    return result

def happy_number(n):
    """Checks if number is happy (sum of squares of digits eventually equals 1)."""
    seen = set()
    while n != 1 and n not in seen:
        seen.add(n)
        n = sum(int(d) ** 2 for d in str(n))
    return n == 1

def count_primes(n):
    """Counts primes less than n using Sieve of Eratosthenes."""
    if n < 2:
        return 0
    is_prime = [True] * n
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n, i):
                is_prime[j] = False
    return sum(is_prime)

def longest_substring_without_repeat(s):
    """Length of longest substring without repeating characters."""
    char_index = {}
    max_len = start = 0
    for i, char in enumerate(s):
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        char_index[char] = i
        max_len = max(max_len, i - start + 1)
    return max_len

def container_with_most_water(heights):
    """Maximum water container between two lines."""
    left, right = 0, len(heights) - 1
    max_water = 0
    while left < right:
        width = right - left
        height = min(heights[left], heights[right])
        max_water = max(max_water, width * height)
        if heights[left] < heights[right]:
            left += 1
        else:
            right -= 1
    return max_water

def search_rotated_sorted(nums, target):
    """Binary search in rotated sorted array."""
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1

def group_anagrams(strs):
    """Groups anagrams together."""
    from collections import defaultdict
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))
        groups[key].append(s)
    return list(groups.values())

def minimum_window_substring(s, t):
    """Minimum window in s containing all chars of t."""
    from collections import Counter
    if not s or not t:
        return ""
    t_count = Counter(t)
    required = len(t_count)
    left = formed = 0
    window_counts = {}
    ans = (float('inf'), 0, 0)
    for right, char in enumerate(s):
        window_counts[char] = window_counts.get(char, 0) + 1
        if char in t_count and window_counts[char] == t_count[char]:
            formed += 1
        while formed == required:
            if right - left + 1 < ans[0]:
                ans = (right - left + 1, left, right)
            left_char = s[left]
            window_counts[left_char] -= 1
            if left_char in t_count and window_counts[left_char] < t_count[left_char]:
                formed -= 1
            left += 1
    return "" if ans[0] == float('inf') else s[ans[1]:ans[2] + 1]

def jump_game(nums):
    """Can reach last index from first?"""
    max_reach = 0
    for i, jump in enumerate(nums):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + jump)
    return True

def meeting_rooms(intervals):
    """Minimum meeting rooms needed."""
    if not intervals:
        return 0
    starts = sorted(i[0] for i in intervals)
    ends = sorted(i[1] for i in intervals)
    rooms = max_rooms = 0
    s = e = 0
    while s < len(starts):
        if starts[s] < ends[e]:
            rooms += 1
            max_rooms = max(max_rooms, rooms)
            s += 1
        else:
            rooms -= 1
            e += 1
    return max_rooms

# Tests
tests = [
    ("missing_number", find_missing_number([3, 0, 1]), 2),
    ("happy_number_19", happy_number(19), True),
    ("happy_number_2", happy_number(2), False),
    ("count_primes", count_primes(10), 4),
    ("longest_substr", longest_substring_without_repeat("abcabcbb"), 3),
    ("container_water", container_with_most_water([1,8,6,2,5,4,8,3,7]), 49),
    ("search_rotated", search_rotated_sorted([4,5,6,7,0,1,2], 0), 4),
    ("group_anagrams", len(group_anagrams(["eat","tea","tan","ate","nat","bat"])), 3),
    ("min_window", minimum_window_substring("ADOBECODEBANC", "ABC"), "BANC"),
    ("jump_game_yes", jump_game([2,3,1,1,4]), True),
    ("jump_game_no", jump_game([3,2,1,0,4]), False),
    ("meeting_rooms", meeting_rooms([[0,30],[5,10],[15,20]]), 2),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
