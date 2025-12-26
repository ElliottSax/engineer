def longest_repeating_character_replacement(s, k):
    """Longest substring with same letter after k replacements."""
    from collections import defaultdict
    count = defaultdict(int)
    max_count = left = result = 0
    for right, char in enumerate(s):
        count[char] += 1
        max_count = max(max_count, count[char])
        while right - left + 1 - max_count > k:
            count[s[left]] -= 1
            left += 1
        result = max(result, right - left + 1)
    return result

def max_vowels_in_substring(s, k):
    """Maximum vowels in substring of length k."""
    vowels = set('aeiou')
    count = sum(1 for c in s[:k] if c in vowels)
    max_count = count
    for i in range(k, len(s)):
        count += (s[i] in vowels) - (s[i - k] in vowels)
        max_count = max(max_count, count)
    return max_count

def grumpy_bookstore_owner(customers, grumpy, minutes):
    """Maximum satisfied customers with k minutes of not grumpy."""
    n = len(customers)
    base = sum(c for c, g in zip(customers, grumpy) if g == 0)
    extra = sum(customers[i] * grumpy[i] for i in range(minutes))
    max_extra = extra
    for i in range(minutes, n):
        extra += customers[i] * grumpy[i] - customers[i - minutes] * grumpy[i - minutes]
        max_extra = max(max_extra, extra)
    return base + max_extra

def permutation_in_string(s1, s2):
    """Check if s2 contains permutation of s1."""
    from collections import Counter
    if len(s1) > len(s2):
        return False
    count1 = Counter(s1)
    count2 = Counter(s2[:len(s1)])
    if count1 == count2:
        return True
    for i in range(len(s1), len(s2)):
        count2[s2[i]] += 1
        count2[s2[i - len(s1)]] -= 1
        if count2[s2[i - len(s1)]] == 0:
            del count2[s2[i - len(s1)]]
        if count1 == count2:
            return True
    return False

def max_sum_two_non_overlapping(nums, first_len, second_len):
    """Maximum sum of two non-overlapping subarrays."""
    n = len(nums)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + nums[i]

    def max_sum(l1, l2):
        max_first = 0
        result = 0
        for i in range(l1 + l2, n + 1):
            max_first = max(max_first, prefix[i - l2] - prefix[i - l2 - l1])
            result = max(result, max_first + prefix[i] - prefix[i - l2])
        return result

    return max(max_sum(first_len, second_len), max_sum(second_len, first_len))

def subarrays_with_k_distinct(nums, k):
    """Count subarrays with exactly k distinct integers."""
    def at_most_k(k):
        from collections import defaultdict
        count = defaultdict(int)
        left = result = 0
        for right, num in enumerate(nums):
            if count[num] == 0:
                k -= 1
            count[num] += 1
            while k < 0:
                count[nums[left]] -= 1
                if count[nums[left]] == 0:
                    k += 1
                left += 1
            result += right - left + 1
        return result

    return at_most_k(k) - at_most_k(k - 1)

def count_nice_subarrays(nums, k):
    """Count subarrays with exactly k odd numbers."""
    def at_most(k):
        left = result = 0
        for right, num in enumerate(nums):
            k -= num % 2
            while k < 0:
                k += nums[left] % 2
                left += 1
            result += right - left + 1
        return result

    return at_most(k) - at_most(k - 1)

def max_erasure_value(nums):
    """Maximum sum of unique element subarray."""
    seen = {}
    left = result = current = 0
    for right, num in enumerate(nums):
        if num in seen and seen[num] >= left:
            while left <= seen[num]:
                current -= nums[left]
                left += 1
        current += num
        seen[num] = right
        result = max(result, current)
    return result

def minimum_window_substring(s, t):
    """Minimum window in s containing all characters of t."""
    from collections import Counter
    if not s or not t:
        return ""
    t_count = Counter(t)
    required = len(t_count)
    formed = 0
    window = {}
    left = 0
    ans = float('inf'), None, None

    for right, char in enumerate(s):
        window[char] = window.get(char, 0) + 1
        if char in t_count and window[char] == t_count[char]:
            formed += 1
        while formed == required:
            if right - left + 1 < ans[0]:
                ans = (right - left + 1, left, right)
            left_char = s[left]
            window[left_char] -= 1
            if left_char in t_count and window[left_char] < t_count[left_char]:
                formed -= 1
            left += 1

    return "" if ans[0] == float('inf') else s[ans[1]:ans[2] + 1]

# Tests
tests = [
    ("char_replace", longest_repeating_character_replacement("AABABBA", 1), 4),
    ("char_replace_2", longest_repeating_character_replacement("ABAB", 2), 4),
    ("max_vowels", max_vowels_in_substring("abciiidef", 3), 3),
    ("max_vowels_2", max_vowels_in_substring("aeiou", 2), 2),
    ("grumpy", grumpy_bookstore_owner([1,0,1,2,1,1,7,5], [0,1,0,1,0,1,0,1], 3), 16),
    ("perm_in_str", permutation_in_string("ab", "eidbaooo"), True),
    ("perm_in_str_no", permutation_in_string("ab", "eidboaoo"), False),
    ("two_non_overlap", max_sum_two_non_overlapping([0,6,5,2,2,5,1,9,4], 1, 2), 20),
    ("k_distinct", subarrays_with_k_distinct([1,2,1,2,3], 2), 7),
    ("nice_subarrays", count_nice_subarrays([1,1,2,1,1], 3), 2),
    ("max_erasure", max_erasure_value([4,2,4,5,6]), 17),
    ("min_window", minimum_window_substring("ADOBECODEBANC", "ABC"), "BANC"),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
