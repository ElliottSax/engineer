def subarray_sum_equals_k(nums, k):
    """Counts subarrays with sum equal to k."""
    from collections import defaultdict
    count = 0
    prefix_sum = 0
    prefix_count = defaultdict(int)
    prefix_count[0] = 1
    for num in nums:
        prefix_sum += num
        count += prefix_count[prefix_sum - k]
        prefix_count[prefix_sum] += 1
    return count

def subarray_sum_divisible_k(nums, k):
    """Counts subarrays with sum divisible by k."""
    from collections import defaultdict
    count = 0
    prefix_sum = 0
    mod_count = defaultdict(int)
    mod_count[0] = 1
    for num in nums:
        prefix_sum += num
        mod = prefix_sum % k
        count += mod_count[mod]
        mod_count[mod] += 1
    return count

def continuous_subarray_sum(nums, k):
    """Checks if subarray of size >= 2 sums to multiple of k."""
    prefix_sum = 0
    mod_index = {0: -1}
    for i, num in enumerate(nums):
        prefix_sum += num
        mod = prefix_sum % k if k != 0 else prefix_sum
        if mod in mod_index:
            if i - mod_index[mod] >= 2:
                return True
        else:
            mod_index[mod] = i
    return False

def max_subarray_sum_k(nums, k):
    """Maximum sum of subarray with length k."""
    n = len(nums)
    if n < k:
        return 0
    window_sum = sum(nums[:k])
    max_sum = window_sum
    for i in range(k, n):
        window_sum += nums[i] - nums[i - k]
        max_sum = max(max_sum, window_sum)
    return max_sum

def shortest_subarray_sum_at_least_k(nums, k):
    """Shortest subarray with sum at least k."""
    from collections import deque
    n = len(nums)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + nums[i]
    result = float('inf')
    dq = deque()
    for i in range(n + 1):
        while dq and prefix[i] - prefix[dq[0]] >= k:
            result = min(result, i - dq.popleft())
        while dq and prefix[i] <= prefix[dq[-1]]:
            dq.pop()
        dq.append(i)
    return result if result != float('inf') else -1

def minimum_size_subarray_sum(nums, target):
    """Minimum length subarray with sum >= target."""
    n = len(nums)
    left = window_sum = 0
    min_len = float('inf')
    for right in range(n):
        window_sum += nums[right]
        while window_sum >= target:
            min_len = min(min_len, right - left + 1)
            window_sum -= nums[left]
            left += 1
    return min_len if min_len != float('inf') else 0

def max_consecutive_ones_iii(nums, k):
    """Maximum consecutive 1s with at most k flips."""
    left = zeros = 0
    max_len = 0
    for right in range(len(nums)):
        if nums[right] == 0:
            zeros += 1
        while zeros > k:
            if nums[left] == 0:
                zeros -= 1
            left += 1
        max_len = max(max_len, right - left + 1)
    return max_len

def longest_subarray_diff_limit(nums, limit):
    """Longest subarray with max difference <= limit."""
    from collections import deque
    max_dq = deque()
    min_dq = deque()
    left = 0
    result = 0
    for right in range(len(nums)):
        while max_dq and nums[right] > max_dq[-1]:
            max_dq.pop()
        max_dq.append(nums[right])
        while min_dq and nums[right] < min_dq[-1]:
            min_dq.pop()
        min_dq.append(nums[right])
        while max_dq[0] - min_dq[0] > limit:
            if nums[left] == max_dq[0]:
                max_dq.popleft()
            if nums[left] == min_dq[0]:
                min_dq.popleft()
            left += 1
        result = max(result, right - left + 1)
    return result

def fruit_into_baskets(fruits):
    """Maximum fruits with at most 2 types."""
    from collections import defaultdict
    basket = defaultdict(int)
    left = 0
    max_fruits = 0
    for right, fruit in enumerate(fruits):
        basket[fruit] += 1
        while len(basket) > 2:
            basket[fruits[left]] -= 1
            if basket[fruits[left]] == 0:
                del basket[fruits[left]]
            left += 1
        max_fruits = max(max_fruits, right - left + 1)
    return max_fruits

def longest_substring_k_distinct(s, k):
    """Longest substring with at most k distinct characters."""
    from collections import defaultdict
    char_count = defaultdict(int)
    left = 0
    max_len = 0
    for right, char in enumerate(s):
        char_count[char] += 1
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1
        max_len = max(max_len, right - left + 1)
    return max_len

def character_replacement(s, k):
    """Longest substring with same letter after k replacements."""
    from collections import defaultdict
    count = defaultdict(int)
    left = max_count = 0
    result = 0
    for right, char in enumerate(s):
        count[char] += 1
        max_count = max(max_count, count[char])
        while right - left + 1 - max_count > k:
            count[s[left]] -= 1
            left += 1
        result = max(result, right - left + 1)
    return result

# Tests
tests = [
    ("subarray_sum_k", subarray_sum_equals_k([1,1,1], 2), 2),
    ("subarray_sum_k_2", subarray_sum_equals_k([1,2,3], 3), 2),
    ("subarray_div_k", subarray_sum_divisible_k([4,5,0,-2,-3,1], 5), 7),
    ("continuous_sum", continuous_subarray_sum([23,2,4,6,7], 6), True),
    ("continuous_sum_no", continuous_subarray_sum([23,2,6,4,7], 13), False),
    ("max_sum_k", max_subarray_sum_k([1,4,2,10,23,3,1,0,20], 4), 39),
    ("shortest_at_least", shortest_subarray_sum_at_least_k([2,-1,2], 3), 3),
    ("min_size_sum", minimum_size_subarray_sum([2,3,1,2,4,3], 7), 2),
    ("max_ones_iii", max_consecutive_ones_iii([1,1,1,0,0,0,1,1,1,1,0], 2), 6),
    ("longest_diff", longest_subarray_diff_limit([8,2,4,7], 4), 2),
    ("fruit_baskets", fruit_into_baskets([1,2,1]), 3),
    ("fruit_baskets_2", fruit_into_baskets([0,1,2,2]), 3),
    ("k_distinct", longest_substring_k_distinct("eceba", 2), 3),
    ("char_replace", character_replacement("AABABBA", 1), 4),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
