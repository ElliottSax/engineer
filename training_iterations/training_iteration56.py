def product_except_self(nums):
    """Product of array except self without division."""
    n = len(nums)
    result = [1] * n

    # Left products
    left = 1
    for i in range(n):
        result[i] = left
        left *= nums[i]

    # Right products
    right = 1
    for i in range(n - 1, -1, -1):
        result[i] *= right
        right *= nums[i]

    return result

def next_permutation(nums):
    """Next lexicographically greater permutation."""
    n = len(nums)
    i = n - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1

    if i >= 0:
        j = n - 1
        while nums[j] <= nums[i]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]

    nums[i + 1:] = reversed(nums[i + 1:])
    return nums

def first_missing_positive(nums):
    """First missing positive in O(n) time O(1) space."""
    n = len(nums)

    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]

    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    return n + 1

def jump_game(nums):
    """Check if can reach last index."""
    max_reach = 0
    for i, num in enumerate(nums):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + num)
    return True

def jump_game_ii(nums):
    """Minimum jumps to reach end."""
    jumps = current_end = farthest = 0
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == current_end:
            jumps += 1
            current_end = farthest
    return jumps

def candy_distribution(ratings):
    """Minimum candies with rating constraints."""
    n = len(ratings)
    candies = [1] * n

    for i in range(1, n):
        if ratings[i] > ratings[i - 1]:
            candies[i] = candies[i - 1] + 1

    for i in range(n - 2, -1, -1):
        if ratings[i] > ratings[i + 1]:
            candies[i] = max(candies[i], candies[i + 1] + 1)

    return sum(candies)

def gas_station(gas, cost):
    """Starting gas station for circular route."""
    total = current = start = 0
    for i in range(len(gas)):
        diff = gas[i] - cost[i]
        total += diff
        current += diff
        if current < 0:
            start = i + 1
            current = 0
    return start if total >= 0 else -1

def container_with_most_water(height):
    """Maximum water container."""
    left, right = 0, len(height) - 1
    max_area = 0

    while left < right:
        area = min(height[left], height[right]) * (right - left)
        max_area = max(max_area, area)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_area

def trapping_rain_water(height):
    """Total trapped rain water."""
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

def rotate_array(nums, k):
    """Rotate array k positions in place."""
    n = len(nums)
    k %= n

    def reverse(start, end):
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1

    reverse(0, n - 1)
    reverse(0, k - 1)
    reverse(k, n - 1)
    return nums

def max_subarray_circular(nums):
    """Maximum subarray sum (circular)."""
    def kadane(arr):
        max_sum = curr = arr[0]
        for num in arr[1:]:
            curr = max(num, curr + num)
            max_sum = max(max_sum, curr)
        return max_sum

    max_normal = kadane(nums)
    total = sum(nums)
    min_subarray = -kadane([-x for x in nums])

    if min_subarray == total:
        return max_normal
    return max(max_normal, total - min_subarray)

def longest_consecutive_sequence(nums):
    """Longest consecutive sequence in O(n)."""
    num_set = set(nums)
    max_len = 0

    for num in num_set:
        if num - 1 not in num_set:
            length = 1
            while num + length in num_set:
                length += 1
            max_len = max(max_len, length)

    return max_len

def majority_element_ii(nums):
    """All elements appearing more than n/3 times."""
    count1 = count2 = 0
    candidate1 = candidate2 = None

    for num in nums:
        if candidate1 == num:
            count1 += 1
        elif candidate2 == num:
            count2 += 1
        elif count1 == 0:
            candidate1 = num
            count1 = 1
        elif count2 == 0:
            candidate2 = num
            count2 = 1
        else:
            count1 -= 1
            count2 -= 1

    # Verify
    threshold = len(nums) // 3
    result = []
    if nums.count(candidate1) > threshold:
        result.append(candidate1)
    if candidate2 != candidate1 and nums.count(candidate2) > threshold:
        result.append(candidate2)
    return sorted(result)

# Tests
tests = [
    ("product", product_except_self([1,2,3,4]), [24,12,8,6]),
    ("product_zero", product_except_self([0,1,2]), [2,0,0]),
    ("next_perm", next_permutation([1,2,3]), [1,3,2]),
    ("next_perm_2", next_permutation([3,2,1]), [1,2,3]),
    ("first_missing", first_missing_positive([3,4,-1,1]), 2),
    ("first_missing_2", first_missing_positive([1,2,0]), 3),
    ("jump", jump_game([2,3,1,1,4]), True),
    ("jump_fail", jump_game([3,2,1,0,4]), False),
    ("jump_ii", jump_game_ii([2,3,1,1,4]), 2),
    ("candy", candy_distribution([1,0,2]), 5),
    ("gas", gas_station([1,2,3,4,5], [3,4,5,1,2]), 3),
    ("container", container_with_most_water([1,8,6,2,5,4,8,3,7]), 49),
    ("trap", trapping_rain_water([0,1,0,2,1,0,1,3,2,1,2,1]), 6),
    ("rotate", rotate_array([1,2,3,4,5,6,7], 3), [5,6,7,1,2,3,4]),
    ("circular_max", max_subarray_circular([5,-3,5]), 10),
    ("circular_max_2", max_subarray_circular([-3,-2,-3]), -2),
    ("consecutive", longest_consecutive_sequence([100,4,200,1,3,2]), 4),
    ("majority", majority_element_ii([3,2,3]), [3]),
    ("majority_2", majority_element_ii([1,2]), [1,2]),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
