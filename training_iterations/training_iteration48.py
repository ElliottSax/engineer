def find_right_interval(intervals):
    """Find minimum right interval for each interval."""
    import bisect
    n = len(intervals)
    sorted_starts = sorted((start, i) for i, (start, end) in enumerate(intervals))
    starts = [s for s, _ in sorted_starts]
    result = []
    for start, end in intervals:
        idx = bisect.bisect_left(starts, end)
        if idx < n:
            result.append(sorted_starts[idx][1])
        else:
            result.append(-1)
    return result

def search_insert_position(nums, target):
    """Find insert position for target."""
    import bisect
    return bisect.bisect_left(nums, target)

def find_first_and_last(nums, target):
    """Find first and last position of target."""
    import bisect
    left = bisect.bisect_left(nums, target)
    if left >= len(nums) or nums[left] != target:
        return [-1, -1]
    right = bisect.bisect_right(nums, target) - 1
    return [left, right]

def count_smaller_than_self(nums):
    """Count smaller elements to the right."""
    import bisect
    sorted_list = []
    result = []
    for num in reversed(nums):
        idx = bisect.bisect_left(sorted_list, num)
        result.append(idx)
        bisect.insort(sorted_list, num)
    return result[::-1]

def find_k_closest_elements(arr, k, x):
    """Find k closest elements to x."""
    left = 0
    right = len(arr) - k
    while left < right:
        mid = (left + right) // 2
        if x - arr[mid] > arr[mid + k] - x:
            left = mid + 1
        else:
            right = mid
    return arr[left:left + k]

def minimum_in_rotated_sorted(nums):
    """Find minimum in rotated sorted array."""
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid
    return nums[left]

def search_in_rotated(nums, target):
    """Search in rotated sorted array."""
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

def find_peak(nums):
    """Find a peak element."""
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[mid + 1]:
            right = mid
        else:
            left = mid + 1
    return left

def search_in_infinite_array(reader, target):
    """Search in sorted array of unknown size."""
    # reader.get(i) returns element at i or MAX_INT if out of bounds
    if reader(0) == target:
        return 0
    left, right = 0, 1
    while reader(right) < target:
        left = right
        right *= 2
    while left <= right:
        mid = (left + right) // 2
        val = reader(mid)
        if val == target:
            return mid
        elif val < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def minimum_speed_to_arrive(dist, hour):
    """Minimum speed to arrive on time."""
    import math
    if len(dist) > math.ceil(hour):
        return -1
    left, right = 1, 10**7
    while left < right:
        mid = (left + right) // 2
        time = sum(math.ceil(d / mid) for d in dist[:-1]) + dist[-1] / mid
        if time <= hour:
            right = mid
        else:
            left = mid + 1
    return left

def koko_eating_bananas(piles, h):
    """Minimum eating speed to finish in h hours."""
    import math
    left, right = 1, max(piles)
    while left < right:
        mid = (left + right) // 2
        hours = sum(math.ceil(p / mid) for p in piles)
        if hours <= h:
            right = mid
        else:
            left = mid + 1
    return left

def ship_within_days(weights, days):
    """Minimum ship capacity to ship within days."""
    left, right = max(weights), sum(weights)
    while left < right:
        mid = (left + right) // 2
        curr_weight = 0
        curr_days = 1
        for w in weights:
            if curr_weight + w > mid:
                curr_days += 1
                curr_weight = 0
            curr_weight += w
        if curr_days <= days:
            right = mid
        else:
            left = mid + 1
    return left

# Tests
tests = [
    ("right_interval", find_right_interval([[3,4],[2,3],[1,2]]), [-1, 0, 1]),
    ("search_insert", search_insert_position([1,3,5,6], 5), 2),
    ("search_insert_2", search_insert_position([1,3,5,6], 2), 1),
    ("first_last", find_first_and_last([5,7,7,8,8,10], 8), [3, 4]),
    ("first_last_no", find_first_and_last([5,7,7,8,8,10], 6), [-1, -1]),
    ("count_smaller", count_smaller_than_self([5,2,6,1]), [2, 1, 1, 0]),
    ("k_closest", find_k_closest_elements([1,2,3,4,5], 4, 3), [1, 2, 3, 4]),
    ("min_rotated", minimum_in_rotated_sorted([3,4,5,1,2]), 1),
    ("search_rotated", search_in_rotated([4,5,6,7,0,1,2], 0), 4),
    ("find_peak", find_peak([1,2,1,3,5,6,4]) in [1, 5], True),
    ("min_speed", minimum_speed_to_arrive([1,3,2], 6), 1),
    ("koko", koko_eating_bananas([3,6,7,11], 8), 4),
    ("ship", ship_within_days([1,2,3,4,5,6,7,8,9,10], 5), 15),
]

# Infinite array search test
def mock_reader(arr):
    def reader(i):
        return arr[i] if i < len(arr) else 2**31 - 1
    return reader

arr = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
tests.append(("infinite_search", search_in_infinite_array(mock_reader(arr), 11), 5))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
