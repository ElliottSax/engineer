import heapq
from collections import Counter

def quick_sort(arr):
    """Quick sort implementation."""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

def heap_sort(arr):
    """Heap sort implementation."""
    heapq.heapify(arr)
    return [heapq.heappop(arr) for _ in range(len(arr))]

def counting_sort(arr, max_val):
    """Counting sort for integers in [0, max_val]."""
    count = [0] * (max_val + 1)
    for num in arr:
        count[num] += 1
    result = []
    for i, c in enumerate(count):
        result.extend([i] * c)
    return result

def radix_sort(arr):
    """Radix sort for non-negative integers."""
    if not arr:
        return arr
    max_val = max(arr)
    exp = 1
    while max_val // exp > 0:
        arr = counting_sort_by_digit(arr, exp)
        exp *= 10
    return arr

def counting_sort_by_digit(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    for num in arr:
        index = (num // exp) % 10
        count[index] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    for i in range(n - 1, -1, -1):
        index = (arr[i] // exp) % 10
        output[count[index] - 1] = arr[i]
        count[index] -= 1

    return output

def kth_largest_quickselect(nums, k):
    """Kth largest using quickselect."""
    k = len(nums) - k  # Convert to kth smallest

    def quickselect(left, right):
        pivot = nums[right]
        p = left
        for i in range(left, right):
            if nums[i] <= pivot:
                nums[p], nums[i] = nums[i], nums[p]
                p += 1
        nums[p], nums[right] = nums[right], nums[p]

        if p == k:
            return nums[p]
        elif p < k:
            return quickselect(p + 1, right)
        else:
            return quickselect(left, p - 1)

    return quickselect(0, len(nums) - 1)

def sort_by_frequency(nums):
    """Sort by frequency (ascending), then by value (descending)."""
    count = Counter(nums)
    return sorted(nums, key=lambda x: (count[x], -x))

def relative_sort_array(arr1, arr2):
    """Sort arr1 in order defined by arr2."""
    order = {v: i for i, v in enumerate(arr2)}
    return sorted(arr1, key=lambda x: (order.get(x, len(arr2)), x))

def maximum_gap(nums):
    """Maximum gap in sorted array."""
    if len(nums) < 2:
        return 0

    min_val, max_val = min(nums), max(nums)
    if min_val == max_val:
        return 0

    n = len(nums)
    bucket_size = max(1, (max_val - min_val) // (n - 1))
    bucket_count = (max_val - min_val) // bucket_size + 1

    buckets_min = [float('inf')] * bucket_count
    buckets_max = [float('-inf')] * bucket_count

    for num in nums:
        idx = (num - min_val) // bucket_size
        buckets_min[idx] = min(buckets_min[idx], num)
        buckets_max[idx] = max(buckets_max[idx], num)

    max_gap = 0
    prev_max = min_val

    for i in range(bucket_count):
        if buckets_min[i] == float('inf'):
            continue
        max_gap = max(max_gap, buckets_min[i] - prev_max)
        prev_max = buckets_max[i]

    return max_gap

def h_index(citations):
    """H-index of researcher."""
    citations.sort(reverse=True)
    h = 0
    for i, c in enumerate(citations):
        if c >= i + 1:
            h = i + 1
        else:
            break
    return h

def wiggle_sort_ii(nums):
    """Wiggle sort: nums[0] < nums[1] > nums[2] < nums[3]..."""
    n = len(nums)
    median = sorted(nums)[n // 2]

    def map_index(i):
        return (1 + 2 * i) % (n | 1)

    left, i, right = 0, 0, n - 1

    while i <= right:
        if nums[map_index(i)] > median:
            nums[map_index(left)], nums[map_index(i)] = nums[map_index(i)], nums[map_index(left)]
            left += 1
            i += 1
        elif nums[map_index(i)] < median:
            nums[map_index(i)], nums[map_index(right)] = nums[map_index(right)], nums[map_index(i)]
            right -= 1
        else:
            i += 1

    return nums

def pancake_sort(arr):
    """Pancake sort - return flip sequence."""
    flips = []
    n = len(arr)

    for size in range(n, 1, -1):
        max_idx = arr.index(max(arr[:size]))
        if max_idx != size - 1:
            if max_idx > 0:
                flips.append(max_idx + 1)
                arr[:max_idx + 1] = arr[:max_idx + 1][::-1]
            flips.append(size)
            arr[:size] = arr[:size][::-1]

    return flips

def sorted_squares(nums):
    """Squares of sorted array in sorted order."""
    n = len(nums)
    result = [0] * n
    left, right = 0, n - 1
    idx = n - 1

    while left <= right:
        if abs(nums[left]) > abs(nums[right]):
            result[idx] = nums[left] ** 2
            left += 1
        else:
            result[idx] = nums[right] ** 2
            right -= 1
        idx -= 1

    return result

# Tests
tests = [
    ("quick", quick_sort([3,1,4,1,5,9,2,6]), [1,1,2,3,4,5,6,9]),
    ("heap", heap_sort([3,1,4,1,5,9,2,6]), [1,1,2,3,4,5,6,9]),
    ("counting", counting_sort([1,4,1,2,7,5,2], 7), [1,1,2,2,4,5,7]),
    ("radix", radix_sort([170,45,75,90,802,24,2,66]), [2,24,45,66,75,90,170,802]),
    ("kth_largest", kth_largest_quickselect([3,2,1,5,6,4], 2), 5),
    ("freq_sort", sort_by_frequency([1,1,2,2,2,3]), [3,1,1,2,2,2]),
    ("relative", relative_sort_array([2,3,1,3,2,4,6,7,9,2,19], [2,1,4,3,9,6]), [2,2,2,1,4,3,3,9,6,7,19]),
    ("max_gap", maximum_gap([3,6,9,1]), 3),
    ("h_index", h_index([3,0,6,1,5]), 3),
    ("pancake", len(pancake_sort([3,2,4,1])) <= 10, True),
    ("squares", sorted_squares([-4,-1,0,3,10]), [0,1,9,16,100]),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
