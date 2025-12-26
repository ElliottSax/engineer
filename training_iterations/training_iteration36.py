def three_sum(nums):
    """Finds all unique triplets that sum to zero."""
    nums.sort()
    result = []
    n = len(nums)
    for i in range(n - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left, right = i + 1, n - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
    return result

def four_sum(nums, target):
    """Finds all unique quadruplets that sum to target."""
    nums.sort()
    result = []
    n = len(nums)
    for i in range(n - 3):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        for j in range(i + 1, n - 2):
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue
            left, right = j + 1, n - 1
            while left < right:
                total = nums[i] + nums[j] + nums[left] + nums[right]
                if total < target:
                    left += 1
                elif total > target:
                    right -= 1
                else:
                    result.append([nums[i], nums[j], nums[left], nums[right]])
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    left += 1
                    right -= 1
    return result

def three_sum_closest(nums, target):
    """Finds triplet sum closest to target."""
    nums.sort()
    n = len(nums)
    closest = float('inf')
    for i in range(n - 2):
        left, right = i + 1, n - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if abs(total - target) < abs(closest - target):
                closest = total
            if total < target:
                left += 1
            elif total > target:
                right -= 1
            else:
                return target
    return closest

def three_sum_smaller(nums, target):
    """Counts triplets with sum less than target."""
    nums.sort()
    n = len(nums)
    count = 0
    for i in range(n - 2):
        left, right = i + 1, n - 1
        while left < right:
            if nums[i] + nums[left] + nums[right] < target:
                count += right - left
                left += 1
            else:
                right -= 1
    return count

def two_sum_sorted(nums, target):
    """Two sum on sorted array."""
    left, right = 0, len(nums) - 1
    while left < right:
        total = nums[left] + nums[right]
        if total == target:
            return [left + 1, right + 1]
        elif total < target:
            left += 1
        else:
            right -= 1
    return []

def two_sum_less_than_k(nums, k):
    """Maximum sum of two numbers less than k."""
    nums.sort()
    left, right = 0, len(nums) - 1
    max_sum = -1
    while left < right:
        total = nums[left] + nums[right]
        if total < k:
            max_sum = max(max_sum, total)
            left += 1
        else:
            right -= 1
    return max_sum

def valid_triangle_number(nums):
    """Counts valid triangles from array."""
    nums.sort()
    count = 0
    n = len(nums)
    for k in range(n - 1, 1, -1):
        i, j = 0, k - 1
        while i < j:
            if nums[i] + nums[j] > nums[k]:
                count += j - i
                j -= 1
            else:
                i += 1
    return count

def boats_to_save_people(people, limit):
    """Minimum boats to save people (at most 2 per boat)."""
    people.sort()
    left, right = 0, len(people) - 1
    boats = 0
    while left <= right:
        if people[left] + people[right] <= limit:
            left += 1
        right -= 1
        boats += 1
    return boats

def sort_colors(nums):
    """Dutch flag problem - sort 0s, 1s, 2s."""
    low = mid = 0
    high = len(nums) - 1
    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1
    return nums

def partition_array(nums, pivot):
    """Partition array around pivot."""
    less = []
    equal = []
    greater = []
    for num in nums:
        if num < pivot:
            less.append(num)
        elif num == pivot:
            equal.append(num)
        else:
            greater.append(num)
    return less + equal + greater

# Tests
tests = [
    ("three_sum", sorted([sorted(x) for x in three_sum([-1,0,1,2,-1,-4])]),
     sorted([sorted(x) for x in [[-1,-1,2],[-1,0,1]]])),
    ("four_sum", sorted([sorted(x) for x in four_sum([1,0,-1,0,-2,2], 0)]),
     sorted([sorted(x) for x in [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]])),
    ("three_sum_closest", three_sum_closest([-1,2,1,-4], 1), 2),
    ("three_sum_smaller", three_sum_smaller([-2,0,1,3], 2), 2),
    ("two_sum_sorted", two_sum_sorted([2,7,11,15], 9), [1, 2]),
    ("two_sum_less", two_sum_less_than_k([34,23,1,24,75,33,54,8], 60), 58),
    ("triangles", valid_triangle_number([2,2,3,4]), 3),
    ("boats", boats_to_save_people([3,2,2,1], 3), 3),
    ("sort_colors", sort_colors([2,0,2,1,1,0]), [0,0,1,1,2,2]),
    ("partition", partition_array([3,1,4,1,5,9,2,6], 4), [3,1,1,2,4,5,9,6]),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
