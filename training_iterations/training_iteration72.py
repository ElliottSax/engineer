def three_sum(nums):
    """Find all unique triplets that sum to zero."""
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
    """Find all unique quadruplets that sum to target."""
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
    """Find triplet sum closest to target."""
    nums.sort()
    closest = float('inf')
    n = len(nums)

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
    """Count triplets with sum less than target."""
    nums.sort()
    count = 0
    n = len(nums)

    for i in range(n - 2):
        left, right = i + 1, n - 1
        while left < right:
            if nums[i] + nums[left] + nums[right] < target:
                count += right - left
                left += 1
            else:
                right -= 1

    return count

def sort_colors(nums):
    """Dutch national flag - sort 0s, 1s, 2s."""
    left, curr, right = 0, 0, len(nums) - 1

    while curr <= right:
        if nums[curr] == 0:
            nums[left], nums[curr] = nums[curr], nums[left]
            left += 1
            curr += 1
        elif nums[curr] == 2:
            nums[curr], nums[right] = nums[right], nums[curr]
            right -= 1
        else:
            curr += 1

    return nums

def move_zeros(nums):
    """Move all zeros to end."""
    left = 0
    for right in range(len(nums)):
        if nums[right] != 0:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
    return nums

def remove_duplicates_sorted(nums):
    """Remove duplicates from sorted array."""
    if not nums:
        return 0
    write = 1
    for i in range(1, len(nums)):
        if nums[i] != nums[i - 1]:
            nums[write] = nums[i]
            write += 1
    return write

def remove_duplicates_at_most_two(nums):
    """Allow at most 2 duplicates."""
    if len(nums) <= 2:
        return len(nums)
    write = 2
    for i in range(2, len(nums)):
        if nums[i] != nums[write - 2]:
            nums[write] = nums[i]
            write += 1
    return write

def merge_sorted_arrays(nums1, m, nums2, n):
    """Merge nums2 into nums1."""
    i, j, k = m - 1, n - 1, m + n - 1

    while j >= 0:
        if i >= 0 and nums1[i] > nums2[j]:
            nums1[k] = nums1[i]
            i -= 1
        else:
            nums1[k] = nums2[j]
            j -= 1
        k -= 1

    return nums1

def boats_to_save_people(people, limit):
    """Minimum boats to save people."""
    people.sort()
    left, right = 0, len(people) - 1
    boats = 0

    while left <= right:
        if people[left] + people[right] <= limit:
            left += 1
        right -= 1
        boats += 1

    return boats

def valid_palindrome_ii(s):
    """Check if palindrome after removing at most one char."""
    def is_palindrome(l, r):
        while l < r:
            if s[l] != s[r]:
                return False
            l += 1
            r -= 1
        return True

    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return is_palindrome(left + 1, right) or is_palindrome(left, right - 1)
        left += 1
        right -= 1
    return True

def remove_element(nums, val):
    """Remove all instances of val."""
    write = 0
    for num in nums:
        if num != val:
            nums[write] = num
            write += 1
    return write

# Tests
tests = [
    ("three_sum", sorted([tuple(x) for x in three_sum([-1,0,1,2,-1,-4])]),
     sorted([tuple(x) for x in [[-1,-1,2],[-1,0,1]]])),
    ("four_sum", sorted([tuple(x) for x in four_sum([1,0,-1,0,-2,2], 0)]),
     sorted([tuple(x) for x in [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]])),
    ("closest", three_sum_closest([-1,2,1,-4], 1), 2),
    ("smaller", three_sum_smaller([-2,0,1,3], 2), 2),
    ("colors", sort_colors([2,0,2,1,1,0]), [0,0,1,1,2,2]),
    ("zeros", move_zeros([0,1,0,3,12]), [1,3,12,0,0]),
    ("remove_dup", remove_duplicates_sorted([1,1,2]), 2),
    ("remove_dup2", remove_duplicates_at_most_two([1,1,1,2,2,3]), 5),
    ("merge", merge_sorted_arrays([1,2,3,0,0,0], 3, [2,5,6], 3), [1,2,2,3,5,6]),
    ("boats", boats_to_save_people([3,2,2,1], 3), 3),
    ("valid_pal_ii", valid_palindrome_ii("abca"), True),
    ("remove_elem", remove_element([3,2,2,3], 3), 2),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
