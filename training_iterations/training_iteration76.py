def sqrt_binary_search(x):
    """Integer square root using binary search."""
    if x < 2:
        return x
    left, right = 1, x // 2
    while left <= right:
        mid = (left + right) // 2
        if mid * mid == x:
            return mid
        elif mid * mid < x:
            left = mid + 1
        else:
            right = mid - 1
    return right

def find_peak_element_2d(mat):
    """Find peak element in 2D matrix."""
    m, n = len(mat), len(mat[0])

    def find_max_col(mid):
        max_row = 0
        for i in range(m):
            if mat[i][mid] > mat[max_row][mid]:
                max_row = i
        return max_row

    left, right = 0, n - 1
    while left <= right:
        mid = (left + right) // 2
        max_row = find_max_col(mid)
        left_val = mat[max_row][mid - 1] if mid > 0 else -1
        right_val = mat[max_row][mid + 1] if mid < n - 1 else -1

        if mat[max_row][mid] > left_val and mat[max_row][mid] > right_val:
            return [max_row, mid]
        elif mat[max_row][mid] < left_val:
            right = mid - 1
        else:
            left = mid + 1

    return [-1, -1]

def search_matrix_ii(matrix, target):
    """Search in row and column sorted matrix."""
    if not matrix:
        return False
    m, n = len(matrix), len(matrix[0])
    row, col = 0, n - 1

    while row < m and col >= 0:
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] > target:
            col -= 1
        else:
            row += 1
    return False

def find_duplicate_number(nums):
    """Find duplicate in [1, n] with n+1 numbers."""
    slow = fast = nums[0]
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break

    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]

    return slow

def single_element_sorted(nums):
    """Find single non-duplicate in sorted array."""
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if mid % 2 == 1:
            mid -= 1
        if nums[mid] == nums[mid + 1]:
            left = mid + 2
        else:
            right = mid
    return nums[left]

def find_min_rotated_ii(nums):
    """Find min in rotated sorted array with duplicates."""
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[right]:
            left = mid + 1
        elif nums[mid] < nums[right]:
            right = mid
        else:
            right -= 1
    return nums[left]

def search_rotated_ii(nums, target):
    """Search in rotated sorted array with duplicates."""
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return True
        if nums[left] == nums[mid] == nums[right]:
            left += 1
            right -= 1
        elif nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return False

def count_negatives(grid):
    """Count negatives in sorted matrix."""
    m, n = len(grid), len(grid[0])
    count = 0
    row, col = 0, n - 1

    while row < m and col >= 0:
        if grid[row][col] < 0:
            count += m - row
            col -= 1
        else:
            row += 1

    return count

def find_kth_positive(arr, k):
    """Find kth missing positive."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        missing = arr[mid] - mid - 1
        if missing < k:
            left = mid + 1
        else:
            right = mid - 1
    return left + k

def split_array_largest_sum(nums, m):
    """Minimize largest sum when splitting array into m parts."""
    def can_split(max_sum):
        count = 1
        curr_sum = 0
        for num in nums:
            curr_sum += num
            if curr_sum > max_sum:
                count += 1
                curr_sum = num
        return count <= m

    left, right = max(nums), sum(nums)
    while left < right:
        mid = (left + right) // 2
        if can_split(mid):
            right = mid
        else:
            left = mid + 1
    return left

def minimize_max_distance_to_gas(stations, k):
    """Minimize max distance after adding k gas stations."""
    def count_stations(dist):
        count = 0
        for i in range(len(stations) - 1):
            count += int((stations[i + 1] - stations[i]) / dist)
        return count

    left, right = 0, stations[-1] - stations[0]
    while right - left > 1e-6:
        mid = (left + right) / 2
        if count_stations(mid) <= k:
            right = mid
        else:
            left = mid
    return round(right, 5)

# Tests
tests = [
    ("sqrt", sqrt_binary_search(8), 2),
    ("sqrt_perfect", sqrt_binary_search(16), 4),
    ("peak_2d", find_peak_element_2d([[1,4],[3,2]]) in [[0,1], [1,0]], True),
    ("search_matrix", search_matrix_ii([[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], 5), True),
    ("duplicate", find_duplicate_number([1,3,4,2,2]), 2),
    ("single_sorted", single_element_sorted([1,1,2,3,3,4,4,8,8]), 2),
    ("min_rotated_dup", find_min_rotated_ii([2,2,2,0,1]), 0),
    ("search_rotated_dup", search_rotated_ii([2,5,6,0,0,1,2], 0), True),
    ("count_neg", count_negatives([[4,3,2,-1],[3,2,1,-1],[1,1,-1,-2],[-1,-1,-2,-3]]), 8),
    ("kth_positive", find_kth_positive([2,3,4,7,11], 5), 9),
    ("split_array", split_array_largest_sum([7,2,5,10,8], 2), 18),
    ("gas_stations", minimize_max_distance_to_gas([1,2,3,4,5,6,7,8,9,10], 9), 0.5),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
