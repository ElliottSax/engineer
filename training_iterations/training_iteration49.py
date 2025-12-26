def merge_sort(arr):
    """Classic merge sort implementation."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def count_inversions(arr):
    """Count inversions using merge sort."""
    def merge_count(arr):
        if len(arr) <= 1:
            return arr, 0
        mid = len(arr) // 2
        left, left_inv = merge_count(arr[:mid])
        right, right_inv = merge_count(arr[mid:])
        merged, split_inv = merge_and_count(left, right)
        return merged, left_inv + right_inv + split_inv

    def merge_and_count(left, right):
        result = []
        inversions = 0
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                inversions += len(left) - i
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result, inversions

    _, count = merge_count(arr[:])
    return count

def reverse_pairs(nums):
    """Count pairs (i,j) where i<j and nums[i]>2*nums[j]."""
    def merge_count(arr, start, end):
        if end - start <= 1:
            return 0
        mid = (start + end) // 2
        count = merge_count(arr, start, mid) + merge_count(arr, mid, end)

        j = mid
        for i in range(start, mid):
            while j < end and arr[i] > 2 * arr[j]:
                j += 1
            count += j - mid

        arr[start:end] = sorted(arr[start:end])
        return count

    return merge_count(nums[:], 0, len(nums))

def merge_k_sorted_lists(lists):
    """Merge k sorted lists using divide and conquer."""
    if not lists:
        return []

    def merge_two(l1, l2):
        result = []
        i = j = 0
        while i < len(l1) and j < len(l2):
            if l1[i] <= l2[j]:
                result.append(l1[i])
                i += 1
            else:
                result.append(l2[j])
                j += 1
        result.extend(l1[i:])
        result.extend(l2[j:])
        return result

    while len(lists) > 1:
        merged = []
        for i in range(0, len(lists), 2):
            if i + 1 < len(lists):
                merged.append(merge_two(lists[i], lists[i+1]))
            else:
                merged.append(lists[i])
        lists = merged
    return lists[0]

def median_of_two_sorted(nums1, nums2):
    """Find median of two sorted arrays in O(log(m+n))."""
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    m, n = len(nums1), len(nums2)
    left, right = 0, m

    while left <= right:
        i = (left + right) // 2
        j = (m + n + 1) // 2 - i

        max_left1 = float('-inf') if i == 0 else nums1[i-1]
        min_right1 = float('inf') if i == m else nums1[i]
        max_left2 = float('-inf') if j == 0 else nums2[j-1]
        min_right2 = float('inf') if j == n else nums2[j]

        if max_left1 <= min_right2 and max_left2 <= min_right1:
            if (m + n) % 2 == 0:
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2
            return max(max_left1, max_left2)
        elif max_left1 > min_right2:
            right = i - 1
        else:
            left = i + 1

    return 0

def kth_smallest_in_sorted_matrix(matrix, k):
    """Kth smallest using binary search on value range."""
    n = len(matrix)

    def count_less_equal(mid):
        count = 0
        row, col = n - 1, 0
        while row >= 0 and col < n:
            if matrix[row][col] <= mid:
                count += row + 1
                col += 1
            else:
                row -= 1
        return count

    left, right = matrix[0][0], matrix[-1][-1]
    while left < right:
        mid = (left + right) // 2
        if count_less_equal(mid) < k:
            left = mid + 1
        else:
            right = mid
    return left

def sort_list_merge(head):
    """Sort linked list using merge sort."""
    if not head or not head.get('next'):
        return head

    # Find middle
    slow = head
    fast = head['next']
    while fast and fast.get('next'):
        slow = slow['next']
        fast = fast['next']['next']

    mid = slow['next']
    slow['next'] = None

    left = sort_list_merge(head)
    right = sort_list_merge(mid)

    # Merge
    dummy = {'next': None}
    curr = dummy
    while left and right:
        if left['val'] <= right['val']:
            curr['next'] = left
            left = left.get('next')
        else:
            curr['next'] = right
            right = right.get('next')
        curr = curr['next']
    curr['next'] = left or right

    return dummy['next']

def count_range_sum(nums, lower, upper):
    """Count range sums within [lower, upper]."""
    prefix = [0]
    for num in nums:
        prefix.append(prefix[-1] + num)

    def merge_count(arr, start, end):
        if end - start <= 1:
            return 0
        mid = (start + end) // 2
        count = merge_count(arr, start, mid) + merge_count(arr, mid, end)

        j = k = mid
        for i in range(start, mid):
            while j < end and arr[j] - arr[i] < lower:
                j += 1
            while k < end and arr[k] - arr[i] <= upper:
                k += 1
            count += k - j

        arr[start:end] = sorted(arr[start:end])
        return count

    return merge_count(prefix, 0, len(prefix))

def merge_intervals(intervals):
    """Merge overlapping intervals."""
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    result = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result

def interval_intersection(A, B):
    """Find intersection of two interval lists."""
    result = []
    i = j = 0
    while i < len(A) and j < len(B):
        lo = max(A[i][0], B[j][0])
        hi = min(A[i][1], B[j][1])
        if lo <= hi:
            result.append([lo, hi])
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1
    return result

# Tests
tests = [
    ("merge_sort", merge_sort([3,1,4,1,5,9,2,6]), [1,1,2,3,4,5,6,9]),
    ("inversions", count_inversions([2,4,1,3,5]), 3),
    ("inversions_sorted", count_inversions([1,2,3,4,5]), 0),
    ("inversions_reverse", count_inversions([5,4,3,2,1]), 10),
    ("reverse_pairs", reverse_pairs([1,3,2,3,1]), 2),
    ("merge_k_lists", merge_k_sorted_lists([[1,4,5],[1,3,4],[2,6]]), [1,1,2,3,4,4,5,6]),
    ("median_two", median_of_two_sorted([1,3], [2]), 2.0),
    ("median_two_2", median_of_two_sorted([1,2], [3,4]), 2.5),
    ("kth_matrix", kth_smallest_in_sorted_matrix([[1,5,9],[10,11,13],[12,13,15]], 8), 13),
    ("count_range", count_range_sum([-2,5,-1], -2, 2), 3),
    ("merge_intervals", merge_intervals([[1,3],[2,6],[8,10],[15,18]]), [[1,6],[8,10],[15,18]]),
    ("interval_inter", interval_intersection([[0,2],[5,10],[13,23],[24,25]], [[1,5],[8,12],[15,24],[25,26]]),
     [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]),
]

# Linked list sort test
def to_list(head):
    result = []
    while head:
        result.append(head['val'])
        head = head.get('next')
    return result

ll = {'val': 4, 'next': {'val': 2, 'next': {'val': 1, 'next': {'val': 3, 'next': None}}}}
sorted_ll = sort_list_merge(ll)
tests.append(("sort_list", to_list(sorted_ll), [1, 2, 3, 4]))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
