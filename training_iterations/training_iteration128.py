#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ULTRA TRAINING ITERATION 128                              ║
║                   Divide & Conquer Masterclass                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 1: Merge Sort with Inversion Count
# ═══════════════════════════════════════════════════════════════════════════════
def merge_sort_inversions(arr):
    """Count inversions while sorting (pairs where i < j but a[i] > a[j])."""
    def merge_count(arr, temp, left, mid, right):
        i = left
        j = mid + 1
        k = left
        inv_count = 0

        while i <= mid and j <= right:
            if arr[i] <= arr[j]:
                temp[k] = arr[i]
                i += 1
            else:
                temp[k] = arr[j]
                inv_count += (mid - i + 1)
                j += 1
            k += 1

        while i <= mid:
            temp[k] = arr[i]
            i += 1
            k += 1

        while j <= right:
            temp[k] = arr[j]
            j += 1
            k += 1

        for i in range(left, right + 1):
            arr[i] = temp[i]

        return inv_count

    def sort_count(arr, temp, left, right):
        inv_count = 0
        if left < right:
            mid = (left + right) // 2
            inv_count += sort_count(arr, temp, left, mid)
            inv_count += sort_count(arr, temp, mid + 1, right)
            inv_count += merge_count(arr, temp, left, mid, right)
        return inv_count

    arr = arr[:]
    n = len(arr)
    temp = [0] * n
    return sort_count(arr, temp, 0, n - 1)

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 2: Quick Select (Kth Smallest)
# ═══════════════════════════════════════════════════════════════════════════════
def quick_select(arr, k):
    """Find k-th smallest element in O(n) average time."""
    arr = arr[:]

    def partition(left, right, pivot_idx):
        pivot = arr[pivot_idx]
        arr[pivot_idx], arr[right] = arr[right], arr[pivot_idx]
        store_idx = left

        for i in range(left, right):
            if arr[i] < pivot:
                arr[store_idx], arr[i] = arr[i], arr[store_idx]
                store_idx += 1

        arr[store_idx], arr[right] = arr[right], arr[store_idx]
        return store_idx

    def select(left, right, k):
        if left == right:
            return arr[left]

        pivot_idx = (left + right) // 2
        pivot_idx = partition(left, right, pivot_idx)

        if k == pivot_idx:
            return arr[k]
        elif k < pivot_idx:
            return select(left, pivot_idx - 1, k)
        else:
            return select(pivot_idx + 1, right, k)

    return select(0, len(arr) - 1, k - 1)

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 3: Closest Pair of Points
# ═══════════════════════════════════════════════════════════════════════════════
def closest_pair(points):
    """Find closest pair of points in O(n log n)."""
    import math

    def dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def closest_rec(px, py):
        n = len(px)
        if n <= 3:
            min_d = float('inf')
            for i in range(n):
                for j in range(i + 1, n):
                    min_d = min(min_d, dist(px[i], px[j]))
            return min_d

        mid = n // 2
        mid_x = px[mid][0]

        pyl = [p for p in py if p[0] <= mid_x]
        pyr = [p for p in py if p[0] > mid_x]

        dl = closest_rec(px[:mid], pyl)
        dr = closest_rec(px[mid:], pyr)
        d = min(dl, dr)

        strip = [p for p in py if abs(p[0] - mid_x) < d]

        for i in range(len(strip)):
            j = i + 1
            while j < len(strip) and strip[j][1] - strip[i][1] < d:
                d = min(d, dist(strip[i], strip[j]))
                j += 1

        return d

    if len(points) < 2:
        return float('inf')

    px = sorted(points, key=lambda p: p[0])
    py = sorted(points, key=lambda p: p[1])
    return closest_rec(px, py)

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 4: Maximum Subarray (Divide & Conquer)
# ═══════════════════════════════════════════════════════════════════════════════
def max_subarray_dc(nums):
    """Find maximum subarray sum using divide and conquer."""
    def max_crossing(nums, lo, mid, hi):
        left_sum = float('-inf')
        curr = 0
        for i in range(mid, lo - 1, -1):
            curr += nums[i]
            left_sum = max(left_sum, curr)

        right_sum = float('-inf')
        curr = 0
        for i in range(mid + 1, hi + 1):
            curr += nums[i]
            right_sum = max(right_sum, curr)

        return left_sum + right_sum

    def max_subarray(nums, lo, hi):
        if lo == hi:
            return nums[lo]

        mid = (lo + hi) // 2
        left = max_subarray(nums, lo, mid)
        right = max_subarray(nums, mid + 1, hi)
        cross = max_crossing(nums, lo, mid, hi)

        return max(left, right, cross)

    if not nums:
        return 0
    return max_subarray(nums, 0, len(nums) - 1)

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 5: Karatsuba Multiplication
# ═══════════════════════════════════════════════════════════════════════════════
def karatsuba(x, y):
    """Karatsuba multiplication algorithm."""
    if x < 10 or y < 10:
        return x * y

    n = max(len(str(x)), len(str(y)))
    m = n // 2

    power = 10 ** m

    a, b = divmod(x, power)
    c, d = divmod(y, power)

    ac = karatsuba(a, c)
    bd = karatsuba(b, d)
    ad_bc = karatsuba(a + b, c + d) - ac - bd

    return ac * (power ** 2) + ad_bc * power + bd

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 6: Strassen's Matrix Multiplication (2x2)
# ═══════════════════════════════════════════════════════════════════════════════
def strassen_multiply(A, B):
    """Strassen's algorithm for 2x2 matrices."""
    if len(A) != 2 or len(B) != 2:
        return None

    a, b, c, d = A[0][0], A[0][1], A[1][0], A[1][1]
    e, f, g, h = B[0][0], B[0][1], B[1][0], B[1][1]

    p1 = a * (f - h)
    p2 = (a + b) * h
    p3 = (c + d) * e
    p4 = d * (g - e)
    p5 = (a + d) * (e + h)
    p6 = (b - d) * (g + h)
    p7 = (a - c) * (e + f)

    return [
        [p5 + p4 - p2 + p6, p1 + p2],
        [p3 + p4, p1 + p5 - p3 - p7]
    ]

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 7: Binary Search Variants
# ═══════════════════════════════════════════════════════════════════════════════
def binary_search_first(arr, target):
    """Find first occurrence of target."""
    lo, hi = 0, len(arr) - 1
    result = -1

    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            result = mid
            hi = mid - 1
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1

    return result

def binary_search_last(arr, target):
    """Find last occurrence of target."""
    lo, hi = 0, len(arr) - 1
    result = -1

    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            result = mid
            lo = mid + 1
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1

    return result

def count_occurrences(arr, target):
    """Count occurrences of target."""
    first = binary_search_first(arr, target)
    if first == -1:
        return 0
    last = binary_search_last(arr, target)
    return last - first + 1

# ═══════════════════════════════════════════════════════════════════════════════
#                              TEST SUITE
# ═══════════════════════════════════════════════════════════════════════════════
def run_tests():
    tests = []

    # Test 1: Inversion count
    tests.append(("inversions", merge_sort_inversions([8, 4, 2, 1]), 6))
    tests.append(("inversions_sorted", merge_sort_inversions([1, 2, 3]), 0))

    # Test 2: Quick select
    tests.append(("quicksel_3", quick_select([3, 2, 1, 5, 4], 3), 3))
    tests.append(("quicksel_1", quick_select([7, 10, 4, 3, 20, 15], 1), 3))

    # Test 3: Closest pair
    points = [(2, 3), (12, 30), (40, 50), (5, 1), (12, 10), (3, 4)]
    cp = closest_pair(points)
    tests.append(("closest_pair", round(cp, 2), 1.41))

    # Test 4: Max subarray
    tests.append(("max_sub", max_subarray_dc([-2, 1, -3, 4, -1, 2, 1, -5, 4]), 6))

    # Test 5: Karatsuba
    tests.append(("karatsuba", karatsuba(1234, 5678), 1234 * 5678))
    tests.append(("karatsuba_2", karatsuba(12345678, 87654321), 12345678 * 87654321))

    # Test 6: Strassen
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    tests.append(("strassen", strassen_multiply(A, B), [[19, 22], [43, 50]]))

    # Test 7: Binary search variants
    arr = [1, 2, 2, 2, 3, 4, 5]
    tests.append(("bs_first", binary_search_first(arr, 2), 1))
    tests.append(("bs_last", binary_search_last(arr, 2), 3))
    tests.append(("count_occ", count_occurrences(arr, 2), 3))

    # Run all tests
    passed = 0
    print("\n" + "─" * 60)
    for name, result, expected in tests:
        if result == expected:
            passed += 1
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name}: got {result}, expected {expected}")

    print("─" * 60)
    print(f"\n  📊 Results: {passed}/{len(tests)} tests passed")
    return passed, len(tests)

if __name__ == "__main__":
    print(__doc__)
    passed, total = run_tests()
    if passed == total:
        print("\n  🎯 PERFECT SCORE!")
