def k_closest_points(points, k):
    """K closest points to origin."""
    import heapq
    return heapq.nsmallest(k, points, key=lambda p: p[0]**2 + p[1]**2)

def top_k_frequent(nums, k):
    """K most frequent elements."""
    from collections import Counter
    import heapq
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)

def top_k_frequent_words(words, k):
    """K most frequent words (tie-break alphabetically)."""
    from collections import Counter
    import heapq
    count = Counter(words)
    return heapq.nsmallest(k, count.keys(), key=lambda w: (-count[w], w))

def kth_largest_in_stream():
    """Kth largest element in a stream."""
    import heapq

    def init(k, nums):
        heap = nums[:k]
        heapq.heapify(heap)
        for num in nums[k:]:
            if num > heap[0]:
                heapq.heapreplace(heap, num)
        return heap, k

    def add(heap, k, val):
        if len(heap) < k:
            heapq.heappush(heap, val)
        elif val > heap[0]:
            heapq.heapreplace(heap, val)
        return heap[0]

    return init, add

def find_k_pairs_smallest_sums(nums1, nums2, k):
    """K pairs with smallest sums from two arrays."""
    import heapq
    if not nums1 or not nums2:
        return []
    heap = [(nums1[0] + nums2[j], 0, j) for j in range(min(k, len(nums2)))]
    heapq.heapify(heap)
    result = []
    while heap and len(result) < k:
        _, i, j = heapq.heappop(heap)
        result.append([nums1[i], nums2[j]])
        if i + 1 < len(nums1):
            heapq.heappush(heap, (nums1[i + 1] + nums2[j], i + 1, j))
    return result

def kth_smallest_in_matrix(matrix, k):
    """Kth smallest element in sorted matrix."""
    import heapq
    n = len(matrix)
    heap = [(matrix[0][j], 0, j) for j in range(min(k, n))]
    heapq.heapify(heap)
    for _ in range(k):
        val, i, j = heapq.heappop(heap)
        if i + 1 < n:
            heapq.heappush(heap, (matrix[i + 1][j], i + 1, j))
    return val

def reorganize_string(s):
    """Reorganize string so no two adjacent chars are same."""
    from collections import Counter
    import heapq
    count = Counter(s)
    heap = [(-cnt, c) for c, cnt in count.items()]
    heapq.heapify(heap)
    result = []
    prev_cnt, prev_c = 0, ''
    while heap:
        cnt, c = heapq.heappop(heap)
        result.append(c)
        if prev_cnt < 0:
            heapq.heappush(heap, (prev_cnt, prev_c))
        prev_cnt, prev_c = cnt + 1, c
    result = ''.join(result)
    return result if len(result) == len(s) else ""

def task_scheduler_with_heap(tasks, n):
    """Minimum intervals to finish tasks with cooldown."""
    from collections import Counter
    import heapq
    count = Counter(tasks)
    heap = [-cnt for cnt in count.values()]
    heapq.heapify(heap)
    time = 0
    while heap:
        temp = []
        for _ in range(n + 1):
            if heap:
                cnt = heapq.heappop(heap) + 1
                if cnt < 0:
                    temp.append(cnt)
            time += 1
            if not heap and not temp:
                break
        for cnt in temp:
            heapq.heappush(heap, cnt)
    return time

def furthest_building(heights, bricks, ladders):
    """Furthest building you can reach."""
    import heapq
    heap = []  # min heap of ladder climbs
    for i in range(len(heights) - 1):
        diff = heights[i + 1] - heights[i]
        if diff > 0:
            heapq.heappush(heap, diff)
            if len(heap) > ladders:
                bricks -= heapq.heappop(heap)
            if bricks < 0:
                return i
    return len(heights) - 1

def ipo(k, w, profits, capital):
    """Maximize capital after k projects."""
    import heapq
    n = len(profits)
    projects = sorted(zip(capital, profits))
    idx = 0
    heap = []
    for _ in range(k):
        while idx < n and projects[idx][0] <= w:
            heapq.heappush(heap, -projects[idx][1])
            idx += 1
        if not heap:
            break
        w -= heapq.heappop(heap)
    return w

def minimum_cost_connect_sticks(sticks):
    """Minimum cost to connect all sticks."""
    import heapq
    heapq.heapify(sticks)
    cost = 0
    while len(sticks) > 1:
        first = heapq.heappop(sticks)
        second = heapq.heappop(sticks)
        combined = first + second
        cost += combined
        heapq.heappush(sticks, combined)
    return cost

# Tests
tests = [
    ("k_closest", sorted(k_closest_points([[1,3],[-2,2]], 1)), sorted([[-2,2]])),
    ("top_k_freq", sorted(top_k_frequent([1,1,1,2,2,3], 2)), [1, 2]),
    ("top_k_words", top_k_frequent_words(["i","love","leetcode","i","love","coding"], 2), ["i", "love"]),
    ("k_pairs", find_k_pairs_smallest_sums([1,7,11], [2,4,6], 3), [[1,2],[1,4],[1,6]]),
    ("kth_matrix", kth_smallest_in_matrix([[1,5,9],[10,11,13],[12,13,15]], 8), 13),
    ("reorganize", reorganize_string("aab"), "aba"),
    ("reorganize_no", reorganize_string("aaab"), ""),
    ("task_scheduler", task_scheduler_with_heap(["A","A","A","B","B","B"], 2), 8),
    ("furthest_bldg", furthest_building([4,2,7,6,9,14,12], 5, 1), 4),
    ("ipo", ipo(2, 0, [1,2,3], [0,1,1]), 4),
    ("connect_sticks", minimum_cost_connect_sticks([2,4,3]), 14),
]

# Kth largest stream test
init_stream, add_stream = kth_largest_in_stream()
heap, k = init_stream(3, [4,5,8,2])
tests.append(("kth_stream_init", heap[0], 4))
result = add_stream(heap, k, 3)
tests.append(("kth_stream_add", result, 4))
result = add_stream(heap, k, 5)
tests.append(("kth_stream_add_2", result, 5))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
