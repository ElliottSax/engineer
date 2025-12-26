def insert_interval(intervals, new_interval):
    """Insert and merge new interval."""
    result = []
    i = 0
    n = len(intervals)

    # Add all intervals before new_interval
    while i < n and intervals[i][1] < new_interval[0]:
        result.append(intervals[i])
        i += 1

    # Merge overlapping intervals
    while i < n and intervals[i][0] <= new_interval[1]:
        new_interval = [min(new_interval[0], intervals[i][0]),
                       max(new_interval[1], intervals[i][1])]
        i += 1
    result.append(new_interval)

    # Add remaining intervals
    while i < n:
        result.append(intervals[i])
        i += 1

    return result

def meeting_rooms_ii(intervals):
    """Minimum meeting rooms needed."""
    if not intervals:
        return 0
    events = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))
    events.sort()

    rooms = max_rooms = 0
    for _, delta in events:
        rooms += delta
        max_rooms = max(max_rooms, rooms)
    return max_rooms

def employee_free_time(schedules):
    """Find common free time across all employees."""
    all_intervals = []
    for schedule in schedules:
        all_intervals.extend(schedule)
    all_intervals.sort()

    result = []
    prev_end = all_intervals[0][1]
    for start, end in all_intervals[1:]:
        if start > prev_end:
            result.append([prev_end, start])
        prev_end = max(prev_end, end)

    return result

def remove_covered_intervals(intervals):
    """Count intervals not covered by another."""
    intervals.sort(key=lambda x: (x[0], -x[1]))
    count = 0
    prev_end = 0

    for _, end in intervals:
        if end > prev_end:
            count += 1
            prev_end = end

    return count

def interval_list_intersections(A, B):
    """Find all intersections of two interval lists."""
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

def minimum_interval_to_include_query(intervals, queries):
    """Smallest interval containing each query point."""
    import heapq
    intervals = sorted(intervals)
    indexed_queries = sorted(enumerate(queries), key=lambda x: x[1])
    result = [-1] * len(queries)
    heap = []
    i = 0

    for idx, q in indexed_queries:
        while i < len(intervals) and intervals[i][0] <= q:
            left, right = intervals[i]
            heapq.heappush(heap, (right - left + 1, right))
            i += 1

        while heap and heap[0][1] < q:
            heapq.heappop(heap)

        if heap:
            result[idx] = heap[0][0]

    return result

def max_non_overlapping_intervals(intervals):
    """Maximum number of non-overlapping intervals."""
    if not intervals:
        return 0
    intervals.sort(key=lambda x: x[1])
    count = 1
    end = intervals[0][1]

    for i in range(1, len(intervals)):
        if intervals[i][0] >= end:
            count += 1
            end = intervals[i][1]

    return count

def min_arrows_burst_balloons(points):
    """Minimum arrows to burst all balloons."""
    if not points:
        return 0
    points.sort(key=lambda x: x[1])
    arrows = 1
    end = points[0][1]

    for i in range(1, len(points)):
        if points[i][0] > end:
            arrows += 1
            end = points[i][1]

    return arrows

def non_overlapping_interval_removal(intervals):
    """Minimum removals for non-overlapping intervals."""
    if not intervals:
        return 0
    return len(intervals) - max_non_overlapping_intervals(intervals)

def car_pooling(trips, capacity):
    """Check if car can complete all trips."""
    events = []
    for passengers, start, end in trips:
        events.append((start, passengers))
        events.append((end, -passengers))
    events.sort()

    current = 0
    for _, delta in events:
        current += delta
        if current > capacity:
            return False
    return True

def my_calendar():
    """Calendar booking without double booking."""
    bookings = []

    def book(start, end):
        for s, e in bookings:
            if start < e and end > s:
                return False
        bookings.append((start, end))
        return True

    return book

def my_calendar_ii():
    """Calendar allowing at most double booking."""
    bookings = []
    overlaps = []

    def book(start, end):
        for s, e in overlaps:
            if start < e and end > s:
                return False
        for s, e in bookings:
            if start < e and end > s:
                overlaps.append((max(start, s), min(end, e)))
        bookings.append((start, end))
        return True

    return book

def range_module():
    """Track ranges that are being tracked."""
    ranges = []

    def add_range(left, right):
        nonlocal ranges
        new_ranges = []
        i = 0
        while i < len(ranges) and ranges[i][1] < left:
            new_ranges.append(ranges[i])
            i += 1
        while i < len(ranges) and ranges[i][0] <= right:
            left = min(left, ranges[i][0])
            right = max(right, ranges[i][1])
            i += 1
        new_ranges.append([left, right])
        while i < len(ranges):
            new_ranges.append(ranges[i])
            i += 1
        ranges = new_ranges

    def query_range(left, right):
        for s, e in ranges:
            if s <= left and right <= e:
                return True
        return False

    return add_range, query_range

# Tests
tests = [
    ("insert", insert_interval([[1,3],[6,9]], [2,5]), [[1,5],[6,9]]),
    ("insert_2", insert_interval([[1,2],[3,5],[6,7],[8,10],[12,16]], [4,8]),
     [[1,2],[3,10],[12,16]]),
    ("meeting_rooms", meeting_rooms_ii([[0,30],[5,10],[15,20]]), 2),
    ("free_time", employee_free_time([[[1,2],[5,6]],[[1,3]],[[4,10]]]), [[3,4]]),
    ("remove_covered", remove_covered_intervals([[1,4],[3,6],[2,8]]), 2),
    ("interval_inter", interval_list_intersections([[0,2],[5,10]],[[1,5],[8,12]]),
     [[1,2],[5,5],[8,10]]),
    ("min_interval", minimum_interval_to_include_query([[1,4],[2,4],[3,6],[4,4]], [2,3,4,5]),
     [3, 3, 1, 4]),
    ("max_non_overlap", max_non_overlapping_intervals([[1,2],[2,3],[3,4],[1,3]]), 3),
    ("min_arrows", min_arrows_burst_balloons([[10,16],[2,8],[1,6],[7,12]]), 2),
    ("removal", non_overlapping_interval_removal([[1,2],[2,3],[3,4],[1,3]]), 1),
    ("carpool_true", car_pooling([[2,1,5],[3,3,7]], 4), False),
    ("carpool_false", car_pooling([[2,1,5],[3,5,7]], 3), True),
]

# Calendar test
book = my_calendar()
tests.append(("calendar_1", book(10, 20), True))
tests.append(("calendar_2", book(15, 25), False))
tests.append(("calendar_3", book(20, 30), True))

# Calendar II test
book2 = my_calendar_ii()
tests.append(("calendar2_1", book2(10, 20), True))
tests.append(("calendar2_2", book2(50, 60), True))
tests.append(("calendar2_3", book2(10, 40), True))
tests.append(("calendar2_4", book2(5, 15), False))

# Range module test
add_r, query_r = range_module()
add_r(10, 20)
tests.append(("range_1", query_r(14, 16), True))
tests.append(("range_2", query_r(13, 25), False))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
