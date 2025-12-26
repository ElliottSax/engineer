# EXTREME: Interval & Sweep Line Problems

from collections import defaultdict
import heapq

# HARD: Meeting Rooms II - Minimum Conference Rooms
def min_meeting_rooms(intervals):
    """Minimum meeting rooms required."""
    if not intervals:
        return 0
    events = []
    for start, end in intervals:
        events.append((start, 1))  # Start
        events.append((end, -1))   # End
    events.sort()
    rooms = max_rooms = 0
    for _, delta in events:
        rooms += delta
        max_rooms = max(max_rooms, rooms)
    return max_rooms

# HARD: Insert Interval
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

# HARD: Interval List Intersections
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

# HARD: Employee Free Time
def employee_free_time(schedules):
    """Find common free time for all employees."""
    # Flatten and sort all intervals
    all_intervals = []
    for schedule in schedules:
        all_intervals.extend(schedule)
    all_intervals.sort()

    # Merge intervals
    merged = []
    for interval in all_intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval[:])
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])

    # Find gaps
    free_time = []
    for i in range(1, len(merged)):
        free_time.append([merged[i-1][1], merged[i][0]])

    return free_time

# HARD: Data Stream as Disjoint Intervals
class SummaryRanges:
    def __init__(self):
        self.intervals = []

    def add_num(self, val):
        intervals = self.intervals
        new_interval = [val, val]

        # Binary search for insertion point
        left, right = 0, len(intervals)
        while left < right:
            mid = (left + right) // 2
            if intervals[mid][0] < val:
                left = mid + 1
            else:
                right = mid

        # Check for merge with previous
        if left > 0 and intervals[left - 1][1] >= val - 1:
            new_interval[0] = intervals[left - 1][0]
            new_interval[1] = max(new_interval[1], intervals[left - 1][1])
            left -= 1

        # Check for merge with next
        right_bound = left
        while right_bound < len(intervals) and intervals[right_bound][0] <= new_interval[1] + 1:
            new_interval[1] = max(new_interval[1], intervals[right_bound][1])
            right_bound += 1

        self.intervals = intervals[:left] + [new_interval] + intervals[right_bound:]

    def get_intervals(self):
        return self.intervals

# HARD: Remove Covered Intervals
def remove_covered_intervals(intervals):
    """Count non-covered intervals."""
    # Sort by start ascending, then by end descending
    intervals.sort(key=lambda x: (x[0], -x[1]))
    count = 0
    max_end = 0
    for start, end in intervals:
        if end > max_end:
            count += 1
            max_end = end
    return count

# HARD: Maximum CPU Load
def max_cpu_load(jobs):
    """Maximum CPU load at any time."""
    events = []
    for start, end, load in jobs:
        events.append((start, load))
        events.append((end, -load))
    events.sort(key=lambda x: (x[0], -x[1]))
    current_load = max_load = 0
    for _, load in events:
        current_load += load
        max_load = max(max_load, current_load)
    return max_load

# HARD: Minimum Interval to Include Each Query
def min_interval(intervals, queries):
    """For each query, find smallest interval containing it."""
    sorted_queries = sorted(enumerate(queries), key=lambda x: x[1])
    intervals = sorted(intervals)
    result = [-1] * len(queries)

    heap = []  # (size, end)
    j = 0

    for idx, q in sorted_queries:
        # Add all intervals starting <= q
        while j < len(intervals) and intervals[j][0] <= q:
            start, end = intervals[j]
            if end >= q:
                heapq.heappush(heap, (end - start + 1, end))
            j += 1

        # Remove intervals that ended before q
        while heap and heap[0][1] < q:
            heapq.heappop(heap)

        if heap:
            result[idx] = heap[0][0]

    return result

# HARD: Count Integers in Intervals
class CountIntervals:
    def __init__(self):
        self.intervals = []
        self.count = 0

    def add(self, left, right):
        new_interval = [left, right]
        merged = []
        added = False

        for interval in self.intervals:
            if interval[1] < new_interval[0]:
                merged.append(interval)
            elif new_interval[1] < interval[0]:
                if not added:
                    merged.append(new_interval)
                    added = True
                merged.append(interval)
            else:
                new_interval = [min(new_interval[0], interval[0]),
                              max(new_interval[1], interval[1])]

        if not added:
            merged.append(new_interval)

        self.intervals = merged
        self.count = sum(end - start + 1 for start, end in self.intervals)

    def count_fn(self):
        return self.count

# Tests
tests = []

# Meeting Rooms
tests.append(("rooms", min_meeting_rooms([[0,30],[5,10],[15,20]]), 2))
tests.append(("rooms2", min_meeting_rooms([[7,10],[2,4]]), 1))

# Insert Interval
tests.append(("insert", insert_interval([[1,3],[6,9]], [2,5]), [[1,5],[6,9]]))
tests.append(("insert2", insert_interval([[1,2],[3,5],[6,7],[8,10],[12,16]], [4,8]),
              [[1,2],[3,10],[12,16]]))

# Interval Intersection
tests.append(("intersect", interval_intersection([[0,2],[5,10],[13,23],[24,25]],
              [[1,5],[8,12],[15,24],[25,26]]), [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]))

# Employee Free Time
tests.append(("free_time", employee_free_time([[[1,3],[6,7]],[[2,4]],[[2,5],[9,12]]]),
              [[5,6],[7,9]]))

# Summary Ranges
sr = SummaryRanges()
sr.add_num(1)
sr.add_num(3)
sr.add_num(7)
sr.add_num(2)
tests.append(("summary", sr.get_intervals(), [[1,3],[7,7]]))

# Remove Covered
tests.append(("covered", remove_covered_intervals([[1,4],[3,6],[2,8]]), 2))

# Max CPU Load
tests.append(("cpu", max_cpu_load([[1,4,3],[2,5,4],[7,9,6]]), 7))

# Min Interval
tests.append(("min_int", min_interval([[1,4],[2,4],[3,6],[4,4]], [2,3,4,5]), [3,3,1,4]))

# Count Intervals
ci = CountIntervals()
ci.add(2, 3)
ci.add(7, 10)
tests.append(("count_int", ci.count_fn(), 6))
ci.add(5, 8)
tests.append(("count_int2", ci.count_fn(), 8))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
