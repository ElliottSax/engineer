#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ULTRA TRAINING ITERATION 123                              ║
║                   Advanced Interval & Sweep Line                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import heapq
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 1: Interval Scheduling Maximization
# ═══════════════════════════════════════════════════════════════════════════════
def max_non_overlapping_intervals(intervals):
    """Select maximum number of non-overlapping intervals (greedy by end time)."""
    if not intervals:
        return []

    # Sort by end time
    sorted_intervals = sorted(enumerate(intervals), key=lambda x: x[1][1])

    result = []
    last_end = float('-inf')

    for idx, (start, end) in sorted_intervals:
        if start >= last_end:
            result.append(idx)
            last_end = end

    return result

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 2: Weighted Interval Scheduling (DP)
# ═══════════════════════════════════════════════════════════════════════════════
def weighted_interval_scheduling(intervals, weights):
    """Select non-overlapping intervals maximizing total weight."""
    n = len(intervals)
    if n == 0:
        return 0, []

    # Sort by end time
    indexed = sorted(range(n), key=lambda i: intervals[i][1])

    # For each interval, find latest non-conflicting interval
    def find_prev(i):
        lo, hi = 0, i - 1
        target_end = intervals[indexed[i]][0]
        result = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if intervals[indexed[mid]][1] <= target_end:
                result = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return result

    # DP
    dp = [0] * (n + 1)
    choice = [False] * n  # True if we include interval i

    for i in range(n):
        prev = find_prev(i)
        include = weights[indexed[i]] + (dp[prev + 1] if prev >= 0 else 0)
        exclude = dp[i]

        if include > exclude:
            dp[i + 1] = include
            choice[i] = True
        else:
            dp[i + 1] = exclude

    # Backtrack
    selected = []
    i = n - 1
    while i >= 0:
        if choice[i]:
            selected.append(indexed[i])
            i = find_prev(i)
        else:
            i -= 1

    return dp[n], selected[::-1]

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 3: Sweep Line - Maximum Overlapping Intervals
# ═══════════════════════════════════════════════════════════════════════════════
def max_overlapping_count(intervals):
    """Find maximum number of intervals overlapping at any point."""
    events = []
    for start, end in intervals:
        events.append((start, 0))   # 0 = start (process starts before ends)
        events.append((end, 1))     # 1 = end

    events.sort()

    max_overlap = 0
    current = 0

    for _, event_type in events:
        if event_type == 0:
            current += 1
            max_overlap = max(max_overlap, current)
        else:
            current -= 1

    return max_overlap

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 4: Merge Overlapping Intervals
# ═══════════════════════════════════════════════════════════════════════════════
def merge_intervals(intervals):
    """Merge all overlapping intervals."""
    if not intervals:
        return []

    sorted_intervals = sorted(intervals)
    merged = [list(sorted_intervals[0])]

    for start, end in sorted_intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [tuple(x) for x in merged]

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 5: Interval Covering (Minimum Intervals to Cover Range)
# ═══════════════════════════════════════════════════════════════════════════════
def min_intervals_to_cover(intervals, target_start, target_end):
    """Find minimum intervals needed to cover [target_start, target_end]."""
    if not intervals:
        return -1

    # Sort by start time
    sorted_intervals = sorted(intervals)

    count = 0
    current_end = target_start
    i = 0
    n = len(sorted_intervals)

    while current_end < target_end:
        # Find interval that starts <= current_end and extends furthest
        max_reach = current_end

        while i < n and sorted_intervals[i][0] <= current_end:
            max_reach = max(max_reach, sorted_intervals[i][1])
            i += 1

        if max_reach == current_end:
            return -1  # Cannot extend further

        current_end = max_reach
        count += 1

    return count

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 6: Skyline Problem
# ═══════════════════════════════════════════════════════════════════════════════
def get_skyline(buildings):
    """Compute skyline from list of buildings [left, right, height]."""
    if not buildings:
        return []

    # Create events: (x, type, height) where type: 0=start, 1=end
    events = []
    for left, right, height in buildings:
        events.append((left, 0, height))
        events.append((right, 1, height))

    # Sort: by x, then starts before ends, then by height (desc for starts)
    events.sort(key=lambda e: (e[0], e[1], -e[2] if e[1] == 0 else e[2]))

    result = []
    # Max heap of active heights (use negative for max heap)
    active = [0]
    height_count = defaultdict(int)
    height_count[0] = 1

    for x, event_type, height in events:
        if event_type == 0:  # Start
            heapq.heappush(active, -height)
            height_count[height] += 1
        else:  # End
            height_count[height] -= 1

        # Get current max height
        while active and height_count[-active[0]] == 0:
            heapq.heappop(active)

        current_max = -active[0] if active else 0

        if not result or result[-1][1] != current_max:
            result.append([x, current_max])

    return result

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 7: Meeting Rooms II (Minimum Rooms Needed)
# ═══════════════════════════════════════════════════════════════════════════════
def min_meeting_rooms(intervals):
    """Find minimum number of meeting rooms required."""
    if not intervals:
        return 0

    # Same as max overlapping count
    return max_overlapping_count(intervals)

# ═══════════════════════════════════════════════════════════════════════════════
#                              TEST SUITE
# ═══════════════════════════════════════════════════════════════════════════════
def run_tests():
    tests = []

    # Test 1: Max non-overlapping intervals
    intervals = [(1, 3), (2, 4), (3, 5), (0, 6), (5, 7), (8, 9)]
    selected = max_non_overlapping_intervals(intervals)
    tests.append(("non_overlap_count", len(selected), 3))

    # Test 2: Weighted interval scheduling
    intervals_w = [(0, 3), (1, 4), (2, 5), (3, 6), (4, 7)]
    weights = [3, 2, 4, 6, 2]
    max_weight, _ = weighted_interval_scheduling(intervals_w, weights)
    tests.append(("weighted", max_weight, 9))  # (0,3) + (3,6) = 3 + 6

    # Test 3: Max overlapping
    intervals_o = [(1, 4), (2, 5), (3, 6), (7, 9)]
    tests.append(("max_overlap", max_overlapping_count(intervals_o), 3))

    # Test 4: Merge intervals
    to_merge = [(1, 3), (2, 6), (8, 10), (15, 18)]
    merged = merge_intervals(to_merge)
    tests.append(("merge", merged, [(1, 6), (8, 10), (15, 18)]))

    # Test 5: Interval covering
    cover_intervals = [(0, 2), (1, 4), (2, 3), (3, 5), (4, 6)]
    tests.append(("cover", min_intervals_to_cover(cover_intervals, 0, 6), 2))

    # Test 6: Skyline
    buildings = [[2, 9, 10], [3, 7, 15], [5, 12, 12]]
    skyline = get_skyline(buildings)
    tests.append(("skyline_len", len(skyline) >= 4, True))

    # Test 7: Meeting rooms
    meetings = [(0, 30), (5, 10), (15, 20)]
    tests.append(("rooms", min_meeting_rooms(meetings), 2))

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
