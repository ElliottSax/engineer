#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ULTRA TRAINING ITERATION 129                              ║
║                    Greedy Algorithms Collection                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import heapq
from collections import Counter

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 1: Activity Selection (Interval Scheduling)
# ═══════════════════════════════════════════════════════════════════════════════
def activity_selection(activities):
    """Select maximum non-overlapping activities."""
    sorted_acts = sorted(enumerate(activities), key=lambda x: x[1][1])

    result = []
    last_end = float('-inf')

    for idx, (start, end) in sorted_acts:
        if start >= last_end:
            result.append(idx)
            last_end = end

    return result

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 2: Huffman Coding
# ═══════════════════════════════════════════════════════════════════════════════
def huffman_coding(freq):
    """Build Huffman tree and return codes."""
    if not freq:
        return {}

    if len(freq) == 1:
        return {list(freq.keys())[0]: '0'}

    heap = [[f, [c, '']] for c, f in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)

        for item in lo[1:]:
            item[1] = '0' + item[1]
        for item in hi[1:]:
            item[1] = '1' + item[1]

        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    codes = {}
    for item in heap[0][1:]:
        codes[item[0]] = item[1]

    return codes

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 3: Fractional Knapsack
# ═══════════════════════════════════════════════════════════════════════════════
def fractional_knapsack(items, capacity):
    """Fractional knapsack: items = [(value, weight), ...]"""
    # Sort by value/weight ratio
    sorted_items = sorted(enumerate(items),
                         key=lambda x: x[1][0] / x[1][1] if x[1][1] > 0 else float('inf'),
                         reverse=True)

    total_value = 0
    fractions = {}

    for idx, (value, weight) in sorted_items:
        if capacity <= 0:
            break

        if weight <= capacity:
            fractions[idx] = 1.0
            total_value += value
            capacity -= weight
        else:
            fraction = capacity / weight
            fractions[idx] = fraction
            total_value += value * fraction
            capacity = 0

    return total_value, fractions

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 4: Job Scheduling with Deadlines
# ═══════════════════════════════════════════════════════════════════════════════
def job_scheduling(jobs):
    """Schedule jobs with deadlines to maximize profit.
    jobs = [(id, deadline, profit), ...]"""
    if not jobs:
        return [], 0

    # Sort by profit descending
    sorted_jobs = sorted(jobs, key=lambda x: x[2], reverse=True)

    max_deadline = max(j[1] for j in jobs)
    slots = [None] * (max_deadline + 1)

    total_profit = 0
    scheduled = []

    for job_id, deadline, profit in sorted_jobs:
        # Find latest available slot
        for t in range(min(deadline, max_deadline), 0, -1):
            if slots[t] is None:
                slots[t] = job_id
                total_profit += profit
                scheduled.append(job_id)
                break

    return scheduled, total_profit

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 5: Minimum Coins (Greedy - works for canonical systems)
# ═══════════════════════════════════════════════════════════════════════════════
def min_coins_greedy(coins, amount):
    """Minimum coins using greedy (works for standard coin systems)."""
    coins = sorted(coins, reverse=True)
    result = []

    for coin in coins:
        while amount >= coin:
            result.append(coin)
            amount -= coin

    return result if amount == 0 else None

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 6: Gas Station Circuit
# ═══════════════════════════════════════════════════════════════════════════════
def can_complete_circuit(gas, cost):
    """Find starting gas station to complete circular route."""
    n = len(gas)
    total_tank = 0
    curr_tank = 0
    start = 0

    for i in range(n):
        total_tank += gas[i] - cost[i]
        curr_tank += gas[i] - cost[i]

        if curr_tank < 0:
            start = i + 1
            curr_tank = 0

    return start if total_tank >= 0 else -1

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 7: Jump Game
# ═══════════════════════════════════════════════════════════════════════════════
def can_jump(nums):
    """Can reach last index?"""
    max_reach = 0
    for i, jump in enumerate(nums):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + jump)
    return True

def min_jumps(nums):
    """Minimum jumps to reach end."""
    if len(nums) <= 1:
        return 0

    jumps = 0
    curr_end = 0
    curr_farthest = 0

    for i in range(len(nums) - 1):
        curr_farthest = max(curr_farthest, i + nums[i])

        if i == curr_end:
            jumps += 1
            curr_end = curr_farthest

            if curr_end >= len(nums) - 1:
                break

    return jumps

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 8: Partition Labels
# ═══════════════════════════════════════════════════════════════════════════════
def partition_labels(s):
    """Partition string so each letter appears in at most one part."""
    last_occurrence = {c: i for i, c in enumerate(s)}

    partitions = []
    start = 0
    end = 0

    for i, c in enumerate(s):
        end = max(end, last_occurrence[c])
        if i == end:
            partitions.append(end - start + 1)
            start = i + 1

    return partitions

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 9: Task Scheduler
# ═══════════════════════════════════════════════════════════════════════════════
def least_interval(tasks, n):
    """Minimum intervals to complete all tasks with cooldown n."""
    freq = Counter(tasks)
    max_freq = max(freq.values())
    max_count = sum(1 for f in freq.values() if f == max_freq)

    # Formula: (max_freq - 1) * (n + 1) + max_count
    result = (max_freq - 1) * (n + 1) + max_count

    return max(result, len(tasks))

# ═══════════════════════════════════════════════════════════════════════════════
#                              TEST SUITE
# ═══════════════════════════════════════════════════════════════════════════════
def run_tests():
    tests = []

    # Test 1: Activity selection
    activities = [(1, 4), (3, 5), (0, 6), (5, 7), (8, 9), (5, 9)]
    selected = activity_selection(activities)
    tests.append(("activity", len(selected), 3))

    # Test 2: Huffman coding
    freq = {'a': 5, 'b': 9, 'c': 12, 'd': 13, 'e': 16, 'f': 45}
    codes = huffman_coding(freq)
    tests.append(("huffman_count", len(codes), 6))
    tests.append(("huffman_f", len(codes['f']), 1))  # Most frequent = shortest code

    # Test 3: Fractional knapsack
    items = [(60, 10), (100, 20), (120, 30)]  # (value, weight)
    value, _ = fractional_knapsack(items, 50)
    tests.append(("frac_knap", value, 240.0))

    # Test 4: Job scheduling
    jobs = [('a', 2, 100), ('b', 1, 19), ('c', 2, 27), ('d', 1, 25), ('e', 3, 15)]
    scheduled, profit = job_scheduling(jobs)
    tests.append(("job_profit", profit, 142))  # 100 + 27 + 15

    # Test 5: Minimum coins
    coins = min_coins_greedy([25, 10, 5, 1], 41)
    tests.append(("min_coins", coins, [25, 10, 5, 1]))

    # Test 6: Gas station
    tests.append(("gas_station", can_complete_circuit([1, 2, 3, 4, 5], [3, 4, 5, 1, 2]), 3))

    # Test 7: Jump game
    tests.append(("can_jump_yes", can_jump([2, 3, 1, 1, 4]), True))
    tests.append(("can_jump_no", can_jump([3, 2, 1, 0, 4]), False))
    tests.append(("min_jumps", min_jumps([2, 3, 1, 1, 4]), 2))

    # Test 8: Partition labels
    tests.append(("partition", partition_labels("ababcbacadefegdehijhklij"), [9, 7, 8]))

    # Test 9: Task scheduler
    tests.append(("scheduler", least_interval(['A', 'A', 'A', 'B', 'B', 'B'], 2), 8))

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
