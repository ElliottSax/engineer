def activity_selection(activities):
    """Maximum non-overlapping activities."""
    activities.sort(key=lambda x: x[1])
    count = 1
    end = activities[0][1]

    for i in range(1, len(activities)):
        if activities[i][0] >= end:
            count += 1
            end = activities[i][1]

    return count

def fractional_knapsack(weights, values, capacity):
    """Fractional knapsack - items can be divided."""
    n = len(weights)
    items = [(values[i] / weights[i], weights[i], values[i]) for i in range(n)]
    items.sort(reverse=True)

    total_value = 0
    remaining = capacity

    for ratio, weight, value in items:
        if remaining >= weight:
            total_value += value
            remaining -= weight
        else:
            total_value += ratio * remaining
            break

    return total_value

def job_sequencing(jobs, deadline_max):
    """Maximum profit with job deadlines."""
    # jobs: [(job_id, deadline, profit)]
    jobs.sort(key=lambda x: x[2], reverse=True)
    slots = [False] * deadline_max
    profit = 0
    count = 0

    for job_id, deadline, job_profit in jobs:
        for j in range(min(deadline, deadline_max) - 1, -1, -1):
            if not slots[j]:
                slots[j] = True
                profit += job_profit
                count += 1
                break

    return count, profit

def huffman_encoding(freqs):
    """Build Huffman tree and return code lengths."""
    import heapq

    heap = [(freq, i, None, None) for i, freq in enumerate(freqs)]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        combined = (left[0] + right[0], -1, left, right)
        heapq.heappush(heap, combined)

    def get_depths(node, depth):
        if node[1] >= 0:
            return [(node[1], depth)]
        result = []
        if node[2]:
            result.extend(get_depths(node[2], depth + 1))
        if node[3]:
            result.extend(get_depths(node[3], depth + 1))
        return result

    if not heap:
        return []
    depths = get_depths(heap[0], 0)
    return [d for _, d in sorted(depths)]

def minimum_platforms(arrivals, departures):
    """Minimum platforms needed at station."""
    arrivals.sort()
    departures.sort()

    platforms = 0
    max_platforms = 0
    i = j = 0
    n = len(arrivals)

    while i < n:
        if arrivals[i] <= departures[j]:
            platforms += 1
            max_platforms = max(max_platforms, platforms)
            i += 1
        else:
            platforms -= 1
            j += 1

    return max_platforms

def minimum_coins_greedy(coins, amount):
    """Minimum coins (greedy - works for standard denominations)."""
    coins.sort(reverse=True)
    count = 0

    for coin in coins:
        count += amount // coin
        amount %= coin

    return count if amount == 0 else -1

def assign_mice_to_holes(mice, holes):
    """Minimum time for all mice to reach holes."""
    mice.sort()
    holes.sort()
    return max(abs(m - h) for m, h in zip(mice, holes))

def partition_labels(s):
    """Partition string into max parts where each letter appears in one part."""
    last = {c: i for i, c in enumerate(s)}
    result = []
    start = end = 0

    for i, c in enumerate(s):
        end = max(end, last[c])
        if i == end:
            result.append(end - start + 1)
            start = i + 1

    return result

def remove_k_digits(num, k):
    """Remove k digits to make smallest number."""
    stack = []

    for digit in num:
        while k and stack and stack[-1] > digit:
            stack.pop()
            k -= 1
        stack.append(digit)

    stack = stack[:-k] if k else stack
    return ''.join(stack).lstrip('0') or '0'

def task_scheduler(tasks, n):
    """Minimum intervals to complete tasks with cooldown."""
    from collections import Counter
    counts = Counter(tasks)
    max_count = max(counts.values())
    max_count_tasks = sum(1 for c in counts.values() if c == max_count)

    return max(len(tasks), (max_count - 1) * (n + 1) + max_count_tasks)

def wiggle_sort(nums):
    """Wiggle sort: nums[0] <= nums[1] >= nums[2] <= nums[3]..."""
    nums.sort()
    n = len(nums)
    result = [0] * n
    left = (n - 1) // 2
    right = n - 1

    for i in range(0, n, 2):
        result[i] = nums[left]
        left -= 1
    for i in range(1, n, 2):
        result[i] = nums[right]
        right -= 1

    return result

def valid_parenthesis_string(s):
    """Check if string with * is valid parentheses."""
    lo = hi = 0

    for c in s:
        if c == '(':
            lo += 1
            hi += 1
        elif c == ')':
            lo = max(0, lo - 1)
            hi -= 1
        else:  # *
            lo = max(0, lo - 1)
            hi += 1
        if hi < 0:
            return False

    return lo == 0

def lemonade_change(bills):
    """Can give change at lemonade stand."""
    five = ten = 0

    for bill in bills:
        if bill == 5:
            five += 1
        elif bill == 10:
            if five == 0:
                return False
            five -= 1
            ten += 1
        else:  # 20
            if ten > 0 and five > 0:
                ten -= 1
                five -= 1
            elif five >= 3:
                five -= 3
            else:
                return False

    return True

# Tests
tests = [
    ("activity", activity_selection([(0,6),(1,4),(3,5),(5,7),(5,9),(8,9)]), 4),
    ("frac_knapsack", round(fractional_knapsack([10,20,30], [60,100,120], 50), 2), 240.0),
    ("job_seq", job_sequencing([('a',2,100),('b',1,19),('c',2,27),('d',1,25),('e',3,15)], 3), (3, 142)),
    ("huffman", sum(huffman_encoding([5,9,12,13,16,45])), 15),
    ("platforms", minimum_platforms([900,940,950,1100,1500,1800], [910,1200,1120,1130,1900,2000]), 3),
    ("coins", minimum_coins_greedy([1,5,10,25,100], 93), 7),
    ("mice", assign_mice_to_holes([4,-4,2], [4,0,5]), 4),
    ("partition", partition_labels("ababcbacadefegdehijhklij"), [9, 7, 8]),
    ("remove_k", remove_k_digits("1432219", 3), "1219"),
    ("remove_k_2", remove_k_digits("10200", 1), "200"),
    ("scheduler", task_scheduler(['A','A','A','B','B','B'], 2), 8),
    ("wiggle", wiggle_sort([1,5,1,1,6,4]), [1,6,1,5,1,4]),
    ("valid_paren", valid_parenthesis_string("(*))"), True),
    ("lemonade", lemonade_change([5,5,5,10,20]), True),
    ("lemonade_no", lemonade_change([5,5,10,10,20]), False),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
