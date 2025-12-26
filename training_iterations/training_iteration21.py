def reconstruct_queue(people):
    """Reconstructs queue by height and people in front."""
    people.sort(key=lambda x: (-x[0], x[1]))
    result = []
    for p in people:
        result.insert(p[1], p)
    return result

def non_overlapping_intervals(intervals):
    """Minimum intervals to remove to make rest non-overlapping."""
    if not intervals:
        return 0
    intervals.sort(key=lambda x: x[1])
    count = 0
    end = intervals[0][1]
    for i in range(1, len(intervals)):
        if intervals[i][0] < end:
            count += 1
        else:
            end = intervals[i][1]
    return count

def merge_intervals(intervals):
    """Merges overlapping intervals."""
    if not intervals:
        return []
    intervals.sort()
    result = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result

def insert_interval(intervals, new_interval):
    """Inserts interval and merges overlapping."""
    result = []
    i = 0
    n = len(intervals)

    while i < n and intervals[i][1] < new_interval[0]:
        result.append(intervals[i])
        i += 1

    while i < n and intervals[i][0] <= new_interval[1]:
        new_interval[0] = min(new_interval[0], intervals[i][0])
        new_interval[1] = max(new_interval[1], intervals[i][1])
        i += 1
    result.append(new_interval)

    while i < n:
        result.append(intervals[i])
        i += 1

    return result

def minimum_arrows_burst_balloons(points):
    """Minimum arrows to burst all balloons."""
    if not points:
        return 0
    points.sort(key=lambda x: x[1])
    arrows = 1
    end = points[0][1]
    for start, e in points[1:]:
        if start > end:
            arrows += 1
            end = e
    return arrows

def partition_labels(s):
    """Partitions string so each letter appears in at most one part."""
    last = {c: i for i, c in enumerate(s)}
    result = []
    start = end = 0
    for i, c in enumerate(s):
        end = max(end, last[c])
        if i == end:
            result.append(end - start + 1)
            start = end + 1
    return result

def gas_station(gas, cost):
    """Starting gas station index for circular route."""
    if sum(gas) < sum(cost):
        return -1
    start = tank = 0
    for i in range(len(gas)):
        tank += gas[i] - cost[i]
        if tank < 0:
            start = i + 1
            tank = 0
    return start

def jump_game_ii(nums):
    """Minimum jumps to reach last index."""
    jumps = current_end = farthest = 0
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == current_end:
            jumps += 1
            current_end = farthest
    return jumps

def wiggle_sort_ii(nums):
    """Reorders so nums[0] < nums[1] > nums[2] < nums[3]..."""
    sorted_nums = sorted(nums)
    mid = (len(nums) + 1) // 2
    small = sorted_nums[:mid][::-1]
    large = sorted_nums[mid:][::-1]
    for i in range(len(nums)):
        nums[i] = small[i // 2] if i % 2 == 0 else large[i // 2]
    return nums

def h_index(citations):
    """H-index of researcher."""
    citations.sort(reverse=True)
    h = 0
    for i, c in enumerate(citations):
        if c >= i + 1:
            h = i + 1
        else:
            break
    return h

def assign_cookies(children, cookies):
    """Maximum children satisfied with cookies."""
    children.sort()
    cookies.sort()
    child = cookie = 0
    while child < len(children) and cookie < len(cookies):
        if cookies[cookie] >= children[child]:
            child += 1
        cookie += 1
    return child

def lemonade_change(bills):
    """Can give change for lemonade stand."""
    five = ten = 0
    for bill in bills:
        if bill == 5:
            five += 1
        elif bill == 10:
            if five == 0:
                return False
            five -= 1
            ten += 1
        else:
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
    ("queue", reconstruct_queue([[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]),
     [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]),
    ("non_overlap", non_overlapping_intervals([[1,2],[2,3],[3,4],[1,3]]), 1),
    ("merge", merge_intervals([[1,3],[2,6],[8,10],[15,18]]), [[1,6],[8,10],[15,18]]),
    ("insert", insert_interval([[1,3],[6,9]], [2,5]), [[1,5],[6,9]]),
    ("insert_2", insert_interval([[1,2],[3,5],[6,7],[8,10],[12,16]], [4,8]), [[1,2],[3,10],[12,16]]),
    ("arrows", minimum_arrows_burst_balloons([[10,16],[2,8],[1,6],[7,12]]), 2),
    ("partition", partition_labels("ababcbacadefegdehijhklij"), [9, 7, 8]),
    ("gas", gas_station([1,2,3,4,5], [3,4,5,1,2]), 3),
    ("gas_no", gas_station([2,3,4], [3,4,3]), -1),
    ("jump_ii", jump_game_ii([2,3,1,1,4]), 2),
    ("wiggle", wiggle_sort_ii([1,5,1,1,6,4]), [1,6,1,5,1,4]),
    ("h_index", h_index([3,0,6,1,5]), 3),
    ("cookies", assign_cookies([1,2,3], [1,1]), 1),
    ("cookies_2", assign_cookies([1,2], [1,2,3]), 2),
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
