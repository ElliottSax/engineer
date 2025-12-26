def topological_sort(graph):
    """Performs topological sort on DAG."""
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] = in_degree.get(neighbor, 0) + 1
    queue = [node for node in graph if in_degree[node] == 0]
    result = []
    while queue:
        node = queue.pop(0)
        result.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return result if len(result) == len(graph) else []

def dijkstra(graph, start):
    """Shortest path from start to all nodes (graph is {node: [(neighbor, weight)]})."""
    import heapq
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    while pq:
        dist, node = heapq.heappop(pq)
        if dist > distances[node]:
            continue
        for neighbor, weight in graph.get(node, []):
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))
    return distances

def connected_components(graph):
    """Counts connected components in undirected graph."""
    visited = set()
    count = 0
    def dfs(node):
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)
    for node in graph:
        if node not in visited:
            dfs(node)
            count += 1
    return count

def lru_cache_impl(capacity):
    """Returns an LRU cache with get and put methods."""
    from collections import OrderedDict
    cache = OrderedDict()
    def get(key):
        if key not in cache:
            return -1
        cache.move_to_end(key)
        return cache[key]
    def put(key, value):
        if key in cache:
            cache.move_to_end(key)
        cache[key] = value
        if len(cache) > capacity:
            cache.popitem(last=False)
    return get, put

def median_sorted_arrays(nums1, nums2):
    """Finds median of two sorted arrays."""
    merged = []
    i = j = 0
    while i < len(nums1) and j < len(nums2):
        if nums1[i] <= nums2[j]:
            merged.append(nums1[i])
            i += 1
        else:
            merged.append(nums2[j])
            j += 1
    merged.extend(nums1[i:])
    merged.extend(nums2[j:])
    n = len(merged)
    if n % 2 == 1:
        return float(merged[n // 2])
    return (merged[n // 2 - 1] + merged[n // 2]) / 2.0

def rotate_matrix(matrix):
    """Rotates matrix 90 degrees clockwise in-place."""
    n = len(matrix)
    # Transpose
    for i in range(n):
        for j in range(i, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    # Reverse rows
    for row in matrix:
        row.reverse()
    return matrix

def longest_palindrome_substring(s):
    """Finds longest palindromic substring."""
    if not s:
        return ""
    start, max_len = 0, 1
    def expand(left, right):
        nonlocal start, max_len
        while left >= 0 and right < len(s) and s[left] == s[right]:
            if right - left + 1 > max_len:
                start = left
                max_len = right - left + 1
            left -= 1
            right += 1
    for i in range(len(s)):
        expand(i, i)
        expand(i, i + 1)
    return s[start:start + max_len]

def minimum_path_sum(grid):
    """Finds minimum path sum from top-left to bottom-right."""
    if not grid or not grid[0]:
        return 0
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
    return dp[m-1][n-1]

# Tests
tests = [
    ("topological", topological_sort({0:[1,2], 1:[3], 2:[3], 3:[]}), [0, 1, 2, 3]),
    ("connected_comp", connected_components({0:[1], 1:[0], 2:[3], 3:[2], 4:[]}), 3),
    ("median_1", median_sorted_arrays([1,3], [2]), 2.0),
    ("median_2", median_sorted_arrays([1,2], [3,4]), 2.5),
    ("rotate", rotate_matrix([[1,2],[3,4]]), [[3,1],[4,2]]),
    ("palindrome_sub", longest_palindrome_substring("babad") in ["bab", "aba"], True),
    ("min_path", minimum_path_sum([[1,3,1],[1,5,1],[4,2,1]]), 7),
]

# LRU cache test
get, put = lru_cache_impl(2)
put(1, 1)
put(2, 2)
lru_test = get(1) == 1
put(3, 3)  # evicts key 2
lru_test = lru_test and get(2) == -1
tests.append(("lru_cache", lru_test, True))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
