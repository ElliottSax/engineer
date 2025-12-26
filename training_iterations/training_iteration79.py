def trap_rain_water_3d(height_map):
    """Trap rain water in 3D."""
    if not height_map or not height_map[0]:
        return 0

    import heapq
    m, n = len(height_map), len(height_map[0])
    visited = [[False] * n for _ in range(m)]
    heap = []

    # Add boundary cells
    for i in range(m):
        for j in range(n):
            if i == 0 or i == m - 1 or j == 0 or j == n - 1:
                heapq.heappush(heap, (height_map[i][j], i, j))
                visited[i][j] = True

    water = 0
    while heap:
        h, r, c = heapq.heappop(heap)
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and not visited[nr][nc]:
                visited[nr][nc] = True
                water += max(0, h - height_map[nr][nc])
                heapq.heappush(heap, (max(h, height_map[nr][nc]), nr, nc))

    return water

def skyline(buildings):
    """Get skyline from buildings."""
    import heapq
    events = []
    for left, right, height in buildings:
        events.append((left, -height, right))  # start event
        events.append((right, 0, 0))           # end event

    events.sort()
    result = [[0, 0]]
    heap = [(0, float('inf'))]  # (neg_height, end_x)

    for x, neg_h, end in events:
        while heap[0][1] <= x:
            heapq.heappop(heap)

        if neg_h:
            heapq.heappush(heap, (neg_h, end))

        max_h = -heap[0][0]
        if max_h != result[-1][1]:
            result.append([x, max_h])

    return result[1:]

def rectangle_area(rectangles):
    """Total area covered by rectangles."""
    MOD = 10**9 + 7
    events = []
    for x1, y1, x2, y2 in rectangles:
        events.append((x1, 0, y1, y2))  # open
        events.append((x2, 1, y1, y2))  # close

    events.sort()

    def calculate_y_length(active):
        if not active:
            return 0
        intervals = sorted(active)
        length = 0
        curr_start, curr_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= curr_end:
                curr_end = max(curr_end, end)
            else:
                length += curr_end - curr_start
                curr_start, curr_end = start, end
        length += curr_end - curr_start
        return length

    active = []
    prev_x = events[0][0]
    area = 0

    for x, typ, y1, y2 in events:
        y_length = calculate_y_length(active)
        area = (area + y_length * (x - prev_x)) % MOD

        if typ == 0:
            active.append((y1, y2))
        else:
            active.remove((y1, y2))

        prev_x = x

    return area

def max_points_on_line(points):
    """Maximum points on a line."""
    from math import gcd
    if len(points) <= 2:
        return len(points)

    def get_slope(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if dx == 0:
            return (float('inf'), 0)
        if dy == 0:
            return (0, float('inf'))
        g = gcd(dx, dy)
        return (dy // g, dx // g)

    max_points = 0
    for i in range(len(points)):
        slopes = {}
        same = 1
        for j in range(i + 1, len(points)):
            if points[i] == points[j]:
                same += 1
            else:
                slope = get_slope(points[i], points[j])
                slopes[slope] = slopes.get(slope, 0) + 1
        local_max = same + (max(slopes.values()) if slopes else 0)
        max_points = max(max_points, local_max)

    return max_points

def longest_increasing_path(matrix):
    """Longest increasing path in matrix."""
    if not matrix:
        return 0

    m, n = len(matrix), len(matrix[0])
    memo = {}

    def dfs(i, j):
        if (i, j) in memo:
            return memo[(i, j)]

        max_len = 1
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and matrix[ni][nj] > matrix[i][j]:
                max_len = max(max_len, 1 + dfs(ni, nj))

        memo[(i, j)] = max_len
        return max_len

    return max(dfs(i, j) for i in range(m) for j in range(n))

def dungeon_game(dungeon):
    """Minimum initial health to reach princess."""
    m, n = len(dungeon), len(dungeon[0])
    dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
    dp[m][n - 1] = dp[m - 1][n] = 1

    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            need = min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j]
            dp[i][j] = max(1, need)

    return dp[0][0]

def cherry_pickup(grid):
    """Maximum cherries picked going and returning."""
    n = len(grid)
    from functools import lru_cache

    @lru_cache(None)
    def dp(r1, c1, r2):
        c2 = r1 + c1 - r2
        if r1 == n or c1 == n or r2 == n or c2 == n:
            return float('-inf')
        if grid[r1][c1] == -1 or grid[r2][c2] == -1:
            return float('-inf')
        if r1 == n - 1 and c1 == n - 1:
            return grid[r1][c1]

        cherries = grid[r1][c1]
        if r1 != r2:
            cherries += grid[r2][c2]

        return cherries + max(
            dp(r1 + 1, c1, r2 + 1),
            dp(r1 + 1, c1, r2),
            dp(r1, c1 + 1, r2 + 1),
            dp(r1, c1 + 1, r2)
        )

    return max(0, dp(0, 0, 0))

# Tests
tests = [
    ("trap_3d", trap_rain_water_3d([[1,4,3,1,3,2],[3,2,1,3,2,4],[2,3,3,2,3,1]]), 4),
    ("skyline", skyline([[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]]),
     [[2,10],[3,15],[7,12],[12,0],[15,10],[20,8],[24,0]]),
    ("max_points", max_points_on_line([[1,1],[2,2],[3,3]]), 3),
    ("max_points_2", max_points_on_line([[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]), 4),
    ("longest_path", longest_increasing_path([[9,9,4],[6,6,8],[2,1,1]]), 4),
    ("dungeon", dungeon_game([[-2,-3,3],[-5,-10,1],[10,30,-5]]), 7),
    ("cherry", cherry_pickup([[0,1,-1],[1,0,-1],[1,1,1]]), 5),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
