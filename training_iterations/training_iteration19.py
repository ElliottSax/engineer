def min_window_subsequence(s1, s2):
    """Minimum window in s1 containing s2 as subsequence."""
    m, n = len(s1), len(s2)
    min_len = float('inf')
    result = ""
    i = 0
    while i < m:
        j = 0
        while i < m and j < n:
            if s1[i] == s2[j]:
                j += 1
            i += 1
        if j < n:
            break
        end = i
        j = n - 1
        i -= 1
        while j >= 0:
            if s1[i] == s2[j]:
                j -= 1
            i -= 1
        i += 1
        if end - i < min_len:
            min_len = end - i
            result = s1[i:end]
        i += 1
    return result

def count_palindromic_substrings(s):
    """Counts palindromic substrings."""
    n = len(s)
    count = 0

    def expand(left, right):
        nonlocal count
        while left >= 0 and right < n and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1

    for i in range(n):
        expand(i, i)      # odd length
        expand(i, i + 1)  # even length
    return count

def maximum_subarray_sum_circular(nums):
    """Maximum subarray sum in circular array."""
    def kadane(arr):
        max_sum = curr = arr[0]
        for num in arr[1:]:
            curr = max(num, curr + num)
            max_sum = max(max_sum, curr)
        return max_sum

    max_kadane = kadane(nums)
    total = sum(nums)
    min_sum = -kadane([-x for x in nums])

    if min_sum == total:  # All negative
        return max_kadane
    return max(max_kadane, total - min_sum)

def shortest_bridge(grid):
    """Minimum flips to connect two islands."""
    from collections import deque
    n = len(grid)
    visited = set()
    queue = deque()

    def dfs(r, c):
        if 0 <= r < n and 0 <= c < n and (r, c) not in visited and grid[r][c] == 1:
            visited.add((r, c))
            queue.append((r, c, 0))
            dfs(r+1, c); dfs(r-1, c); dfs(r, c+1); dfs(r, c-1)

    # Find first island
    found = False
    for i in range(n):
        if found:
            break
        for j in range(n):
            if grid[i][j] == 1:
                dfs(i, j)
                found = True
                break

    # BFS to second island
    while queue:
        r, c, dist = queue.popleft()
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in visited:
                if grid[nr][nc] == 1:
                    return dist
                visited.add((nr, nc))
                queue.append((nr, nc, dist + 1))
    return -1

def snakes_and_ladders(board):
    """Minimum moves in snakes and ladders."""
    from collections import deque
    n = len(board)

    def get_pos(num):
        r = (num - 1) // n
        c = (num - 1) % n
        if r % 2 == 1:
            c = n - 1 - c
        return n - 1 - r, c

    visited = set()
    queue = deque([(1, 0)])
    visited.add(1)

    while queue:
        pos, moves = queue.popleft()
        for i in range(1, 7):
            next_pos = pos + i
            if next_pos > n * n:
                continue
            r, c = get_pos(next_pos)
            if board[r][c] != -1:
                next_pos = board[r][c]
            if next_pos == n * n:
                return moves + 1
            if next_pos not in visited:
                visited.add(next_pos)
                queue.append((next_pos, moves + 1))
    return -1

def rotting_oranges(grid):
    """Time for all oranges to rot."""
    from collections import deque
    m, n = len(grid), len(grid[0])
    queue = deque()
    fresh = 0

    for i in range(m):
        for j in range(n):
            if grid[i][j] == 2:
                queue.append((i, j, 0))
            elif grid[i][j] == 1:
                fresh += 1

    if fresh == 0:
        return 0

    time = 0
    while queue:
        r, c, t = queue.popleft()
        time = t
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == 1:
                grid[nr][nc] = 2
                fresh -= 1
                queue.append((nr, nc, t + 1))

    return time if fresh == 0 else -1

def walls_and_gates(rooms):
    """Fills distance to nearest gate (0)."""
    from collections import deque
    if not rooms:
        return rooms
    m, n = len(rooms), len(rooms[0])
    queue = deque()
    INF = 2147483647

    for i in range(m):
        for j in range(n):
            if rooms[i][j] == 0:
                queue.append((i, j))

    while queue:
        r, c = queue.popleft()
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and rooms[nr][nc] == INF:
                rooms[nr][nc] = rooms[r][c] + 1
                queue.append((nr, nc))
    return rooms

def shortest_path_binary_matrix(grid):
    """Shortest path from top-left to bottom-right (8-directional)."""
    from collections import deque
    n = len(grid)
    if grid[0][0] == 1 or grid[n-1][n-1] == 1:
        return -1

    directions = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    queue = deque([(0, 0, 1)])
    grid[0][0] = 1

    while queue:
        r, c, dist = queue.popleft()
        if r == n - 1 and c == n - 1:
            return dist
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0:
                grid[nr][nc] = 1
                queue.append((nr, nc, dist + 1))
    return -1

def pacific_atlantic(heights):
    """Cells that can flow to both Pacific and Atlantic."""
    if not heights:
        return []
    m, n = len(heights), len(heights[0])

    def dfs(starts):
        visited = set()
        stack = list(starts)
        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and (nr, nc) not in visited:
                    if heights[nr][nc] >= heights[r][c]:
                        stack.append((nr, nc))
        return visited

    pacific = [(0, j) for j in range(n)] + [(i, 0) for i in range(1, m)]
    atlantic = [(m-1, j) for j in range(n)] + [(i, n-1) for i in range(m-1)]

    pac_reach = dfs(pacific)
    atl_reach = dfs(atlantic)

    return [[r, c] for r, c in pac_reach & atl_reach]

# Tests
tests = [
    ("min_window_subseq", min_window_subsequence("abcdebdde", "bde"), "bcde"),
    ("count_palindromes", count_palindromic_substrings("aaa"), 6),
    ("count_palindromes_2", count_palindromic_substrings("abc"), 3),
    ("max_circular", maximum_subarray_sum_circular([1,-2,3,-2]), 3),
    ("max_circular_wrap", maximum_subarray_sum_circular([5,-3,5]), 10),
    ("bridge", shortest_bridge([[0,1],[1,0]]), 1),
    ("bridge_2", shortest_bridge([[0,1,0],[0,0,0],[0,0,1]]), 2),
    ("rotting", rotting_oranges([[2,1,1],[1,1,0],[0,1,1]]), 4),
    ("rotting_impossible", rotting_oranges([[2,1,1],[0,1,1],[1,0,1]]), -1),
    ("binary_path", shortest_path_binary_matrix([[0,0,0],[1,1,0],[1,1,0]]), 4),
    ("binary_path_blocked", shortest_path_binary_matrix([[1,0],[0,0]]), -1),
]

# Walls and gates test
rooms = [[2147483647,-1,0,2147483647],[2147483647,2147483647,2147483647,-1],[2147483647,-1,2147483647,-1],[0,-1,2147483647,2147483647]]
walls_and_gates(rooms)
tests.append(("walls_gates", rooms[0][0], 3))

# Pacific Atlantic test
heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
result = pacific_atlantic(heights)
tests.append(("pacific_atlantic", len(result), 7))

# Snakes test
board = [[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,35,-1,-1,13,-1],[-1,-1,-1,-1,-1,-1],[-1,15,-1,-1,-1,-1]]
tests.append(("snakes", snakes_and_ladders(board), 4))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
