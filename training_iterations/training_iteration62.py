from collections import deque

def number_of_islands(grid):
    """Count number of islands."""
    if not grid:
        return 0

    m, n = len(grid), len(grid[0])
    count = 0

    def dfs(i, j):
        if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != '1':
            return
        grid[i][j] = '0'
        dfs(i + 1, j)
        dfs(i - 1, j)
        dfs(i, j + 1)
        dfs(i, j - 1)

    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                count += 1
                dfs(i, j)

    return count

def max_area_island(grid):
    """Maximum area of island."""
    if not grid:
        return 0

    m, n = len(grid), len(grid[0])

    def dfs(i, j):
        if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != 1:
            return 0
        grid[i][j] = 0
        return 1 + dfs(i+1, j) + dfs(i-1, j) + dfs(i, j+1) + dfs(i, j-1)

    return max(dfs(i, j) for i in range(m) for j in range(n))

def shortest_path_binary_matrix(grid):
    """Shortest path from top-left to bottom-right."""
    n = len(grid)
    if grid[0][0] == 1 or grid[n-1][n-1] == 1:
        return -1

    queue = deque([(0, 0, 1)])
    grid[0][0] = 1

    while queue:
        r, c, dist = queue.popleft()
        if r == n - 1 and c == n - 1:
            return dist
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0:
                    grid[nr][nc] = 1
                    queue.append((nr, nc, dist + 1))

    return -1

def walls_and_gates(rooms):
    """Fill each empty room with distance to nearest gate."""
    if not rooms:
        return rooms

    m, n = len(rooms), len(rooms[0])
    INF = 2147483647
    queue = deque()

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

def surrounded_regions(board):
    """Capture surrounded regions."""
    if not board:
        return board

    m, n = len(board), len(board[0])

    def dfs(i, j):
        if i < 0 or i >= m or j < 0 or j >= n or board[i][j] != 'O':
            return
        board[i][j] = 'T'
        dfs(i+1, j)
        dfs(i-1, j)
        dfs(i, j+1)
        dfs(i, j-1)

    # Mark border connected O's
    for i in range(m):
        dfs(i, 0)
        dfs(i, n - 1)
    for j in range(n):
        dfs(0, j)
        dfs(m - 1, j)

    # Flip
    for i in range(m):
        for j in range(n):
            if board[i][j] == 'O':
                board[i][j] = 'X'
            elif board[i][j] == 'T':
                board[i][j] = 'O'

    return board

def pacific_atlantic(heights):
    """Cells that can reach both oceans."""
    if not heights:
        return []

    m, n = len(heights), len(heights[0])
    pacific = set()
    atlantic = set()

    def dfs(i, j, visited):
        visited.add((i, j))
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = i + di, j + dj
            if (0 <= ni < m and 0 <= nj < n and
                (ni, nj) not in visited and
                heights[ni][nj] >= heights[i][j]):
                dfs(ni, nj, visited)

    for i in range(m):
        dfs(i, 0, pacific)
        dfs(i, n - 1, atlantic)
    for j in range(n):
        dfs(0, j, pacific)
        dfs(m - 1, j, atlantic)

    return sorted(list(pacific & atlantic))

def rotting_oranges(grid):
    """Minutes until all oranges rot."""
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
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == 1:
                grid[nr][nc] = 2
                fresh -= 1
                time = t + 1
                queue.append((nr, nc, t + 1))

    return time if fresh == 0 else -1

def shortest_bridge(grid):
    """Minimum flips to connect two islands."""
    m, n = len(grid), len(grid[0])
    queue = deque()

    def dfs(i, j):
        if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != 1:
            return
        grid[i][j] = 2
        queue.append((i, j, 0))
        dfs(i+1, j)
        dfs(i-1, j)
        dfs(i, j+1)
        dfs(i, j-1)

    # Find first island
    found = False
    for i in range(m):
        if found:
            break
        for j in range(n):
            if grid[i][j] == 1:
                dfs(i, j)
                found = True
                break

    # BFS to find second island
    while queue:
        r, c, dist = queue.popleft()
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n:
                if grid[nr][nc] == 1:
                    return dist
                if grid[nr][nc] == 0:
                    grid[nr][nc] = 2
                    queue.append((nr, nc, dist + 1))

    return -1

def making_large_island(grid):
    """Maximum island size by changing one 0 to 1."""
    n = len(grid)
    if n == 0:
        return 0

    # Label islands and compute sizes
    island_id = 2
    sizes = {0: 0}

    def dfs(i, j, id):
        if i < 0 or i >= n or j < 0 or j >= n or grid[i][j] != 1:
            return 0
        grid[i][j] = id
        return 1 + dfs(i+1, j, id) + dfs(i-1, j, id) + dfs(i, j+1, id) + dfs(i, j-1, id)

    for i in range(n):
        for j in range(n):
            if grid[i][j] == 1:
                sizes[island_id] = dfs(i, j, island_id)
                island_id += 1

    if not sizes:
        return 1

    result = max(sizes.values())

    for i in range(n):
        for j in range(n):
            if grid[i][j] == 0:
                neighbors = set()
                for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n:
                        neighbors.add(grid[ni][nj])
                result = max(result, 1 + sum(sizes.get(id, 0) for id in neighbors))

    return result

def count_sub_islands(grid1, grid2):
    """Count sub-islands in grid2 that are also in grid1."""
    m, n = len(grid1), len(grid1[0])

    def dfs(i, j):
        if i < 0 or i >= m or j < 0 or j >= n or grid2[i][j] != 1:
            return True
        grid2[i][j] = 0
        is_sub = grid1[i][j] == 1
        # Must check all cells even if already not a sub-island
        is_sub &= dfs(i+1, j)
        is_sub &= dfs(i-1, j)
        is_sub &= dfs(i, j+1)
        is_sub &= dfs(i, j-1)
        return is_sub

    count = 0
    for i in range(m):
        for j in range(n):
            if grid2[i][j] == 1 and dfs(i, j):
                count += 1
    return count

# Tests
tests = [
    ("islands", number_of_islands([['1','1','0','0','0'],['1','1','0','0','0'],['0','0','1','0','0'],['0','0','0','1','1']]), 3),
    ("max_area", max_area_island([[0,0,1,0,0],[0,0,0,0,0],[0,1,1,1,0],[0,0,0,0,0]]), 4),
    ("shortest_bin", shortest_path_binary_matrix([[0,0,0],[1,1,0],[1,1,0]]), 4),
    ("walls_gates", walls_and_gates([[2147483647,-1,0,2147483647],[2147483647,2147483647,2147483647,-1],[2147483647,-1,2147483647,-1],[0,-1,2147483647,2147483647]]),
     [[3,-1,0,1],[2,2,1,-1],[1,-1,2,-1],[0,-1,3,4]]),
    ("surrounded", surrounded_regions([['X','X','X','X'],['X','O','O','X'],['X','X','O','X'],['X','O','X','X']]),
     [['X','X','X','X'],['X','X','X','X'],['X','X','X','X'],['X','O','X','X']]),
    ("pacific", pacific_atlantic([[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]),
     [(0,4),(1,3),(1,4),(2,2),(3,0),(3,1),(4,0)]),
    ("rotting", rotting_oranges([[2,1,1],[1,1,0],[0,1,1]]), 4),
    ("bridge", shortest_bridge([[0,1],[1,0]]), 1),
    ("large_island", making_large_island([[1,0],[0,1]]), 3),
    ("sub_islands", count_sub_islands([[1,1,1,0,0],[0,1,1,1,1],[0,0,0,0,0],[1,0,0,0,0],[1,1,0,1,1]],
                                      [[1,1,1,0,0],[0,0,1,1,1],[0,1,0,0,0],[1,0,1,1,0],[0,1,0,1,0]]), 3),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
