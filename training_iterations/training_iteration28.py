def max_area_island(grid):
    """Finds maximum area of island."""
    if not grid:
        return 0
    m, n = len(grid), len(grid[0])
    max_area = 0

    def dfs(r, c):
        if r < 0 or r >= m or c < 0 or c >= n or grid[r][c] != 1:
            return 0
        grid[r][c] = 0
        return 1 + dfs(r+1, c) + dfs(r-1, c) + dfs(r, c+1) + dfs(r, c-1)

    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                max_area = max(max_area, dfs(i, j))
    return max_area

def number_of_distinct_islands(grid):
    """Counts distinct islands by shape."""
    if not grid:
        return 0
    m, n = len(grid), len(grid[0])
    shapes = set()

    def dfs(r, c, origin_r, origin_c, shape):
        if r < 0 or r >= m or c < 0 or c >= n or grid[r][c] != 1:
            return
        grid[r][c] = 0
        shape.append((r - origin_r, c - origin_c))
        dfs(r+1, c, origin_r, origin_c, shape)
        dfs(r-1, c, origin_r, origin_c, shape)
        dfs(r, c+1, origin_r, origin_c, shape)
        dfs(r, c-1, origin_r, origin_c, shape)

    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                shape = []
                dfs(i, j, i, j, shape)
                shapes.add(tuple(shape))
    return len(shapes)

def making_large_island(grid):
    """Maximum island size by changing at most one 0 to 1."""
    n = len(grid)
    island_id = 2
    island_size = {}

    def dfs(r, c, iid):
        if r < 0 or r >= n or c < 0 or c >= n or grid[r][c] != 1:
            return 0
        grid[r][c] = iid
        return 1 + dfs(r+1, c, iid) + dfs(r-1, c, iid) + dfs(r, c+1, iid) + dfs(r, c-1, iid)

    for i in range(n):
        for j in range(n):
            if grid[i][j] == 1:
                island_size[island_id] = dfs(i, j, island_id)
                island_id += 1

    if not island_size:
        return 1

    max_size = max(island_size.values())
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 0:
                neighbors = set()
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = i + dr, j + dc
                    if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] > 1:
                        neighbors.add(grid[nr][nc])
                size = 1 + sum(island_size[nid] for nid in neighbors)
                max_size = max(max_size, size)
    return max_size

def flood_fill(image, sr, sc, new_color):
    """Flood fills image from (sr, sc)."""
    if not image:
        return image
    m, n = len(image), len(image[0])
    original_color = image[sr][sc]
    if original_color == new_color:
        return image

    def dfs(r, c):
        if r < 0 or r >= m or c < 0 or c >= n or image[r][c] != original_color:
            return
        image[r][c] = new_color
        dfs(r+1, c); dfs(r-1, c); dfs(r, c+1); dfs(r, c-1)

    dfs(sr, sc)
    return image

def island_perimeter(grid):
    """Calculates perimeter of island."""
    perimeter = 0
    m, n = len(grid), len(grid[0])
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                perimeter += 4
                if i > 0 and grid[i-1][j] == 1:
                    perimeter -= 2
                if j > 0 and grid[i][j-1] == 1:
                    perimeter -= 2
    return perimeter

def coloring_border(grid, r0, c0, color):
    """Colors border of connected component."""
    m, n = len(grid), len(grid[0])
    original = grid[r0][c0]
    visited = set()
    border = []

    def dfs(r, c):
        if (r, c) in visited:
            return
        if r < 0 or r >= m or c < 0 or c >= n or grid[r][c] != original:
            return
        visited.add((r, c))
        is_border = False
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= m or nc < 0 or nc >= n or grid[nr][nc] != original:
                is_border = True
            else:
                dfs(nr, nc)
        if is_border:
            border.append((r, c))

    dfs(r0, c0)
    for r, c in border:
        grid[r][c] = color
    return grid

def count_sub_islands(grid1, grid2):
    """Counts islands in grid2 that are sub-islands of grid1."""
    m, n = len(grid1), len(grid1[0])

    def dfs(r, c):
        if r < 0 or r >= m or c < 0 or c >= n or grid2[r][c] == 0:
            return True
        grid2[r][c] = 0
        is_sub = grid1[r][c] == 1
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            if not dfs(r + dr, c + dc):
                is_sub = False
        return is_sub

    count = 0
    for i in range(m):
        for j in range(n):
            if grid2[i][j] == 1:
                if dfs(i, j):
                    count += 1
    return count

def closed_island(grid):
    """Counts islands not touching border."""
    m, n = len(grid), len(grid[0])

    def dfs(r, c):
        if r < 0 or r >= m or c < 0 or c >= n:
            return False
        if grid[r][c] == 1:
            return True
        grid[r][c] = 1
        left = dfs(r, c-1)
        right = dfs(r, c+1)
        up = dfs(r-1, c)
        down = dfs(r+1, c)
        return left and right and up and down

    count = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 0:
                if dfs(i, j):
                    count += 1
    return count

def enclaves(grid):
    """Counts land cells that cannot walk off boundary."""
    m, n = len(grid), len(grid[0])

    def dfs(r, c):
        if r < 0 or r >= m or c < 0 or c >= n or grid[r][c] == 0:
            return
        grid[r][c] = 0
        dfs(r+1, c); dfs(r-1, c); dfs(r, c+1); dfs(r, c-1)

    # Remove border-connected land
    for i in range(m):
        dfs(i, 0)
        dfs(i, n-1)
    for j in range(n):
        dfs(0, j)
        dfs(m-1, j)

    return sum(sum(row) for row in grid)

# Tests
tests = [
    ("max_area", max_area_island([[0,1,1,0],[0,1,0,0],[1,1,0,0],[0,0,0,0]]), 5),
    ("distinct_islands", number_of_distinct_islands([[1,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,1]]), 1),
    ("large_island", making_large_island([[1,0],[0,1]]), 3),
    ("large_island_full", making_large_island([[1,1],[1,1]]), 4),
    ("flood_fill", flood_fill([[1,1,1],[1,1,0],[1,0,1]], 1, 1, 2), [[2,2,2],[2,2,0],[2,0,1]]),
    ("perimeter", island_perimeter([[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]), 16),
    ("sub_islands", count_sub_islands([[1,1,1,0,0],[0,1,1,1,1],[0,0,0,0,0],[1,0,0,0,0],[1,1,0,1,1]],
                                       [[1,1,1,0,0],[0,0,1,1,1],[0,1,0,0,0],[1,0,1,1,0],[0,1,0,1,0]]), 3),
    ("closed", closed_island([[1,1,1,1,1,1,1,0],[1,0,0,0,0,1,1,0],[1,0,1,0,1,1,1,0],[1,0,0,0,0,1,0,1],[1,1,1,1,1,1,1,0]]), 2),
    ("enclaves", enclaves([[0,0,0,0],[1,0,1,0],[0,1,1,0],[0,0,0,0]]), 3),
]

# Color border test
grid = [[1,1],[1,2]]
coloring_border(grid, 0, 0, 3)
tests.append(("color_border", grid, [[3,3],[3,2]]))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
