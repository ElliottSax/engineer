def rotate_matrix_90(matrix):
    """Rotate matrix 90 degrees clockwise in-place."""
    n = len(matrix)
    # Transpose
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    # Reverse rows
    for row in matrix:
        row.reverse()
    return matrix

def spiral_matrix(matrix):
    """Traverse matrix in spiral order."""
    if not matrix:
        return []
    result = []
    top, bottom, left, right = 0, len(matrix) - 1, 0, len(matrix[0]) - 1

    while top <= bottom and left <= right:
        for j in range(left, right + 1):
            result.append(matrix[top][j])
        top += 1
        for i in range(top, bottom + 1):
            result.append(matrix[i][right])
        right -= 1
        if top <= bottom:
            for j in range(right, left - 1, -1):
                result.append(matrix[bottom][j])
            bottom -= 1
        if left <= right:
            for i in range(bottom, top - 1, -1):
                result.append(matrix[i][left])
            left += 1

    return result

def generate_spiral_matrix(n):
    """Generate n x n spiral matrix."""
    matrix = [[0] * n for _ in range(n)]
    top, bottom, left, right = 0, n - 1, 0, n - 1
    num = 1

    while top <= bottom and left <= right:
        for j in range(left, right + 1):
            matrix[top][j] = num
            num += 1
        top += 1
        for i in range(top, bottom + 1):
            matrix[i][right] = num
            num += 1
        right -= 1
        if top <= bottom:
            for j in range(right, left - 1, -1):
                matrix[bottom][j] = num
                num += 1
            bottom -= 1
        if left <= right:
            for i in range(bottom, top - 1, -1):
                matrix[i][left] = num
                num += 1
            left += 1

    return matrix

def set_matrix_zeros(matrix):
    """Set entire row/col to 0 if element is 0."""
    m, n = len(matrix), len(matrix[0])
    first_row_zero = any(matrix[0][j] == 0 for j in range(n))
    first_col_zero = any(matrix[i][0] == 0 for i in range(m))

    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] == 0:
                matrix[i][0] = 0
                matrix[0][j] = 0

    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0

    if first_row_zero:
        for j in range(n):
            matrix[0][j] = 0
    if first_col_zero:
        for i in range(m):
            matrix[i][0] = 0

    return matrix

def search_2d_matrix(matrix, target):
    """Search in row/col sorted matrix."""
    if not matrix:
        return False
    m, n = len(matrix), len(matrix[0])
    row, col = 0, n - 1

    while row < m and col >= 0:
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] > target:
            col -= 1
        else:
            row += 1
    return False

def diagonal_traverse(mat):
    """Traverse matrix diagonally."""
    if not mat:
        return []
    m, n = len(mat), len(mat[0])
    result = []
    row, col = 0, 0
    going_up = True

    for _ in range(m * n):
        result.append(mat[row][col])
        if going_up:
            if col == n - 1:
                row += 1
                going_up = False
            elif row == 0:
                col += 1
                going_up = False
            else:
                row -= 1
                col += 1
        else:
            if row == m - 1:
                col += 1
                going_up = True
            elif col == 0:
                row += 1
                going_up = True
            else:
                row += 1
                col -= 1

    return result

def longest_increasing_path_matrix(matrix):
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

def word_search(board, word):
    """Find word in grid."""
    m, n = len(board), len(board[0])

    def dfs(i, j, k):
        if k == len(word):
            return True
        if i < 0 or i >= m or j < 0 or j >= n or board[i][j] != word[k]:
            return False
        temp = board[i][j]
        board[i][j] = '#'
        found = (dfs(i+1, j, k+1) or dfs(i-1, j, k+1) or
                 dfs(i, j+1, k+1) or dfs(i, j-1, k+1))
        board[i][j] = temp
        return found

    return any(dfs(i, j, 0) for i in range(m) for j in range(n))

def count_submatrices_with_ones(mat):
    """Count submatrices with all 1s."""
    if not mat:
        return 0
    m, n = len(mat), len(mat[0])
    heights = [0] * n
    result = 0

    for i in range(m):
        for j in range(n):
            heights[j] = heights[j] + 1 if mat[i][j] else 0

        # Count using stack
        stack = []
        counts = [0] * n
        for j in range(n):
            while stack and heights[stack[-1]] >= heights[j]:
                stack.pop()
            if stack:
                counts[j] = counts[stack[-1]] + heights[j] * (j - stack[-1])
            else:
                counts[j] = heights[j] * (j + 1)
            stack.append(j)
        result += sum(counts)

    return result

def maximal_square(matrix):
    """Largest square containing only 1s."""
    if not matrix:
        return 0
    m, n = len(matrix), len(matrix[0])
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_side = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if matrix[i-1][j-1] == 1:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                max_side = max(max_side, dp[i][j])

    return max_side * max_side

def game_of_life(board):
    """Conway's Game of Life next state."""
    m, n = len(board), len(board[0])

    def count_neighbors(i, j):
        count = 0
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n and board[ni][nj] in [1, 2]:
                    count += 1
        return count

    for i in range(m):
        for j in range(n):
            neighbors = count_neighbors(i, j)
            if board[i][j] == 1:
                if neighbors < 2 or neighbors > 3:
                    board[i][j] = 2  # Live to dead
            else:
                if neighbors == 3:
                    board[i][j] = 3  # Dead to live

    for i in range(m):
        for j in range(n):
            board[i][j] = board[i][j] % 2

    return board

# Tests
matrix1 = [[1,2,3],[4,5,6],[7,8,9]]
tests = [
    ("rotate", rotate_matrix_90([[1,2,3],[4,5,6],[7,8,9]]), [[7,4,1],[8,5,2],[9,6,3]]),
    ("spiral", spiral_matrix([[1,2,3],[4,5,6],[7,8,9]]), [1,2,3,6,9,8,7,4,5]),
    ("gen_spiral", generate_spiral_matrix(3), [[1,2,3],[8,9,4],[7,6,5]]),
    ("set_zeros", set_matrix_zeros([[1,1,1],[1,0,1],[1,1,1]]), [[1,0,1],[0,0,0],[1,0,1]]),
    ("search_2d", search_2d_matrix([[1,4,7],[2,5,8],[3,6,9]], 5), True),
    ("search_2d_no", search_2d_matrix([[1,4,7],[2,5,8],[3,6,9]], 10), False),
    ("diagonal", diagonal_traverse([[1,2,3],[4,5,6],[7,8,9]]), [1,2,4,7,5,3,6,8,9]),
    ("longest_path", longest_increasing_path_matrix([[9,9,4],[6,6,8],[2,1,1]]), 4),
    ("word_search", word_search([['A','B'],['C','D']], "ABDC"), True),
    ("submatrices", count_submatrices_with_ones([[1,0,1],[1,1,0],[1,1,0]]), 13),
    ("max_square", maximal_square([[1,0,1,0,0],[1,0,1,1,1],[1,1,1,1,1],[1,0,0,1,0]]), 4),
]

# Game of Life test
gol = [[0,1,0],[0,0,1],[1,1,1],[0,0,0]]
game_of_life(gol)
tests.append(("game_of_life", gol, [[0,0,0],[1,0,1],[0,1,1],[0,1,0]]))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
