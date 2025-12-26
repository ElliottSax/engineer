def sparse_matrix_multiply(mat1, mat2):
    """Multiplies two sparse matrices efficiently."""
    m, k = len(mat1), len(mat1[0])
    n = len(mat2[0])
    result = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(k):
            if mat1[i][j] != 0:
                for l in range(n):
                    if mat2[j][l] != 0:
                        result[i][l] += mat1[i][j] * mat2[j][l]
    return result

def game_of_life(board):
    """Next state of Game of Life."""
    m, n = len(board), len(board[0])
    directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    for i in range(m):
        for j in range(n):
            live = 0
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n:
                    live += board[ni][nj] & 1
            if board[i][j] == 1 and live in (2, 3):
                board[i][j] = 3  # 11 in binary
            elif board[i][j] == 0 and live == 3:
                board[i][j] = 2  # 10 in binary

    for i in range(m):
        for j in range(n):
            board[i][j] >>= 1
    return board

def valid_word_square(words):
    """Checks if words form valid word square."""
    n = len(words)
    for i in range(n):
        for j in range(len(words[i])):
            if j >= n or i >= len(words[j]) or words[i][j] != words[j][i]:
                return False
    return True

def word_squares(words):
    """Finds all word squares from given words."""
    from collections import defaultdict
    n = len(words[0]) if words else 0
    prefix_map = defaultdict(list)
    for word in words:
        for i in range(n):
            prefix_map[word[:i]].append(word)

    result = []

    def backtrack(square):
        if len(square) == n:
            result.append(square[:])
            return
        idx = len(square)
        prefix = ''.join(word[idx] for word in square)
        for word in prefix_map[prefix]:
            square.append(word)
            backtrack(square)
            square.pop()

    for word in words:
        backtrack([word])
    return result

def paint_house(costs):
    """Minimum cost to paint houses (no adjacent same color)."""
    if not costs:
        return 0
    n = len(costs)
    dp = costs[0][:]
    for i in range(1, n):
        new_dp = [0, 0, 0]
        new_dp[0] = costs[i][0] + min(dp[1], dp[2])
        new_dp[1] = costs[i][1] + min(dp[0], dp[2])
        new_dp[2] = costs[i][2] + min(dp[0], dp[1])
        dp = new_dp
    return min(dp)

def paint_house_ii(costs):
    """Minimum cost to paint houses with k colors."""
    if not costs:
        return 0
    n, k = len(costs), len(costs[0])

    prev_min1 = prev_min2 = 0
    prev_min1_idx = -1

    for i in range(n):
        curr_min1 = curr_min2 = float('inf')
        curr_min1_idx = -1

        for j in range(k):
            cost = costs[i][j]
            if j != prev_min1_idx:
                cost += prev_min1
            else:
                cost += prev_min2

            if cost < curr_min1:
                curr_min2 = curr_min1
                curr_min1 = cost
                curr_min1_idx = j
            elif cost < curr_min2:
                curr_min2 = cost

        prev_min1, prev_min2, prev_min1_idx = curr_min1, curr_min2, curr_min1_idx

    return prev_min1

def paint_fence(n, k):
    """Ways to paint n posts with k colors (no 3 consecutive same)."""
    if n == 0:
        return 0
    if n == 1:
        return k
    same = k
    diff = k * (k - 1)
    for _ in range(3, n + 1):
        same, diff = diff, (same + diff) * (k - 1)
    return same + diff

def flip_game(s):
    """All possible states after flipping ++ to --."""
    result = []
    for i in range(len(s) - 1):
        if s[i] == '+' and s[i + 1] == '+':
            result.append(s[:i] + '--' + s[i+2:])
    return result

def flip_game_ii(s):
    """Can first player guarantee win in flip game?"""
    memo = {}

    def can_win(state):
        if state in memo:
            return memo[state]
        for i in range(len(state) - 1):
            if state[i] == '+' and state[i + 1] == '+':
                new_state = state[:i] + '--' + state[i+2:]
                if not can_win(new_state):
                    memo[state] = True
                    return True
        memo[state] = False
        return False

    return can_win(s)

def nim_game(n):
    """Can you win Nim game (take 1-3 stones)?"""
    return n % 4 != 0

def can_i_win(max_choosable, desired_total):
    """Can first player force a win in choose-numbers game?"""
    if (max_choosable + 1) * max_choosable // 2 < desired_total:
        return False
    if desired_total <= 0:
        return True

    memo = {}

    def can_win(used, total):
        if used in memo:
            return memo[used]
        for i in range(1, max_choosable + 1):
            if used & (1 << i):
                continue
            if i >= total or not can_win(used | (1 << i), total - i):
                memo[used] = True
                return True
        memo[used] = False
        return False

    return can_win(0, desired_total)

# Tests
tests = [
    ("sparse_mult", sparse_matrix_multiply([[1,0,0],[-1,0,3]], [[7,0,0],[0,0,0],[0,0,1]]), [[7,0,0],[-7,0,3]]),
    ("word_square_valid", valid_word_square(["abcd","bnrt","crmy","dtye"]), True),
    ("word_square_invalid", valid_word_square(["abcd","bnrt","crm","dt"]), False),
    ("paint_house", paint_house([[17,2,17],[16,16,5],[14,3,19]]), 10),
    ("paint_house_ii", paint_house_ii([[1,5,3],[2,9,4]]), 5),
    ("paint_fence", paint_fence(3, 2), 6),
    ("flip_game", sorted(flip_game("++++")), sorted(["--++","+--+","++--"])),
    ("flip_game_ii", flip_game_ii("++++"), True),
    ("flip_game_ii_no", flip_game_ii("++"), False),
    ("nim", nim_game(4), False),
    ("nim_win", nim_game(5), True),
    ("can_i_win", can_i_win(10, 11), False),
    ("can_i_win_2", can_i_win(10, 0), True),
]

# Game of life test
board = [[0,1,0],[0,0,1],[1,1,1],[0,0,0]]
game_of_life(board)
tests.append(("game_of_life", board[1][1], 1))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
