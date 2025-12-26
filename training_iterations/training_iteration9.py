def longest_consecutive_sequence(nums):
    """Length of longest consecutive elements sequence."""
    num_set = set(nums)
    max_len = 0
    for num in num_set:
        if num - 1 not in num_set:  # Start of sequence
            current = num
            length = 1
            while current + 1 in num_set:
                current += 1
                length += 1
            max_len = max(max_len, length)
    return max_len

def word_search(board, word):
    """Checks if word exists in 2D board of letters."""
    rows, cols = len(board), len(board[0])
    def dfs(r, c, idx):
        if idx == len(word):
            return True
        if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != word[idx]:
            return False
        temp = board[r][c]
        board[r][c] = '#'
        found = dfs(r+1, c, idx+1) or dfs(r-1, c, idx+1) or dfs(r, c+1, idx+1) or dfs(r, c-1, idx+1)
        board[r][c] = temp
        return found
    for i in range(rows):
        for j in range(cols):
            if dfs(i, j, 0):
                return True
    return False

def palindrome_partitioning(s):
    """All palindrome partitions of string."""
    result = []
    def is_palindrome(sub):
        return sub == sub[::-1]
    def backtrack(start, path):
        if start == len(s):
            result.append(path[:])
            return
        for end in range(start + 1, len(s) + 1):
            if is_palindrome(s[start:end]):
                path.append(s[start:end])
                backtrack(end, path)
                path.pop()
    backtrack(0, [])
    return result

def surrounded_regions(board):
    """Capture surrounded 'O' regions by converting to 'X'."""
    if not board or not board[0]:
        return board
    rows, cols = len(board), len(board[0])
    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != 'O':
            return
        board[r][c] = 'S'  # Safe
        dfs(r+1, c); dfs(r-1, c); dfs(r, c+1); dfs(r, c-1)
    # Mark border-connected O's as safe
    for i in range(rows):
        dfs(i, 0); dfs(i, cols-1)
    for j in range(cols):
        dfs(0, j); dfs(rows-1, j)
    # Convert
    for i in range(rows):
        for j in range(cols):
            if board[i][j] == 'O':
                board[i][j] = 'X'
            elif board[i][j] == 'S':
                board[i][j] = 'O'
    return board

def partition_equal_subset_sum(nums):
    """Can partition array into two equal sum subsets?"""
    total = sum(nums)
    if total % 2 != 0:
        return False
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    for num in nums:
        for i in range(target, num - 1, -1):
            dp[i] = dp[i] or dp[i - num]
    return dp[target]

def target_sum(nums, target):
    """Number of ways to assign +/- to reach target."""
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def dp(idx, current_sum):
        if idx == len(nums):
            return 1 if current_sum == target else 0
        return dp(idx + 1, current_sum + nums[idx]) + dp(idx + 1, current_sum - nums[idx])
    return dp(0, 0)

def interval_intersection(A, B):
    """Finds intersection of two interval lists."""
    result = []
    i = j = 0
    while i < len(A) and j < len(B):
        start = max(A[i][0], B[j][0])
        end = min(A[i][1], B[j][1])
        if start <= end:
            result.append([start, end])
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1
    return result

def task_scheduler(tasks, n):
    """Minimum intervals to finish all tasks with cooldown n."""
    from collections import Counter
    counts = list(Counter(tasks).values())
    max_count = max(counts)
    max_count_tasks = counts.count(max_count)
    return max(len(tasks), (max_count - 1) * (n + 1) + max_count_tasks)

def valid_sudoku(board):
    """Checks if Sudoku board is valid."""
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    for i in range(9):
        for j in range(9):
            num = board[i][j]
            if num == '.':
                continue
            box_idx = (i // 3) * 3 + j // 3
            if num in rows[i] or num in cols[j] or num in boxes[box_idx]:
                return False
            rows[i].add(num)
            cols[j].add(num)
            boxes[box_idx].add(num)
    return True

# Tests
tests = [
    ("longest_consec", longest_consecutive_sequence([100,4,200,1,3,2]), 4),
    ("word_search_yes", word_search([["A","B","C"],["S","F","C"],["A","D","E"]], "ABCCF"), True),
    ("word_search_no", word_search([["A","B"],["C","D"]], "ABDC"), False),
    ("palindrome_part", len(palindrome_partitioning("aab")), 2),
    ("partition_sum_yes", partition_equal_subset_sum([1,5,11,5]), True),
    ("partition_sum_no", partition_equal_subset_sum([1,2,3,5]), False),
    ("target_sum", target_sum([1,1,1,1,1], 3), 5),
    ("interval_inter", interval_intersection([[0,2],[5,10]], [[1,5],[8,12]]), [[1,2],[5,5],[8,10]]),
    ("task_scheduler", task_scheduler(["A","A","A","B","B","B"], 2), 8),
]

# Surrounded regions test
board = [['X','X','X','X'],['X','O','O','X'],['X','X','O','X'],['X','O','X','X']]
surrounded_regions(board)
tests.append(("surrounded_reg", board[1][1], 'X'))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
