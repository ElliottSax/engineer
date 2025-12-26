def permutations(nums):
    """Generate all permutations."""
    result = []

    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return
        for i, num in enumerate(remaining):
            path.append(num)
            backtrack(path, remaining[:i] + remaining[i+1:])
            path.pop()

    backtrack([], nums)
    return result

def permutations_unique(nums):
    """Permutations with duplicates."""
    result = []
    nums.sort()

    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return
        for i, num in enumerate(remaining):
            if i > 0 and remaining[i] == remaining[i-1]:
                continue
            path.append(num)
            backtrack(path, remaining[:i] + remaining[i+1:])
            path.pop()

    backtrack([], nums)
    return result

def combinations(n, k):
    """Generate all C(n,k) combinations."""
    result = []

    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return
        for i in range(start, n + 1):
            path.append(i)
            backtrack(i + 1, path)
            path.pop()

    backtrack(1, [])
    return result

def combination_sum(candidates, target):
    """Combinations that sum to target (can reuse)."""
    result = []
    candidates.sort()

    def backtrack(start, path, remaining):
        if remaining == 0:
            result.append(path[:])
            return
        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break
            path.append(candidates[i])
            backtrack(i, path, remaining - candidates[i])
            path.pop()

    backtrack(0, [], target)
    return result

def combination_sum_ii(candidates, target):
    """Combinations that sum to target (no reuse)."""
    result = []
    candidates.sort()

    def backtrack(start, path, remaining):
        if remaining == 0:
            result.append(path[:])
            return
        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break
            if i > start and candidates[i] == candidates[i-1]:
                continue
            path.append(candidates[i])
            backtrack(i + 1, path, remaining - candidates[i])
            path.pop()

    backtrack(0, [], target)
    return result

def subsets(nums):
    """Generate all subsets."""
    result = []

    def backtrack(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result

def subsets_with_dup(nums):
    """Subsets with duplicates."""
    result = []
    nums.sort()

    def backtrack(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i-1]:
                continue
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result

def generate_parentheses(n):
    """Generate all valid parentheses combinations."""
    result = []

    def backtrack(open_count, close_count, path):
        if len(path) == 2 * n:
            result.append(''.join(path))
            return
        if open_count < n:
            path.append('(')
            backtrack(open_count + 1, close_count, path)
            path.pop()
        if close_count < open_count:
            path.append(')')
            backtrack(open_count, close_count + 1, path)
            path.pop()

    backtrack(0, 0, [])
    return result

def letter_combinations(digits):
    """Letter combinations of phone number."""
    if not digits:
        return []

    mapping = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }
    result = []

    def backtrack(idx, path):
        if idx == len(digits):
            result.append(''.join(path))
            return
        for c in mapping[digits[idx]]:
            path.append(c)
            backtrack(idx + 1, path)
            path.pop()

    backtrack(0, [])
    return result

def n_queens(n):
    """Solve N-Queens problem."""
    result = []
    board = ['.' * n for _ in range(n)]
    cols = set()
    diag1 = set()  # row - col
    diag2 = set()  # row + col

    def backtrack(row):
        if row == n:
            result.append(board[:])
            return
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            board[row] = '.' * col + 'Q' + '.' * (n - col - 1)
            backtrack(row + 1)
            board[row] = '.' * n
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    backtrack(0)
    return result

def sudoku_solver(board):
    """Solve sudoku puzzle."""
    def is_valid(r, c, ch):
        for i in range(9):
            if board[r][i] == ch:
                return False
            if board[i][c] == ch:
                return False
            if board[3*(r//3) + i//3][3*(c//3) + i%3] == ch:
                return False
        return True

    def solve():
        for r in range(9):
            for c in range(9):
                if board[r][c] == '.':
                    for ch in '123456789':
                        if is_valid(r, c, ch):
                            board[r][c] = ch
                            if solve():
                                return True
                            board[r][c] = '.'
                    return False
        return True

    solve()
    return board

def partition_palindrome(s):
    """Partition string into palindrome substrings."""
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

def restore_ip_addresses(s):
    """Restore valid IP addresses from string."""
    result = []

    def is_valid(segment):
        if len(segment) > 1 and segment[0] == '0':
            return False
        return 0 <= int(segment) <= 255

    def backtrack(start, parts):
        if len(parts) == 4:
            if start == len(s):
                result.append('.'.join(parts))
            return
        for length in range(1, 4):
            if start + length <= len(s):
                segment = s[start:start + length]
                if is_valid(segment):
                    backtrack(start + length, parts + [segment])

    backtrack(0, [])
    return result

# Tests
tests = [
    ("perms", len(permutations([1,2,3])), 6),
    ("perms_uniq", len(permutations_unique([1,1,2])), 3),
    ("combos", len(combinations(4, 2)), 6),
    ("combo_sum", sorted(combination_sum([2,3,6,7], 7)), [[2,2,3], [7]]),
    ("combo_sum_ii", sorted(combination_sum_ii([10,1,2,7,6,1,5], 8)), sorted([[1,1,6],[1,2,5],[1,7],[2,6]])),
    ("subsets", len(subsets([1,2,3])), 8),
    ("subsets_dup", len(subsets_with_dup([1,2,2])), 6),
    ("parens", len(generate_parentheses(3)), 5),
    ("letters", sorted(letter_combinations("23")), sorted(["ad","ae","af","bd","be","bf","cd","ce","cf"])),
    ("queens", len(n_queens(4)), 2),
    ("queens_8", len(n_queens(8)), 92),
    ("palindrome", len(partition_palindrome("aab")), 2),
    ("ip", sorted(restore_ip_addresses("25525511135")), sorted(["255.255.11.135","255.255.111.35"])),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
