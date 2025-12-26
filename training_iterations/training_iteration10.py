def n_queens(n):
    """Counts solutions to N-Queens problem."""
    def is_safe(board, row, col):
        for i in range(row):
            if board[i] == col or abs(board[i] - col) == row - i:
                return False
        return True
    def solve(row):
        if row == n:
            return 1
        count = 0
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                count += solve(row + 1)
        return count
    board = [-1] * n
    return solve(0)

def wildcard_matching(s, p):
    """Wildcard pattern matching (* = any seq, ? = any char)."""
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for j in range(1, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-1]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '*':
                dp[i][j] = dp[i-1][j] or dp[i][j-1]
            elif p[j-1] == '?' or s[i-1] == p[j-1]:
                dp[i][j] = dp[i-1][j-1]
    return dp[m][n]

def regular_expression_matching(s, p):
    """Regex matching (. = any char, * = 0+ of preceding)."""
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for j in range(2, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-2]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '*':
                dp[i][j] = dp[i][j-2]
                if p[j-2] == '.' or p[j-2] == s[i-1]:
                    dp[i][j] = dp[i][j] or dp[i-1][j]
            elif p[j-1] == '.' or p[j-1] == s[i-1]:
                dp[i][j] = dp[i-1][j-1]
    return dp[m][n]

def min_cost_climbing(cost):
    """Minimum cost to climb stairs (can start at 0 or 1)."""
    n = len(cost)
    if n <= 1:
        return 0
    a, b = cost[0], cost[1]
    for i in range(2, n):
        a, b = b, cost[i] + min(a, b)
    return min(a, b)

def rob_houses_circular(nums):
    """Maximum robbery from circular houses (can't rob adjacent or first+last)."""
    if len(nums) == 1:
        return nums[0]
    def rob_linear(houses):
        if not houses:
            return 0
        if len(houses) == 1:
            return houses[0]
        a, b = houses[0], max(houses[0], houses[1])
        for i in range(2, len(houses)):
            a, b = b, max(b, a + houses[i])
        return b
    return max(rob_linear(nums[:-1]), rob_linear(nums[1:]))

def longest_valid_parentheses(s):
    """Length of longest valid parentheses substring."""
    max_len = 0
    stack = [-1]
    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)
            else:
                max_len = max(max_len, i - stack[-1])
    return max_len

def candy_distribution(ratings):
    """Minimum candies so higher rated gets more than neighbors."""
    n = len(ratings)
    candies = [1] * n
    for i in range(1, n):
        if ratings[i] > ratings[i-1]:
            candies[i] = candies[i-1] + 1
    for i in range(n-2, -1, -1):
        if ratings[i] > ratings[i+1]:
            candies[i] = max(candies[i], candies[i+1] + 1)
    return sum(candies)

def find_celebrity(n, knows):
    """Find celebrity (known by all, knows no one)."""
    candidate = 0
    for i in range(1, n):
        if knows(candidate, i):
            candidate = i
    for i in range(n):
        if i != candidate:
            if knows(candidate, i) or not knows(i, candidate):
                return -1
    return candidate

def palindrome_pairs(words):
    """Finds all pairs (i,j) where words[i]+words[j] is palindrome."""
    def is_palindrome(s):
        return s == s[::-1]
    word_to_idx = {w: i for i, w in enumerate(words)}
    result = []
    for i, word in enumerate(words):
        for j in range(len(word) + 1):
            prefix, suffix = word[:j], word[j:]
            if is_palindrome(prefix):
                rev_suffix = suffix[::-1]
                if rev_suffix in word_to_idx and word_to_idx[rev_suffix] != i:
                    result.append([word_to_idx[rev_suffix], i])
            if j != len(word) and is_palindrome(suffix):
                rev_prefix = prefix[::-1]
                if rev_prefix in word_to_idx and word_to_idx[rev_prefix] != i:
                    result.append([i, word_to_idx[rev_prefix]])
    return result

# Tests
tests = [
    ("n_queens_4", n_queens(4), 2),
    ("n_queens_8", n_queens(8), 92),
    ("wildcard_yes", wildcard_matching("adceb", "*a*b"), True),
    ("wildcard_no", wildcard_matching("cb", "?a"), False),
    ("regex_yes", regular_expression_matching("aab", "c*a*b"), True),
    ("regex_no", regular_expression_matching("mississippi", "mis*is*p*."), False),
    ("min_cost_climb", min_cost_climbing([10,15,20]), 15),
    ("rob_circular", rob_houses_circular([2,3,2]), 3),
    ("longest_valid_parens", longest_valid_parentheses(")()())"), 4),
    ("candy", candy_distribution([1,0,2]), 5),
    ("palindrome_pairs", len(palindrome_pairs(["abcd","dcba","lls","s","sssll"])), 4),
]

# Celebrity test
def create_knows(celebrity, n):
    def knows(a, b):
        if a == celebrity:
            return False
        return b == celebrity
    return knows
tests.append(("celebrity", find_celebrity(3, create_knows(1, 3)), 1))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
