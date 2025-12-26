def binary_tree_depth(root):
    """Calculates max depth of binary tree (dict with val, left, right)."""
    if root is None:
        return 0
    left_depth = binary_tree_depth(root.get('left'))
    right_depth = binary_tree_depth(root.get('right'))
    return 1 + max(left_depth, right_depth)

def invert_tree(root):
    """Inverts a binary tree (swaps left and right)."""
    if root is None:
        return None
    root['left'], root['right'] = invert_tree(root.get('right')), invert_tree(root.get('left'))
    return root

def generate_parentheses(n):
    """Generates all valid combinations of n pairs of parentheses."""
    result = []
    def backtrack(s, open_count, close_count):
        if len(s) == 2 * n:
            result.append(s)
            return
        if open_count < n:
            backtrack(s + '(', open_count + 1, close_count)
        if close_count < open_count:
            backtrack(s + ')', open_count, close_count + 1)
    backtrack('', 0, 0)
    return result

def word_break(s, word_dict):
    """Checks if string can be segmented into dictionary words."""
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    word_set = set(word_dict)
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    return dp[n]

def permutations(lst):
    """Generates all permutations of a list."""
    if len(lst) <= 1:
        return [lst[:]]
    result = []
    for i in range(len(lst)):
        rest = lst[:i] + lst[i+1:]
        for p in permutations(rest):
            result.append([lst[i]] + p)
    return result

def combinations(lst, k):
    """Generates all combinations of k elements from list."""
    if k == 0:
        return [[]]
    if not lst:
        return []
    result = []
    for combo in combinations(lst[1:], k - 1):
        result.append([lst[0]] + combo)
    result.extend(combinations(lst[1:], k))
    return result

def power_set(lst):
    """Generates all subsets of a list."""
    if not lst:
        return [[]]
    rest = power_set(lst[1:])
    return rest + [[lst[0]] + subset for subset in rest]

def spiral_matrix(matrix):
    """Returns elements of matrix in spiral order."""
    if not matrix or not matrix[0]:
        return []
    result = []
    top, bottom, left, right = 0, len(matrix) - 1, 0, len(matrix[0]) - 1
    while top <= bottom and left <= right:
        for i in range(left, right + 1):
            result.append(matrix[top][i])
        top += 1
        for i in range(top, bottom + 1):
            result.append(matrix[i][right])
        right -= 1
        if top <= bottom:
            for i in range(right, left - 1, -1):
                result.append(matrix[bottom][i])
            bottom -= 1
        if left <= right:
            for i in range(bottom, top - 1, -1):
                result.append(matrix[i][left])
            left += 1
    return result

# Tests
tests = [
    ("tree_depth", binary_tree_depth({'val':1, 'left':{'val':2,'left':None,'right':None}, 'right':{'val':3,'left':None,'right':None}}), 2),
    ("generate_parens_2", sorted(generate_parentheses(2)), sorted(["(())", "()()"])),
    ("generate_parens_3", len(generate_parentheses(3)), 5),
    ("word_break_yes", word_break("leetcode", ["leet", "code"]), True),
    ("word_break_no", word_break("catsandog", ["cats","dog","sand","and","cat"]), False),
    ("permutations", sorted(permutations([1,2])), sorted([[1,2], [2,1]])),
    ("combinations", sorted(combinations([1,2,3,4], 2)), sorted([[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]])),
    ("power_set", sorted([sorted(x) for x in power_set([1,2])]), sorted([sorted(x) for x in [[], [1], [2], [1,2]]])),
    ("spiral", spiral_matrix([[1,2,3],[4,5,6],[7,8,9]]), [1,2,3,6,9,8,7,4,5]),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
