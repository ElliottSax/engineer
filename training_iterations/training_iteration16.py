def range_sum_query_2d(matrix):
    """2D prefix sum for range queries."""
    if not matrix or not matrix[0]:
        return lambda *args: 0
    m, n = len(matrix), len(matrix[0])
    prefix = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            prefix[i][j] = matrix[i-1][j-1] + prefix[i-1][j] + prefix[i][j-1] - prefix[i-1][j-1]

    def sum_region(r1, c1, r2, c2):
        return prefix[r2+1][c2+1] - prefix[r1][c2+1] - prefix[r2+1][c1] + prefix[r1][c1]
    return sum_region

def kth_smallest_sorted_matrix(matrix, k):
    """Kth smallest in row/col sorted matrix."""
    import heapq
    n = len(matrix)
    heap = [(matrix[0][0], 0, 0)]
    visited = {(0, 0)}
    for _ in range(k):
        val, r, c = heapq.heappop(heap)
        if r + 1 < n and (r + 1, c) not in visited:
            heapq.heappush(heap, (matrix[r+1][c], r+1, c))
            visited.add((r + 1, c))
        if c + 1 < n and (r, c + 1) not in visited:
            heapq.heappush(heap, (matrix[r][c+1], r, c+1))
            visited.add((r, c + 1))
    return val

def search_2d_matrix(matrix, target):
    """Searches sorted 2D matrix (each row sorted, first > prev last)."""
    if not matrix or not matrix[0]:
        return False
    m, n = len(matrix), len(matrix[0])
    left, right = 0, m * n - 1
    while left <= right:
        mid = (left + right) // 2
        val = matrix[mid // n][mid % n]
        if val == target:
            return True
        elif val < target:
            left = mid + 1
        else:
            right = mid - 1
    return False

def search_2d_matrix_ii(matrix, target):
    """Searches row/col sorted matrix (staircase search)."""
    if not matrix or not matrix[0]:
        return False
    m, n = len(matrix), len(matrix[0])
    r, c = 0, n - 1
    while r < m and c >= 0:
        if matrix[r][c] == target:
            return True
        elif matrix[r][c] < target:
            r += 1
        else:
            c -= 1
    return False

def valid_anagram(s, t):
    """Checks if t is anagram of s."""
    from collections import Counter
    return Counter(s) == Counter(t)

def isomorphic_strings(s, t):
    """Checks if strings are isomorphic."""
    if len(s) != len(t):
        return False
    s_to_t = {}
    t_to_s = {}
    for c1, c2 in zip(s, t):
        if c1 in s_to_t:
            if s_to_t[c1] != c2:
                return False
        else:
            s_to_t[c1] = c2
        if c2 in t_to_s:
            if t_to_s[c2] != c1:
                return False
        else:
            t_to_s[c2] = c1
    return True

def word_pattern(pattern, s):
    """Checks if string follows pattern."""
    words = s.split()
    if len(pattern) != len(words):
        return False
    p_to_w = {}
    w_to_p = {}
    for p, w in zip(pattern, words):
        if p in p_to_w:
            if p_to_w[p] != w:
                return False
        else:
            p_to_w[p] = w
        if w in w_to_p:
            if w_to_p[w] != p:
                return False
        else:
            w_to_p[w] = p
    return True

def find_all_anagrams(s, p):
    """Finds all start indices of p's anagrams in s."""
    from collections import Counter
    if len(p) > len(s):
        return []
    p_count = Counter(p)
    s_count = Counter(s[:len(p)])
    result = []
    if s_count == p_count:
        result.append(0)
    for i in range(len(p), len(s)):
        s_count[s[i]] += 1
        old_char = s[i - len(p)]
        s_count[old_char] -= 1
        if s_count[old_char] == 0:
            del s_count[old_char]
        if s_count == p_count:
            result.append(i - len(p) + 1)
    return result

def longest_common_prefix(strs):
    """Finds longest common prefix among strings."""
    if not strs:
        return ""
    prefix = strs[0]
    for s in strs[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix

def decode_string(s):
    """Decodes string like 3[a2[c]] -> accaccacc."""
    stack = []
    current_num = 0
    current_str = ""
    for char in s:
        if char.isdigit():
            current_num = current_num * 10 + int(char)
        elif char == '[':
            stack.append((current_str, current_num))
            current_str = ""
            current_num = 0
        elif char == ']':
            prev_str, num = stack.pop()
            current_str = prev_str + current_str * num
        else:
            current_str += char
    return current_str

def evaluate_rpn(tokens):
    """Evaluates Reverse Polish Notation expression."""
    stack = []
    ops = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: int(a / b)
    }
    for token in tokens:
        if token in ops:
            b, a = stack.pop(), stack.pop()
            stack.append(ops[token](a, b))
        else:
            stack.append(int(token))
    return stack[0]

def simplify_path(path):
    """Simplifies Unix-style absolute path."""
    stack = []
    for part in path.split('/'):
        if part == '..':
            if stack:
                stack.pop()
        elif part and part != '.':
            stack.append(part)
    return '/' + '/'.join(stack)

# Tests
sum_region = range_sum_query_2d([[3,0,1,4,2],[5,6,3,2,1],[1,2,0,1,5],[4,1,0,1,7],[1,0,3,0,5]])

tests = [
    ("range_sum", sum_region(2, 1, 4, 3), 8),
    ("range_sum_2", sum_region(1, 1, 2, 2), 11),
    ("kth_smallest", kth_smallest_sorted_matrix([[1,5,9],[10,11,13],[12,13,15]], 8), 13),
    ("search_2d", search_2d_matrix([[1,3,5,7],[10,11,16,20],[23,30,34,60]], 3), True),
    ("search_2d_no", search_2d_matrix([[1,3,5,7],[10,11,16,20],[23,30,34,60]], 13), False),
    ("search_2d_ii", search_2d_matrix_ii([[1,4,7,11],[2,5,8,12],[3,6,9,16],[10,13,14,17]], 5), True),
    ("valid_anagram", valid_anagram("anagram", "nagaram"), True),
    ("valid_anagram_no", valid_anagram("rat", "car"), False),
    ("isomorphic", isomorphic_strings("egg", "add"), True),
    ("isomorphic_no", isomorphic_strings("foo", "bar"), False),
    ("word_pattern", word_pattern("abba", "dog cat cat dog"), True),
    ("find_anagrams", find_all_anagrams("cbaebabacd", "abc"), [0, 6]),
    ("lcp", longest_common_prefix(["flower","flow","flight"]), "fl"),
    ("decode_str", decode_string("3[a2[c]]"), "accaccacc"),
    ("rpn", evaluate_rpn(["2","1","+","3","*"]), 9),
    ("simplify", simplify_path("/a/./b/../../c/"), "/c"),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
