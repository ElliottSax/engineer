def maximal_rectangle(matrix):
    """Largest rectangle of 1s in binary matrix."""
    if not matrix:
        return 0
    m, n = len(matrix), len(matrix[0])
    heights = [0] * n
    max_area = 0

    def largest_histogram(heights):
        stack = []
        max_area = 0
        for i, h in enumerate(heights + [0]):
            while stack and heights[stack[-1]] > h:
                height = heights[stack.pop()]
                width = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, height * width)
            stack.append(i)
        return max_area

    for i in range(m):
        for j in range(n):
            heights[j] = heights[j] + 1 if matrix[i][j] == '1' else 0
        max_area = max(max_area, largest_histogram(heights))
    return max_area

def word_search_ii(board, words):
    """Finds all words from dictionary in board."""
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.word = None

    root = TrieNode()
    for word in words:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.word = word

    m, n = len(board), len(board[0])
    result = []

    def dfs(r, c, node):
        char = board[r][c]
        if char not in node.children:
            return
        next_node = node.children[char]
        if next_node.word:
            result.append(next_node.word)
            next_node.word = None

        board[r][c] = '#'
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and board[nr][nc] != '#':
                dfs(nr, nc, next_node)
        board[r][c] = char

    for i in range(m):
        for j in range(n):
            dfs(i, j, root)
    return result

def split_array_largest_sum(nums, m):
    """Minimizes largest sum when splitting array into m parts."""
    def can_split(max_sum):
        count = 1
        current = 0
        for num in nums:
            if current + num > max_sum:
                count += 1
                current = num
            else:
                current += num
        return count <= m

    left, right = max(nums), sum(nums)
    while left < right:
        mid = (left + right) // 2
        if can_split(mid):
            right = mid
        else:
            left = mid + 1
    return left

def find_min_rotated_sorted(nums):
    """Finds minimum in rotated sorted array with duplicates."""
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[right]:
            left = mid + 1
        elif nums[mid] < nums[right]:
            right = mid
        else:
            right -= 1
    return nums[left]

def text_justification(words, max_width):
    """Fully justifies text to max_width."""
    result = []
    line = []
    line_len = 0

    for word in words:
        if line_len + len(word) + len(line) > max_width:
            spaces = max_width - line_len
            if len(line) == 1:
                result.append(line[0] + ' ' * spaces)
            else:
                space_between = spaces // (len(line) - 1)
                extra = spaces % (len(line) - 1)
                justified = ''
                for i, w in enumerate(line[:-1]):
                    justified += w + ' ' * (space_between + (1 if i < extra else 0))
                justified += line[-1]
                result.append(justified)
            line = []
            line_len = 0
        line.append(word)
        line_len += len(word)

    last_line = ' '.join(line)
    result.append(last_line + ' ' * (max_width - len(last_line)))
    return result

def word_break_ii(s, word_dict):
    """All ways to break string into dictionary words."""
    word_set = set(word_dict)
    memo = {}

    def backtrack(start):
        if start in memo:
            return memo[start]
        if start == len(s):
            return ['']
        result = []
        for end in range(start + 1, len(s) + 1):
            word = s[start:end]
            if word in word_set:
                for sub in backtrack(end):
                    if sub:
                        result.append(word + ' ' + sub)
                    else:
                        result.append(word)
        memo[start] = result
        return result

    return backtrack(0)

def concatenated_words(words):
    """Finds words that can be formed from other words in list."""
    word_set = set(words)

    def can_form(word):
        if not word:
            return False
        dp = [False] * (len(word) + 1)
        dp[0] = True
        for i in range(1, len(word) + 1):
            for j in range(i):
                if dp[j] and word[j:i] in word_set and word[j:i] != word:
                    dp[i] = True
                    break
        return dp[len(word)]

    return [word for word in words if can_form(word)]

def alien_order(words):
    """Determines order of characters from sorted alien words."""
    from collections import defaultdict, deque
    graph = defaultdict(set)
    in_degree = {c: 0 for word in words for c in word}

    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        min_len = min(len(w1), len(w2))
        if len(w1) > len(w2) and w1[:min_len] == w2[:min_len]:
            return ""
        for j in range(min_len):
            if w1[j] != w2[j]:
                if w2[j] not in graph[w1[j]]:
                    graph[w1[j]].add(w2[j])
                    in_degree[w2[j]] += 1
                break

    queue = deque([c for c in in_degree if in_degree[c] == 0])
    result = []
    while queue:
        c = queue.popleft()
        result.append(c)
        for neighbor in graph[c]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return ''.join(result) if len(result) == len(in_degree) else ""

def minimum_swaps_binary(s):
    """Minimum swaps to group all 1s together."""
    ones = s.count('1')
    if ones == 0:
        return 0
    zeros_in_window = s[:ones].count('0')
    min_swaps = zeros_in_window
    for i in range(ones, len(s)):
        zeros_in_window += (1 if s[i] == '0' else 0)
        zeros_in_window -= (1 if s[i - ones] == '0' else 0)
        min_swaps = min(min_swaps, zeros_in_window)
    return min_swaps

# Tests
tests = [
    ("maximal_rect", maximal_rectangle([['1','0','1','0','0'],['1','0','1','1','1'],['1','1','1','1','1'],['1','0','0','1','0']]), 6),
    ("word_search_ii", sorted(word_search_ii([["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], ["oath","pea","eat","rain"])), ["eat","oath"]),
    ("split_array", split_array_largest_sum([7,2,5,10,8], 2), 18),
    ("find_min_dup", find_min_rotated_sorted([2,2,2,0,1]), 0),
    ("find_min", find_min_rotated_sorted([3,4,5,1,2]), 1),
    ("text_justify", text_justification(["This","is","an","example","of","text","justification."], 16),
     ["This    is    an", "example  of text", "justification.  "]),
    ("word_break_ii", sorted(word_break_ii("catsanddog", ["cat","cats","and","sand","dog"])),
     sorted(["cat sand dog", "cats and dog"])),
    ("concatenated", sorted(concatenated_words(["cat","cats","catsdogcats","dog","dogcatsdog","hippopotamuses","rat","ratcatdogcat"])),
     sorted(["catsdogcats","dogcatsdog","ratcatdogcat"])),
    ("alien_order", alien_order(["wrt","wrf","er","ett","rftt"]), "wertf"),
    ("alien_order_invalid", alien_order(["abc","ab"]), ""),
    ("min_swaps", minimum_swaps_binary("1101"), 1),
    ("min_swaps_2", minimum_swaps_binary("00011101"), 1),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
