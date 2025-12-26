def longest_palindromic_substring_manacher(s):
    """Manacher's algorithm for longest palindromic substring."""
    # Transform string
    t = '#' + '#'.join(s) + '#'
    n = len(t)
    p = [0] * n
    center = right = 0

    for i in range(n):
        if i < right:
            mirror = 2 * center - i
            p[i] = min(right - i, p[mirror])

        # Expand
        while i - p[i] - 1 >= 0 and i + p[i] + 1 < n and t[i - p[i] - 1] == t[i + p[i] + 1]:
            p[i] += 1

        if i + p[i] > right:
            center = i
            right = i + p[i]

    # Find max
    max_len = max(p)
    center_idx = p.index(max_len)
    start = (center_idx - max_len) // 2
    return s[start:start + max_len]

def suffix_array(s):
    """Build suffix array using radix sort."""
    n = len(s)
    sa = list(range(n))
    rank = [ord(c) for c in s]

    k = 1
    while k < n:
        def key(i):
            return (rank[i], rank[i + k] if i + k < n else -1)
        sa.sort(key=key)

        new_rank = [0] * n
        for i in range(1, n):
            new_rank[sa[i]] = new_rank[sa[i-1]]
            if key(sa[i]) != key(sa[i-1]):
                new_rank[sa[i]] += 1
        rank = new_rank
        k *= 2

    return sa

def lcp_array(s, sa):
    """Build LCP array from suffix array using Kasai's algorithm."""
    n = len(s)
    rank = [0] * n
    for i, suffix in enumerate(sa):
        rank[suffix] = i

    lcp = [0] * n
    k = 0
    for i in range(n):
        if rank[i] == 0:
            k = 0
            continue
        j = sa[rank[i] - 1]
        while i + k < n and j + k < n and s[i + k] == s[j + k]:
            k += 1
        lcp[rank[i]] = k
        if k > 0:
            k -= 1

    return lcp

def longest_repeated_substring(s):
    """Find longest repeated substring using suffix array."""
    if not s:
        return ""
    sa = suffix_array(s)
    lcp = lcp_array(s, sa)
    max_lcp = max(lcp)
    if max_lcp == 0:
        return ""
    idx = lcp.index(max_lcp)
    return s[sa[idx]:sa[idx] + max_lcp]

def distinct_substrings_count(s):
    """Count distinct substrings using suffix array."""
    n = len(s)
    if n == 0:
        return 0
    sa = suffix_array(s)
    lcp = lcp_array(s, sa)
    total = n * (n + 1) // 2
    return total - sum(lcp)

def aho_corasick_build(patterns):
    """Build Aho-Corasick automaton."""
    from collections import defaultdict, deque

    class Node:
        def __init__(self):
            self.children = {}
            self.fail = None
            self.output = []

    root = Node()

    # Build trie
    for i, pattern in enumerate(patterns):
        node = root
        for c in pattern:
            if c not in node.children:
                node.children[c] = Node()
            node = node.children[c]
        node.output.append(i)

    # Build failure links
    queue = deque()
    for child in root.children.values():
        child.fail = root
        queue.append(child)

    while queue:
        node = queue.popleft()
        for c, child in node.children.items():
            fail = node.fail
            while fail and c not in fail.children:
                fail = fail.fail
            child.fail = fail.children[c] if fail else root
            child.output = child.output + child.fail.output
            queue.append(child)

    return root

def aho_corasick_search(text, root):
    """Search for all patterns in text."""
    results = []
    node = root
    for i, c in enumerate(text):
        while node and c not in node.children:
            node = node.fail
        node = node.children[c] if node else root
        if node is None:
            node = root
        for pattern_idx in node.output:
            results.append((i, pattern_idx))
    return results

def edit_distance_with_operations(s1, s2):
    """Edit distance with operation sequence."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    # Backtrack
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i-1] == s2[j-1]:
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j-1] <= dp[i-1][j] and dp[i][j-1] <= dp[i-1][j-1]):
            ops.append(('insert', s2[j-1]))
            j -= 1
        elif i > 0 and (j == 0 or dp[i-1][j] <= dp[i][j-1] and dp[i-1][j] <= dp[i-1][j-1]):
            ops.append(('delete', s1[i-1]))
            i -= 1
        else:
            ops.append(('replace', s1[i-1], s2[j-1]))
            i -= 1
            j -= 1

    return dp[m][n], ops[::-1]

def shortest_palindrome_kmp(s):
    """Shortest palindrome by adding chars to front using KMP."""
    if not s:
        return s
    rev = s[::-1]
    concat = s + '#' + rev
    n = len(concat)

    # KMP failure function
    fail = [0] * n
    for i in range(1, n):
        j = fail[i - 1]
        while j > 0 and concat[i] != concat[j]:
            j = fail[j - 1]
        if concat[i] == concat[j]:
            j += 1
        fail[i] = j

    return rev[:len(s) - fail[-1]] + s

def longest_common_substring(s1, s2):
    """Longest common substring using DP."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    end_idx = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    end_idx = i

    return s1[end_idx - max_len:end_idx]

def word_break_all(s, word_dict):
    """All ways to break string into dictionary words."""
    word_set = set(word_dict)
    memo = {}

    def backtrack(start):
        if start in memo:
            return memo[start]
        if start == len(s):
            return [[]]

        result = []
        for end in range(start + 1, len(s) + 1):
            word = s[start:end]
            if word in word_set:
                for rest in backtrack(end):
                    result.append([word] + rest)

        memo[start] = result
        return result

    return [' '.join(words) for words in backtrack(0)]

# Tests
tests = [
    ("manacher", longest_palindromic_substring_manacher("babad") in ["bab", "aba"], True),
    ("manacher_2", longest_palindromic_substring_manacher("cbbd"), "bb"),
    ("suffix_array", suffix_array("banana"), [5, 3, 1, 0, 4, 2]),
    ("lcp", lcp_array("banana", [5, 3, 1, 0, 4, 2]), [0, 1, 3, 0, 0, 2]),
    ("longest_repeat", longest_repeated_substring("banana"), "ana"),
    ("distinct_subs", distinct_substrings_count("abc"), 6),
    ("edit_ops", edit_distance_with_operations("kitten", "sitting")[0], 3),
    ("shortest_palin", shortest_palindrome_kmp("aacecaaa"), "aaacecaaa"),
    ("lcs", longest_common_substring("ABABC", "BABCA"), "BABC"),
    ("word_break", sorted(word_break_all("catsanddog", ["cat","cats","and","sand","dog"])),
     sorted(["cats and dog", "cat sand dog"])),
]

# Aho-Corasick test
patterns = ["he", "she", "his", "hers"]
root = aho_corasick_build(patterns)
matches = aho_corasick_search("ushers", root)
tests.append(("aho_corasick", len(matches), 3))  # she, he, hers

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
