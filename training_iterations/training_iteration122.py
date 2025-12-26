# ULTRA: Advanced String Matching III

from collections import defaultdict

# ULTRA: Aho-Corasick Algorithm
class AhoCorasick:
    def __init__(self):
        self.goto = [{}]
        self.fail = [0]
        self.output = [[]]

    def add_pattern(self, pattern, pattern_idx):
        """Add a pattern to the automaton."""
        state = 0
        for char in pattern:
            if char not in self.goto[state]:
                self.goto[state][char] = len(self.goto)
                self.goto.append({})
                self.fail.append(0)
                self.output.append([])
            state = self.goto[state][char]
        self.output[state].append(pattern_idx)

    def build(self):
        """Build failure links using BFS."""
        from collections import deque
        queue = deque()

        # Initialize depth 1 nodes
        for char, state in self.goto[0].items():
            queue.append(state)

        while queue:
            curr = queue.popleft()
            for char, next_state in self.goto[curr].items():
                queue.append(next_state)

                # Find failure link
                failure = self.fail[curr]
                while failure > 0 and char not in self.goto[failure]:
                    failure = self.fail[failure]

                self.fail[next_state] = self.goto[failure].get(char, 0)
                if self.fail[next_state] == next_state:
                    self.fail[next_state] = 0

                # Merge outputs
                self.output[next_state] = self.output[next_state] + self.output[self.fail[next_state]]

    def search(self, text):
        """Search for all patterns in text."""
        results = []
        state = 0

        for i, char in enumerate(text):
            while state > 0 and char not in self.goto[state]:
                state = self.fail[state]
            state = self.goto[state].get(char, 0)

            for pattern_idx in self.output[state]:
                results.append((i, pattern_idx))

        return results

# ULTRA: Rabin-Karp with Rolling Hash
def rabin_karp(text, pattern):
    """Find all occurrences of pattern in text using rolling hash."""
    if not pattern or not text or len(pattern) > len(text):
        return []

    BASE = 256
    MOD = 10**9 + 7
    n, m = len(text), len(pattern)

    # Compute hash of pattern and first window
    pattern_hash = 0
    window_hash = 0
    h = pow(BASE, m - 1, MOD)

    for i in range(m):
        pattern_hash = (pattern_hash * BASE + ord(pattern[i])) % MOD
        window_hash = (window_hash * BASE + ord(text[i])) % MOD

    results = []

    for i in range(n - m + 1):
        if pattern_hash == window_hash:
            # Verify match
            if text[i:i + m] == pattern:
                results.append(i)

        # Roll hash
        if i < n - m:
            window_hash = (window_hash - ord(text[i]) * h) % MOD
            window_hash = (window_hash * BASE + ord(text[i + m])) % MOD
            window_hash = (window_hash + MOD) % MOD

    return results

# ULTRA: Suffix Trie (for pattern matching)
class SuffixTrie:
    def __init__(self, text):
        self.root = {}
        self.text = text
        self._build(text)

    def _build(self, text):
        for i in range(len(text)):
            node = self.root
            for j in range(i, len(text)):
                char = text[j]
                if char not in node:
                    node[char] = {}
                node = node[char]

    def search(self, pattern):
        """Check if pattern exists as substring."""
        node = self.root
        for char in pattern:
            if char not in node:
                return False
            node = node[char]
        return True

    def count_occurrences(self, pattern):
        """Count occurrences of pattern (by counting leaves in subtrie)."""
        node = self.root
        for char in pattern:
            if char not in node:
                return 0
            node = node[char]

        # Count leaf paths
        def count_leaves(n, depth):
            if not n:
                return 1
            return sum(count_leaves(child, depth + 1) for child in n.values())

        return count_leaves(node, 0)

# ULTRA: Longest Common Substring using Suffix Array
def longest_common_substring(s1, s2):
    """Find longest common substring of s1 and s2."""
    if not s1 or not s2:
        return ""

    # Concatenate with separator
    sep = chr(0)
    s = s1 + sep + s2
    n = len(s)

    # Build suffix array (simple O(n log^2 n))
    suffixes = [(s[i:], i) for i in range(n)]
    suffixes.sort()
    sa = [suf[1] for suf in suffixes]

    # Build LCP array
    rank = [0] * n
    for i, (_, idx) in enumerate(suffixes):
        rank[idx] = i

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

    # Find longest LCP where adjacent suffixes are from different strings
    max_len = 0
    max_idx = 0
    len1 = len(s1)

    for i in range(1, n):
        # Check if from different strings
        from_s1_curr = sa[i] < len1
        from_s1_prev = sa[i - 1] < len1
        if from_s1_curr != from_s1_prev and lcp[i] > max_len:
            max_len = lcp[i]
            max_idx = sa[i]

    return s[max_idx:max_idx + max_len]

# ULTRA: Palindrome Factorization Count
def count_palindrome_factorizations(s):
    """Count number of ways to factorize s into palindromes."""
    n = len(s)
    if n == 0:
        return 1

    # Precompute palindrome table
    is_pal = [[False] * n for _ in range(n)]
    for i in range(n):
        is_pal[i][i] = True
    for i in range(n - 1):
        is_pal[i][i + 1] = s[i] == s[i + 1]
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            is_pal[i][j] = s[i] == s[j] and is_pal[i + 1][j - 1]

    # DP: dp[i] = number of factorizations of s[0:i]
    dp = [0] * (n + 1)
    dp[0] = 1

    for i in range(1, n + 1):
        for j in range(i):
            if is_pal[j][i - 1]:
                dp[i] += dp[j]

    return dp[n]

# ULTRA: Tandem Repeat Finder
def find_tandem_repeats(s):
    """Find all tandem repeats (strings of form xx) in s."""
    n = len(s)
    repeats = []

    for length in range(1, n // 2 + 1):
        for i in range(n - 2 * length + 1):
            if s[i:i + length] == s[i + length:i + 2 * length]:
                repeats.append((i, length, s[i:i + length]))

    return repeats

# Tests
tests = []

# Aho-Corasick
ac = AhoCorasick()
patterns = ["he", "she", "his", "hers"]
for i, p in enumerate(patterns):
    ac.add_pattern(p, i)
ac.build()
results = ac.search("ushers")
pattern_ids = [r[1] for r in results]
tests.append(("ac_she", 1 in pattern_ids, True))  # "she" found
tests.append(("ac_he", 0 in pattern_ids, True))   # "he" found

# Rabin-Karp
rk_results = rabin_karp("abababab", "aba")
tests.append(("rk_count", len(rk_results), 3))  # Found at 0, 2, 4
tests.append(("rk_pos", rk_results, [0, 2, 4]))

# Suffix Trie
st = SuffixTrie("banana")
tests.append(("trie_ana", st.search("ana"), True))
tests.append(("trie_xyz", st.search("xyz"), False))

# Longest Common Substring
lcs = longest_common_substring("abcdefg", "xyzabcpqr")
tests.append(("lcs", lcs, "abc"))

# Palindrome Factorizations
tests.append(("pal_fact_a", count_palindrome_factorizations("a"), 1))
tests.append(("pal_fact_aab", count_palindrome_factorizations("aab"), 2))  # "a|a|b" or "aa|b"

# Tandem Repeats
repeats = find_tandem_repeats("abcabc")
tests.append(("tandem", len(repeats), 1))  # "abc" repeated

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
