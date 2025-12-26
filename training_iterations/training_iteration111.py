# ULTRA: Advanced String Algorithms II

from collections import defaultdict

# ULTRA: Z-Algorithm
def z_function(s):
    """Compute Z-array where Z[i] = length of longest substring starting at i
    that matches a prefix of s."""
    n = len(s)
    z = [0] * n
    z[0] = n
    l, r = 0, 0

    for i in range(1, n):
        if i < r:
            z[i] = min(r - i, z[i - l])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] > r:
            l, r = i, i + z[i]

    return z

# ULTRA: Manacher's Algorithm for Longest Palindromic Substring
def manacher(s):
    """Find longest palindromic substring using Manacher's algorithm."""
    # Transform: "abc" -> "^#a#b#c#$"
    t = "^#" + "#".join(s) + "#$"
    n = len(t)
    p = [0] * n
    center = right = 0

    for i in range(1, n - 1):
        if i < right:
            mirror = 2 * center - i
            p[i] = min(right - i, p[mirror])

        while t[i + p[i] + 1] == t[i - p[i] - 1]:
            p[i] += 1

        if i + p[i] > right:
            center, right = i, i + p[i]

    # Find max
    max_len = max(p)
    center_idx = p.index(max_len)
    start = (center_idx - max_len) // 2
    return s[start:start + max_len]

# ULTRA: Booth's Algorithm (Lexicographically Minimum Rotation)
def min_rotation(s):
    """Find lexicographically minimum rotation of s."""
    s = s + s
    n = len(s)
    f = [-1] * n
    k = 0

    for j in range(1, n):
        sj = s[j]
        i = f[j - k - 1]

        while i != -1 and sj != s[k + i + 1]:
            if sj < s[k + i + 1]:
                k = j - i - 1
            i = f[i]

        if sj != s[k + i + 1]:
            if sj < s[k]:
                k = j
            f[j - k] = -1
        else:
            f[j - k] = i + 1

    return k

# ULTRA: Lyndon Factorization (Duval's Algorithm)
def lyndon_factorization(s):
    """Decompose s into Lyndon words."""
    n = len(s)
    result = []
    i = 0

    while i < n:
        j, k = i, i + 1
        while k < n and s[j] <= s[k]:
            if s[j] < s[k]:
                j = i
            else:
                j += 1
            k += 1

        while i <= j:
            result.append(s[i:i + k - j])
            i += k - j

    return result

# ULTRA: Longest Repeated Substring using Suffix Array
def longest_repeated_substring(s):
    """Find longest substring that appears at least twice."""
    if not s:
        return ""

    n = len(s)

    # Build suffix array using simple O(n log^2 n) method
    suffixes = [(s[i:], i) for i in range(n)]
    suffixes.sort()
    sa = [suf[1] for suf in suffixes]

    # Compute LCP array
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

    # Find max LCP
    max_lcp = max(lcp) if lcp else 0
    if max_lcp == 0:
        return ""

    idx = lcp.index(max_lcp)
    return s[sa[idx]:sa[idx] + max_lcp]

# ULTRA: Minimum Palindrome Partition
def min_palindrome_partition(s):
    """Minimum cuts needed to partition s into palindromes."""
    n = len(s)
    if n == 0:
        return 0

    # is_palindrome[i][j] = True if s[i:j+1] is palindrome
    is_pal = [[False] * n for _ in range(n)]

    for i in range(n):
        is_pal[i][i] = True

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if length == 2:
                is_pal[i][j] = s[i] == s[j]
            else:
                is_pal[i][j] = s[i] == s[j] and is_pal[i + 1][j - 1]

    # dp[i] = min cuts for s[0:i+1]
    dp = [0] * n

    for i in range(n):
        if is_pal[0][i]:
            dp[i] = 0
        else:
            dp[i] = i  # max cuts
            for j in range(i):
                if is_pal[j + 1][i]:
                    dp[i] = min(dp[i], dp[j] + 1)

    return dp[n - 1]

# ULTRA: String Hashing with Double Hash
class StringHash:
    def __init__(self, s, mod1=10**9+7, mod2=10**9+9, base1=31, base2=37):
        self.n = len(s)
        self.mod1, self.mod2 = mod1, mod2
        self.base1, self.base2 = base1, base2

        # Precompute hashes
        self.hash1 = [0] * (self.n + 1)
        self.hash2 = [0] * (self.n + 1)
        self.pow1 = [1] * (self.n + 1)
        self.pow2 = [1] * (self.n + 1)

        for i in range(self.n):
            self.hash1[i + 1] = (self.hash1[i] * base1 + ord(s[i])) % mod1
            self.hash2[i + 1] = (self.hash2[i] * base2 + ord(s[i])) % mod2
            self.pow1[i + 1] = (self.pow1[i] * base1) % mod1
            self.pow2[i + 1] = (self.pow2[i] * base2) % mod2

    def get_hash(self, l, r):
        """Get hash of substring s[l:r+1]."""
        h1 = (self.hash1[r + 1] - self.hash1[l] * self.pow1[r - l + 1]) % self.mod1
        h2 = (self.hash2[r + 1] - self.hash2[l] * self.pow2[r - l + 1]) % self.mod2
        return (h1, h2)

# Tests
tests = []

# Z-function
z = z_function("aabxaab")
tests.append(("z_func", z, [7, 1, 0, 0, 3, 1, 0]))

# Z-function pattern matching
def z_pattern_match(text, pattern):
    combined = pattern + "$" + text
    z = z_function(combined)
    matches = []
    for i in range(len(pattern) + 1, len(combined)):
        if z[i] == len(pattern):
            matches.append(i - len(pattern) - 1)
    return matches

tests.append(("z_match", z_pattern_match("abcabcabc", "abc"), [0, 3, 6]))

# Manacher
tests.append(("manacher", manacher("babad") in ["bab", "aba"], True))
tests.append(("manacher2", manacher("cbbd"), "bb"))

# Min rotation
tests.append(("min_rot", min_rotation("bca"), 1))  # "abc" starts at index 1
tests.append(("min_rot2", min_rotation("cba"), 2))  # "acb" starts at index 2

# Lyndon factorization
tests.append(("lyndon", lyndon_factorization("abab"), ["ab", "ab"]))
tests.append(("lyndon2", lyndon_factorization("abaab"), ["ab", "aab"]))

# Longest repeated substring
tests.append(("lrs", longest_repeated_substring("banana"), "ana"))
tests.append(("lrs2", longest_repeated_substring("abcd"), ""))

# Min palindrome partition
tests.append(("min_pal", min_palindrome_partition("aab"), 1))  # "aa" | "b"
tests.append(("min_pal2", min_palindrome_partition("aba"), 0))  # already palindrome

# String hashing
sh = StringHash("abcabc")
tests.append(("hash_eq", sh.get_hash(0, 2) == sh.get_hash(3, 5), True))
tests.append(("hash_neq", sh.get_hash(0, 1) == sh.get_hash(1, 2), False))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
