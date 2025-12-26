# EXTREME: String Matching & Advanced DP

# HARD: KMP Pattern Matching
def kmp_search(text, pattern):
    """KMP string matching - returns all match indices."""
    if not pattern:
        return []

    # Build failure function
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        elif length != 0:
            length = lps[length - 1]
        else:
            lps[i] = 0
            i += 1

    # Search
    matches = []
    n = len(text)
    i = j = 0
    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1
            if j == m:
                matches.append(i - j)
                j = lps[j - 1]
        elif j != 0:
            j = lps[j - 1]
        else:
            i += 1

    return matches

# HARD: Z-Algorithm
def z_algorithm(s):
    """Z-array: z[i] = length of longest substring starting at i matching prefix."""
    n = len(s)
    z = [0] * n
    z[0] = n
    l = r = 0
    for i in range(1, n):
        if i < r:
            z[i] = min(r - i, z[i - l])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] > r:
            l, r = i, i + z[i]
    return z

def z_search(text, pattern):
    """Pattern matching using Z-algorithm."""
    concat = pattern + "$" + text
    z = z_algorithm(concat)
    m = len(pattern)
    return [i - m - 1 for i in range(m + 1, len(concat)) if z[i] == m]

# HARD: Rabin-Karp Rolling Hash
def rabin_karp(text, pattern):
    """Rabin-Karp string matching with rolling hash."""
    if not pattern or len(pattern) > len(text):
        return []

    base = 256
    mod = 10**9 + 7
    m, n = len(pattern), len(text)

    # Compute pattern hash and first window hash
    p_hash = 0
    t_hash = 0
    h = pow(base, m - 1, mod)

    for i in range(m):
        p_hash = (p_hash * base + ord(pattern[i])) % mod
        t_hash = (t_hash * base + ord(text[i])) % mod

    matches = []
    for i in range(n - m + 1):
        if p_hash == t_hash and text[i:i+m] == pattern:
            matches.append(i)
        if i < n - m:
            t_hash = ((t_hash - ord(text[i]) * h) * base + ord(text[i + m])) % mod

    return matches

# HARD: Longest Palindromic Substring (Manacher's)
def manacher(s):
    """Manacher's algorithm for longest palindromic substring."""
    # Transform string
    t = '#' + '#'.join(s) + '#'
    n = len(t)
    p = [0] * n
    c = r = 0

    for i in range(n):
        if i < r:
            p[i] = min(r - i, p[2 * c - i])
        while i - p[i] - 1 >= 0 and i + p[i] + 1 < n and t[i - p[i] - 1] == t[i + p[i] + 1]:
            p[i] += 1
        if i + p[i] > r:
            c, r = i, i + p[i]

    # Find maximum
    max_len = max(p)
    center = p.index(max_len)
    start = (center - max_len) // 2
    return s[start:start + max_len]

# HARD: Edit Distance with Operations
def min_distance_ops(word1, word2):
    """Edit distance returning the operations."""
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n]

# HARD: Regular Expression Matching
def is_match_regex(s, p):
    """Regular expression matching with . and *."""
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    # Handle patterns like a*, a*b*, etc.
    for j in range(2, n + 1, 2):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-2]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '*':
                dp[i][j] = dp[i][j-2]  # zero occurrences
                if p[j-2] == '.' or p[j-2] == s[i-1]:
                    dp[i][j] = dp[i][j] or dp[i-1][j]  # one or more
            elif p[j-1] == '.' or p[j-1] == s[i-1]:
                dp[i][j] = dp[i-1][j-1]

    return dp[m][n]

# HARD: Wildcard Matching
def is_match_wildcard(s, p):
    """Wildcard matching with ? and *."""
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
            elif p[j-1] == '?' or p[j-1] == s[i-1]:
                dp[i][j] = dp[i-1][j-1]

    return dp[m][n]

# HARD: Interleaving String
def is_interleave(s1, s2, s3):
    """Check if s3 is interleaving of s1 and s2."""
    m, n = len(s1), len(s2)
    if m + n != len(s3):
        return False

    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1]
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] and s2[j-1] == s3[j-1]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = (dp[i-1][j] and s1[i-1] == s3[i+j-1]) or \
                       (dp[i][j-1] and s2[j-1] == s3[i+j-1])

    return dp[m][n]

# HARD: Scramble String
def is_scramble(s1, s2):
    """Check if s2 is scrambled version of s1."""
    if len(s1) != len(s2):
        return False
    if s1 == s2:
        return True
    if sorted(s1) != sorted(s2):
        return False

    n = len(s1)
    memo = {}

    def dp(i1, i2, length):
        if (i1, i2, length) in memo:
            return memo[(i1, i2, length)]

        if s1[i1:i1+length] == s2[i2:i2+length]:
            return True

        if sorted(s1[i1:i1+length]) != sorted(s2[i2:i2+length]):
            memo[(i1, i2, length)] = False
            return False

        for k in range(1, length):
            # No swap
            if dp(i1, i2, k) and dp(i1+k, i2+k, length-k):
                memo[(i1, i2, length)] = True
                return True
            # Swap
            if dp(i1, i2+length-k, k) and dp(i1+k, i2, length-k):
                memo[(i1, i2, length)] = True
                return True

        memo[(i1, i2, length)] = False
        return False

    return dp(0, 0, n)

# Tests
tests = [
    # KMP
    ("kmp", kmp_search("AABAACAADAABAAABAA", "AABA"), [0, 9, 13]),
    ("kmp_none", kmp_search("abcd", "xyz"), []),

    # Z-algorithm
    ("z_array", z_algorithm("aabxaab"), [7, 1, 0, 0, 3, 1, 0]),
    ("z_search", z_search("AABAACAADAABAAABAA", "AABA"), [0, 9, 13]),

    # Rabin-Karp
    ("rk", rabin_karp("AABAACAADAABAAABAA", "AABA"), [0, 9, 13]),

    # Manacher
    ("manacher", manacher("babad") in ["bab", "aba"], True),
    ("manacher_long", manacher("cbbd"), "bb"),

    # Edit distance
    ("edit", min_distance_ops("horse", "ros"), 3),
    ("edit2", min_distance_ops("intention", "execution"), 5),

    # Regex
    ("regex1", is_match_regex("aa", "a"), False),
    ("regex2", is_match_regex("aa", "a*"), True),
    ("regex3", is_match_regex("ab", ".*"), True),
    ("regex4", is_match_regex("mississippi", "mis*is*p*."), False),

    # Wildcard
    ("wild1", is_match_wildcard("aa", "a"), False),
    ("wild2", is_match_wildcard("aa", "*"), True),
    ("wild3", is_match_wildcard("cb", "?a"), False),
    ("wild4", is_match_wildcard("adceb", "*a*b"), True),

    # Interleaving
    ("interleave1", is_interleave("aabcc", "dbbca", "aadbbcbcac"), True),
    ("interleave2", is_interleave("aabcc", "dbbca", "aadbbbaccc"), False),

    # Scramble
    ("scramble1", is_scramble("great", "rgeat"), True),
    ("scramble2", is_scramble("abcde", "caebd"), False),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
