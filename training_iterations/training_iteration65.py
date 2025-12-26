def longest_common_subsequence(text1, text2):
    """Length of longest common subsequence."""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]

def edit_distance(word1, word2):
    """Minimum operations to convert word1 to word2."""
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

def longest_palindromic_subsequence(s):
    """Length of longest palindromic subsequence."""
    n = len(s)
    dp = [[0] * n for _ in range(n)]

    for i in range(n):
        dp[i][i] = 1

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1] + 2
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])

    return dp[0][n-1]

def min_insertions_for_palindrome(s):
    """Minimum insertions to make palindrome."""
    return len(s) - longest_palindromic_subsequence(s)

def distinct_subsequences(s, t):
    """Count distinct subsequences of s equal to t."""
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = 1

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j]
            if s[i-1] == t[j-1]:
                dp[i][j] += dp[i-1][j-1]

    return dp[m][n]

def interleaving_string(s1, s2, s3):
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

def longest_increasing_subsequence(nums):
    """Length of longest increasing subsequence."""
    if not nums:
        return 0

    from bisect import bisect_left
    tails = []

    for num in nums:
        pos = bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num

    return len(tails)

def number_of_lis(nums):
    """Number of longest increasing subsequences."""
    n = len(nums)
    if n == 0:
        return 0

    lengths = [1] * n
    counts = [1] * n

    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                if lengths[j] + 1 > lengths[i]:
                    lengths[i] = lengths[j] + 1
                    counts[i] = counts[j]
                elif lengths[j] + 1 == lengths[i]:
                    counts[i] += counts[j]

    max_len = max(lengths)
    return sum(c for l, c in zip(lengths, counts) if l == max_len)

def russian_doll_envelopes(envelopes):
    """Max envelopes you can put inside another."""
    if not envelopes:
        return 0

    envelopes.sort(key=lambda x: (x[0], -x[1]))
    heights = [e[1] for e in envelopes]

    return longest_increasing_subsequence(heights)

def wildcard_matching(s, p):
    """Wildcard pattern matching."""
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
    """Regular expression with . and *."""
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

def shortest_common_supersequence(str1, str2):
    """Shortest supersequence containing both strings."""
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1])

    return dp[m][n]

# Tests
tests = [
    ("lcs", longest_common_subsequence("abcde", "ace"), 3),
    ("edit", edit_distance("horse", "ros"), 3),
    ("edit_2", edit_distance("intention", "execution"), 5),
    ("lps", longest_palindromic_subsequence("bbbab"), 4),
    ("min_insert", min_insertions_for_palindrome("mbadm"), 2),
    ("distinct", distinct_subsequences("rabbbit", "rabbit"), 3),
    ("interleave", interleaving_string("aabcc", "dbbca", "aadbbcbcac"), True),
    ("interleave_no", interleaving_string("aabcc", "dbbca", "aadbbbaccc"), False),
    ("lis", longest_increasing_subsequence([10,9,2,5,3,7,101,18]), 4),
    ("num_lis", number_of_lis([1,3,5,4,7]), 2),
    ("envelopes", russian_doll_envelopes([[5,4],[6,4],[6,7],[2,3]]), 3),
    ("wildcard", wildcard_matching("adceb", "*a*b"), True),
    ("wildcard_no", wildcard_matching("cb", "?a"), False),
    ("regex", regular_expression_matching("aa", "a*"), True),
    ("regex_2", regular_expression_matching("aab", "c*a*b"), True),
    ("scs", shortest_common_supersequence("abac", "cab"), 5),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
