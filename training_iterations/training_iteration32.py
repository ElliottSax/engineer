def palindrome_number(x):
    """Checks if integer is palindrome without converting to string."""
    if x < 0 or (x % 10 == 0 and x != 0):
        return False
    reversed_half = 0
    while x > reversed_half:
        reversed_half = reversed_half * 10 + x % 10
        x //= 10
    return x == reversed_half or x == reversed_half // 10

def reverse_words(s):
    """Reverses words in string."""
    return ' '.join(s.split()[::-1])

def reverse_words_in_place(s):
    """Reverses each word in string."""
    return ' '.join(word[::-1] for word in s.split())

def rotate_string(s, goal):
    """Checks if goal is rotation of s."""
    return len(s) == len(goal) and goal in s + s

def repeated_substring_pattern(s):
    """Checks if string can be constructed by repeating substring."""
    return s in (s + s)[1:-1]

def strstr(haystack, needle):
    """Finds first occurrence of needle in haystack."""
    if not needle:
        return 0
    n, m = len(haystack), len(needle)
    for i in range(n - m + 1):
        if haystack[i:i+m] == needle:
            return i
    return -1

def kmp_search(text, pattern):
    """KMP string matching algorithm."""
    if not pattern:
        return 0
    # Build failure function
    m = len(pattern)
    failure = [0] * m
    j = 0
    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = failure[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
        failure[i] = j
    # Search
    j = 0
    for i, char in enumerate(text):
        while j > 0 and char != pattern[j]:
            j = failure[j - 1]
        if char == pattern[j]:
            j += 1
        if j == m:
            return i - m + 1
    return -1

def rabin_karp(text, pattern):
    """Rabin-Karp string matching with rolling hash."""
    if not pattern:
        return 0
    n, m = len(text), len(pattern)
    if m > n:
        return -1
    base, mod = 26, 10**9 + 7
    pattern_hash = 0
    text_hash = 0
    power = 1
    for i in range(m - 1):
        power = (power * base) % mod
    for i in range(m):
        pattern_hash = (pattern_hash * base + ord(pattern[i])) % mod
        text_hash = (text_hash * base + ord(text[i])) % mod
    for i in range(n - m + 1):
        if text_hash == pattern_hash and text[i:i+m] == pattern:
            return i
        if i < n - m:
            text_hash = ((text_hash - ord(text[i]) * power) * base + ord(text[i + m])) % mod
            text_hash = (text_hash + mod) % mod
    return -1

def z_algorithm(s):
    """Z-algorithm for pattern matching."""
    n = len(s)
    z = [0] * n
    z[0] = n
    l, r = 0, 0
    for i in range(1, n):
        if i <= r:
            z[i] = min(r - i + 1, z[i - l])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] - 1 > r:
            l, r = i, i + z[i] - 1
    return z

def longest_duplicate_substring(s):
    """Finds longest duplicate substring using binary search + rolling hash."""
    n = len(s)
    base, mod = 26, 2**63 - 1
    nums = [ord(c) - ord('a') for c in s]

    def search(length):
        h = 0
        power = pow(base, length, mod)
        for i in range(length):
            h = (h * base + nums[i]) % mod
        seen = {h: 0}
        for i in range(1, n - length + 1):
            h = (h * base - nums[i - 1] * power + nums[i + length - 1]) % mod
            if h in seen:
                if s[seen[h]:seen[h] + length] == s[i:i + length]:
                    return i
            seen[h] = i
        return -1

    left, right = 1, n - 1
    result = ""
    while left <= right:
        mid = (left + right) // 2
        idx = search(mid)
        if idx != -1:
            result = s[idx:idx + mid]
            left = mid + 1
        else:
            right = mid - 1
    return result

def shortest_palindrome(s):
    """Shortest palindrome by adding characters to front."""
    if not s:
        return s
    rev = s[::-1]
    combined = s + '#' + rev
    n = len(combined)
    lps = [0] * n
    for i in range(1, n):
        j = lps[i - 1]
        while j > 0 and combined[i] != combined[j]:
            j = lps[j - 1]
        if combined[i] == combined[j]:
            j += 1
        lps[i] = j
    return rev[:len(s) - lps[-1]] + s

def count_distinct_substrings(s):
    """Counts distinct substrings using suffix array concept."""
    seen = set()
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            seen.add(s[i:j])
    return len(seen)

# Tests
tests = [
    ("palindrome_num", palindrome_number(121), True),
    ("palindrome_num_neg", palindrome_number(-121), False),
    ("palindrome_num_10", palindrome_number(10), False),
    ("reverse_words", reverse_words("the sky is blue"), "blue is sky the"),
    ("reverse_words_spaces", reverse_words("  hello world  "), "world hello"),
    ("reverse_in_place", reverse_words_in_place("Let's take LeetCode"), "s'teL ekat edoCteeL"),
    ("rotate", rotate_string("abcde", "cdeab"), True),
    ("rotate_no", rotate_string("abcde", "abced"), False),
    ("repeated_sub", repeated_substring_pattern("abab"), True),
    ("repeated_sub_no", repeated_substring_pattern("aba"), False),
    ("strstr", strstr("hello", "ll"), 2),
    ("strstr_miss", strstr("aaaaa", "bba"), -1),
    ("kmp", kmp_search("hello", "ll"), 2),
    ("kmp_2", kmp_search("AABAACAADAABAAABAA", "AABA"), 0),
    ("rabin_karp", rabin_karp("hello world", "world"), 6),
    ("z_algo", z_algorithm("aabcaab")[:4], [7, 1, 0, 0]),
    ("longest_dup", longest_duplicate_substring("banana"), "ana"),
    ("shortest_palin", shortest_palindrome("aacecaaa"), "aaacecaaa"),
    ("distinct_sub", count_distinct_substrings("abc"), 6),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
