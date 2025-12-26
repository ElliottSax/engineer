from collections import defaultdict, Counter

def two_sum(nums, target):
    """Find indices of two numbers that add to target."""
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

def group_anagrams(strs):
    """Group strings that are anagrams."""
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))
        groups[key].append(s)
    return list(groups.values())

def longest_substring_without_repeat(s):
    """Longest substring without repeating characters."""
    seen = {}
    left = result = 0

    for right, char in enumerate(s):
        if char in seen and seen[char] >= left:
            left = seen[char] + 1
        seen[char] = right
        result = max(result, right - left + 1)

    return result

def subarray_sum_equals_k(nums, k):
    """Count subarrays with sum equals k."""
    count = 0
    prefix_sum = 0
    prefix_counts = {0: 1}

    for num in nums:
        prefix_sum += num
        count += prefix_counts.get(prefix_sum - k, 0)
        prefix_counts[prefix_sum] = prefix_counts.get(prefix_sum, 0) + 1

    return count

def longest_consecutive_sequence_hash(nums):
    """Longest consecutive sequence using hash set."""
    num_set = set(nums)
    max_len = 0

    for num in num_set:
        if num - 1 not in num_set:
            length = 1
            while num + length in num_set:
                length += 1
            max_len = max(max_len, length)

    return max_len

def valid_anagram(s, t):
    """Check if t is anagram of s."""
    return Counter(s) == Counter(t)

def find_all_anagrams(s, p):
    """Find all start indices of p's anagrams in s."""
    result = []
    if len(p) > len(s):
        return result

    p_count = Counter(p)
    s_count = Counter(s[:len(p)])

    if s_count == p_count:
        result.append(0)

    for i in range(len(p), len(s)):
        s_count[s[i]] += 1
        s_count[s[i - len(p)]] -= 1
        if s_count[s[i - len(p)]] == 0:
            del s_count[s[i - len(p)]]
        if s_count == p_count:
            result.append(i - len(p) + 1)

    return result

def most_common_word(paragraph, banned):
    """Most common word not in banned list."""
    banned_set = set(banned)
    words = ''.join(c.lower() if c.isalpha() else ' ' for c in paragraph).split()
    count = Counter(w for w in words if w not in banned_set)
    return count.most_common(1)[0][0]

def isomorphic_strings(s, t):
    """Check if strings are isomorphic."""
    if len(s) != len(t):
        return False
    s_to_t = {}
    t_to_s = {}

    for c1, c2 in zip(s, t):
        if c1 in s_to_t and s_to_t[c1] != c2:
            return False
        if c2 in t_to_s and t_to_s[c2] != c1:
            return False
        s_to_t[c1] = c2
        t_to_s[c2] = c1

    return True

def word_pattern(pattern, s):
    """Check if s follows pattern."""
    words = s.split()
    if len(pattern) != len(words):
        return False

    p_to_w = {}
    w_to_p = {}

    for p, w in zip(pattern, words):
        if p in p_to_w and p_to_w[p] != w:
            return False
        if w in w_to_p and w_to_p[w] != p:
            return False
        p_to_w[p] = w
        w_to_p[w] = p

    return True

def contiguous_array(nums):
    """Longest subarray with equal 0s and 1s."""
    count = 0
    max_len = 0
    count_map = {0: -1}

    for i, num in enumerate(nums):
        count += 1 if num == 1 else -1
        if count in count_map:
            max_len = max(max_len, i - count_map[count])
        else:
            count_map[count] = i

    return max_len

def pairs_divisible_by_60(time):
    """Count pairs with sum divisible by 60."""
    count = 0
    remainder_count = [0] * 60

    for t in time:
        r = t % 60
        complement = (60 - r) % 60
        count += remainder_count[complement]
        remainder_count[r] += 1

    return count

def brick_wall(wall):
    """Minimum bricks to cross."""
    edge_counts = defaultdict(int)

    for row in wall:
        edge = 0
        for brick in row[:-1]:
            edge += brick
            edge_counts[edge] += 1

    return len(wall) - (max(edge_counts.values()) if edge_counts else 0)

# Tests
tests = [
    ("two_sum", two_sum([2,7,11,15], 9), [0, 1]),
    ("anagrams", len(group_anagrams(["eat","tea","tan","ate","nat","bat"])), 3),
    ("no_repeat", longest_substring_without_repeat("abcabcbb"), 3),
    ("subarray_k", subarray_sum_equals_k([1,1,1], 2), 2),
    ("consecutive", longest_consecutive_sequence_hash([100,4,200,1,3,2]), 4),
    ("valid_anagram", valid_anagram("anagram", "nagaram"), True),
    ("find_anagrams", find_all_anagrams("cbaebabacd", "abc"), [0, 6]),
    ("common_word", most_common_word("Bob hit a ball, the hit BALL flew far after it was hit.", ["hit"]), "ball"),
    ("isomorphic", isomorphic_strings("egg", "add"), True),
    ("isomorphic_no", isomorphic_strings("foo", "bar"), False),
    ("word_pattern", word_pattern("abba", "dog cat cat dog"), True),
    ("contiguous", contiguous_array([0,1,0]), 2),
    ("div_60", pairs_divisible_by_60([30,20,150,100,40]), 3),
    ("brick", brick_wall([[1,2,2,1],[3,1,2],[1,3,2],[2,4],[3,1,2],[1,3,1,1]]), 2),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
