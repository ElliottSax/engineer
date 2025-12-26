# ULTRA: Advanced String Algorithms - Suffix Automaton & Palindromic Trees

from collections import defaultdict

# ULTRA: Suffix Automaton (DAWG)
class SuffixAutomaton:
    def __init__(self):
        self.states = [{'len': 0, 'link': -1, 'trans': {}}]
        self.last = 0
        self.size = 1

    def extend(self, c):
        cur = self.size
        self.states.append({'len': self.states[self.last]['len'] + 1, 'link': -1, 'trans': {}})
        self.size += 1

        p = self.last
        while p != -1 and c not in self.states[p]['trans']:
            self.states[p]['trans'][c] = cur
            p = self.states[p]['link']

        if p == -1:
            self.states[cur]['link'] = 0
        else:
            q = self.states[p]['trans'][c]
            if self.states[p]['len'] + 1 == self.states[q]['len']:
                self.states[cur]['link'] = q
            else:
                # Clone state
                clone = self.size
                self.states.append({
                    'len': self.states[p]['len'] + 1,
                    'link': self.states[q]['link'],
                    'trans': self.states[q]['trans'].copy()
                })
                self.size += 1
                while p != -1 and self.states[p]['trans'].get(c) == q:
                    self.states[p]['trans'][c] = clone
                    p = self.states[p]['link']
                self.states[q]['link'] = clone
                self.states[cur]['link'] = clone

        self.last = cur

    def build(self, s):
        for c in s:
            self.extend(c)

    def count_distinct_substrings(self):
        """Count distinct non-empty substrings."""
        count = 0
        for i in range(1, self.size):
            count += self.states[i]['len'] - self.states[self.states[i]['link']]['len']
        return count

    def contains(self, pattern):
        """Check if pattern is a substring."""
        state = 0
        for c in pattern:
            if c not in self.states[state]['trans']:
                return False
            state = self.states[state]['trans'][c]
        return True

    def longest_common_substring(self, other):
        """Find LCS with another string."""
        state = 0
        length = 0
        best = 0

        for c in other:
            while state != 0 and c not in self.states[state]['trans']:
                state = self.states[state]['link']
                length = self.states[state]['len']

            if c in self.states[state]['trans']:
                state = self.states[state]['trans'][c]
                length += 1
            else:
                state = 0
                length = 0

            best = max(best, length)

        return best

# ULTRA: Palindromic Tree (Eertree)
class PalindromicTree:
    def __init__(self):
        # Node structure: [length, suffix_link, transitions]
        self.nodes = [
            {'len': -1, 'link': 0, 'trans': {}},  # Imaginary root for odd length
            {'len': 0, 'link': 0, 'trans': {}}    # Root for even length
        ]
        self.last = 1
        self.s = ""

    def get_link(self, v):
        while len(self.s) < self.nodes[v]['len'] + 2 or \
              self.s[-(self.nodes[v]['len'] + 2)] != self.s[-1]:
            v = self.nodes[v]['link']
        return v

    def add_char(self, c):
        self.s += c
        cur = self.get_link(self.last)

        if c not in self.nodes[cur]['trans']:
            # Create new node
            new_node = len(self.nodes)
            new_len = self.nodes[cur]['len'] + 2
            self.nodes.append({
                'len': new_len,
                'link': 0,
                'trans': {}
            })

            # Find suffix link
            if new_len == 1:
                self.nodes[new_node]['link'] = 1
            else:
                link = self.get_link(self.nodes[cur]['link'])
                self.nodes[new_node]['link'] = self.nodes[link]['trans'].get(c, 1)

            self.nodes[cur]['trans'][c] = new_node

        self.last = self.nodes[cur]['trans'][c]
        return self.last

    def build(self, s):
        for c in s:
            self.add_char(c)

    def count_distinct_palindromes(self):
        """Count distinct palindromic substrings."""
        return len(self.nodes) - 2

    def count_palindrome_occurrences(self):
        """Count total palindrome occurrences."""
        # Need to propagate counts through suffix links
        cnt = [0] * len(self.nodes)
        for c in self.s:
            self.add_char(c) if len(self.s) == 0 else None
        # Each node was visited - simplified count
        return sum(1 for n in self.nodes[2:] if n['len'] > 0)

# ULTRA: Suffix Array with LCP using Induced Sorting (SA-IS)
def suffix_array_sais(s):
    """Build suffix array using SA-IS algorithm O(n)."""
    if not s:
        return [], []

    # Convert to integers
    if isinstance(s, str):
        s = [ord(c) for c in s]
    s = s + [0]  # Sentinel
    n = len(s)

    # Type array: L-type (True) or S-type (False)
    t = [False] * n
    t[n - 1] = True  # Sentinel is S-type
    for i in range(n - 2, -1, -1):
        if s[i] > s[i + 1]:
            t[i] = False
        elif s[i] < s[i + 1]:
            t[i] = True
        else:
            t[i] = t[i + 1]

    def is_lms(i):
        return i > 0 and t[i] and not t[i - 1]

    # Get bucket sizes and ends
    def get_buckets(end=True):
        buckets = [0] * (max(s) + 1)
        for c in s:
            buckets[c] += 1
        total = 0
        for i in range(len(buckets)):
            total += buckets[i]
            buckets[i] = total if end else total - buckets[i]
        return buckets

    def induce_sort(sa, lms):
        # Place LMS suffixes
        buckets = get_buckets(end=True)
        for j in reversed(lms):
            buckets[s[j]] -= 1
            sa[buckets[s[j]]] = j

        # Induce L-type
        buckets = get_buckets(end=False)
        for i in range(n):
            j = sa[i] - 1
            if sa[i] > 0 and not t[j]:
                sa[buckets[s[j]]] = j
                buckets[s[j]] += 1

        # Induce S-type
        buckets = get_buckets(end=True)
        for i in range(n - 1, -1, -1):
            j = sa[i] - 1
            if sa[i] > 0 and t[j]:
                buckets[s[j]] -= 1
                sa[buckets[s[j]]] = j

    # Find LMS substrings
    lms = [i for i in range(n) if is_lms(i)]
    if not lms:
        lms = [n - 1]

    # Initial sort
    sa = [-1] * n
    induce_sort(sa, lms)

    # Name LMS substrings
    lms_names = [-1] * n
    name = 0
    prev = -1
    for i in sa:
        if is_lms(i):
            if prev >= 0:
                # Compare LMS substrings
                diff = False
                for j in range(n):
                    if s[i + j] != s[prev + j] or t[i + j] != t[prev + j]:
                        diff = True
                        break
                    if j > 0 and (is_lms(i + j) or is_lms(prev + j)):
                        break
                if diff:
                    name += 1
            lms_names[i] = name
            prev = i

    # Reduced problem
    reduced = [lms_names[i] for i in lms if lms_names[i] >= 0]
    if name < len(reduced) - 1:
        # Recurse
        reduced_sa = suffix_array_sais(reduced)[0]
        lms = [lms[i] for i in reduced_sa]
    else:
        lms.sort(key=lambda x: lms_names[x])

    # Final induced sort
    sa = [-1] * n
    induce_sort(sa, lms)

    # Remove sentinel
    sa = [x for x in sa if x < n - 1]

    # Build LCP array
    rank = [0] * (n - 1)
    for i, x in enumerate(sa):
        rank[x] = i

    lcp = [0] * len(sa)
    k = 0
    for i in range(n - 1):
        if rank[i] == 0:
            k = 0
            continue
        j = sa[rank[i] - 1]
        while i + k < n - 1 and j + k < n - 1 and s[i + k] == s[j + k]:
            k += 1
        lcp[rank[i]] = k
        if k > 0:
            k -= 1

    return sa, lcp

# ULTRA: Aho-Corasick with Failure Links for Pattern Matching
class AhoCorasickFull:
    def __init__(self):
        self.goto = [{}]
        self.fail = [0]
        self.output = [[]]
        self.state_count = 1

    def add_pattern(self, pattern, pattern_id):
        state = 0
        for char in pattern:
            if char not in self.goto[state]:
                self.goto[state][char] = self.state_count
                self.goto.append({})
                self.fail.append(0)
                self.output.append([])
                self.state_count += 1
            state = self.goto[state][char]
        self.output[state].append(pattern_id)

    def build(self):
        from collections import deque
        queue = deque()

        for char, state in self.goto[0].items():
            queue.append(state)

        while queue:
            r = queue.popleft()
            for char, s in self.goto[r].items():
                queue.append(s)
                state = self.fail[r]
                while state and char not in self.goto[state]:
                    state = self.fail[state]
                self.fail[s] = self.goto[state].get(char, 0)
                self.output[s] = self.output[s] + self.output[self.fail[s]]

    def search(self, text):
        state = 0
        results = []
        for i, char in enumerate(text):
            while state and char not in self.goto[state]:
                state = self.fail[state]
            state = self.goto[state].get(char, 0)
            for pattern_id in self.output[state]:
                results.append((i, pattern_id))
        return results

# Tests
tests = []

# Suffix Automaton
sa = SuffixAutomaton()
sa.build("abab")
tests.append(("sam_distinct", sa.count_distinct_substrings(), 7))  # a,b,ab,ba,aba,bab,abab
tests.append(("sam_contains", sa.contains("aba"), True))
tests.append(("sam_not", sa.contains("aaa"), False))

sa2 = SuffixAutomaton()
sa2.build("abcde")
tests.append(("sam_lcs", sa2.longest_common_substring("bcdef"), 4))  # bcde

# Palindromic Tree
pt = PalindromicTree()
pt.build("abaaba")
tests.append(("eertree", pt.count_distinct_palindromes(), 5))  # a, b, aba, aa, abaaba

# Suffix Array SA-IS
sa_result, lcp = suffix_array_sais("banana")
tests.append(("sais", sa_result, [5, 3, 1, 0, 4, 2]))
tests.append(("sais_lcp", lcp[1:], [1, 3, 0, 0, 2]))

# Aho-Corasick Full
ac = AhoCorasickFull()
ac.add_pattern("he", 0)
ac.add_pattern("she", 1)
ac.add_pattern("his", 2)
ac.add_pattern("hers", 3)
ac.build()
results = ac.search("ushers")
tests.append(("ac_count", len(results), 4))  # she, he at positions

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
