def longest_word_in_dictionary(words):
    """Longest word that can be built one char at a time."""
    words.sort()
    valid = {''}
    result = ''
    for word in words:
        if word[:-1] in valid:
            valid.add(word)
            if len(word) > len(result):
                result = word
    return result

def replace_words(dictionary, sentence):
    """Replace words with their shortest root in dictionary."""
    roots = set(dictionary)
    result = []
    for word in sentence.split():
        prefix = ''
        for i in range(1, len(word) + 1):
            if word[:i] in roots:
                prefix = word[:i]
                break
        result.append(prefix if prefix else word)
    return ' '.join(result)

def map_sum():
    """Map with sum of values for keys with given prefix."""
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.val = 0

    root = TrieNode()

    def insert(key, val):
        node = root
        for c in key:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.val = val

    def get_sum(prefix):
        node = root
        for c in prefix:
            if c not in node.children:
                return 0
            node = node.children[c]
        return sum_all(node)

    def sum_all(node):
        total = node.val
        for child in node.children.values():
            total += sum_all(child)
        return total

    return insert, get_sum

def autocomplete_system():
    """Autocomplete with frequency-based suggestions."""
    from collections import defaultdict

    class TrieNode:
        def __init__(self):
            self.children = {}
            self.sentences = defaultdict(int)

    root = TrieNode()
    current = [root]
    prefix = ['']

    def insert(sentence, times=1):
        node = root
        for c in sentence:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
            node.sentences[sentence] += times

    def input_char(c):
        if c == '#':
            insert(prefix[0])
            prefix[0] = ''
            current[0] = root
            return []
        prefix[0] += c
        if current[0] and c in current[0].children:
            current[0] = current[0].children[c]
            sentences = sorted(current[0].sentences.items(), key=lambda x: (-x[1], x[0]))
            return [s for s, _ in sentences[:3]]
        current[0] = None
        return []

    return insert, input_char

def stream_checker():
    """Check if suffix of stream matches any word."""
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_end = False

    root = TrieNode()
    stream = []

    def add_word(word):
        node = root
        for c in reversed(word):
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_end = True

    def query(letter):
        stream.append(letter)
        node = root
        for c in reversed(stream):
            if c not in node.children:
                return False
            node = node.children[c]
            if node.is_end:
                return True
        return False

    return add_word, query

def add_search_word():
    """Dictionary with wildcard search support."""
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_end = False

    root = TrieNode()

    def add_word(word):
        node = root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_end = True

    def search(word):
        def dfs(node, idx):
            if idx == len(word):
                return node.is_end
            c = word[idx]
            if c == '.':
                for child in node.children.values():
                    if dfs(child, idx + 1):
                        return True
                return False
            if c not in node.children:
                return False
            return dfs(node.children[c], idx + 1)
        return dfs(root, 0)

    return add_word, search

def concatenated_words_trie(words):
    """Find all words formed by concatenating other words."""
    words = sorted(words, key=len)
    word_set = set()
    result = []

    def can_form(word, idx):
        if idx == len(word):
            return True
        for end in range(idx + 1, len(word) + 1):
            if word[idx:end] in word_set and can_form(word, end):
                return True
        return False

    for word in words:
        if word and can_form(word, 0):
            result.append(word)
        word_set.add(word)
    return result

def palindrome_pairs_trie(words):
    """Find all pairs forming palindromes using trie."""
    def is_palindrome(s):
        return s == s[::-1]

    word_to_idx = {w: i for i, w in enumerate(words)}
    result = []

    for i, word in enumerate(words):
        for j in range(len(word) + 1):
            prefix, suffix = word[:j], word[j:]
            if is_palindrome(prefix):
                rev_suffix = suffix[::-1]
                if rev_suffix in word_to_idx and word_to_idx[rev_suffix] != i:
                    result.append([word_to_idx[rev_suffix], i])
            if j != len(word) and is_palindrome(suffix):
                rev_prefix = prefix[::-1]
                if rev_prefix in word_to_idx and word_to_idx[rev_prefix] != i:
                    result.append([i, word_to_idx[rev_prefix]])
    return result

# Tests
tests = [
    ("longest_word", longest_word_in_dictionary(["w","wo","wor","worl","world"]), "world"),
    ("longest_word_2", longest_word_in_dictionary(["a","banana","app","appl","ap","apply","apple"]), "apple"),
    ("replace_roots", replace_words(["cat","bat","rat"], "the cattle was rattled by the battery"),
     "the cat was rat by the bat"),
]

# Map sum tests
insert_ms, get_sum = map_sum()
insert_ms("apple", 3)
tests.append(("map_sum_1", get_sum("ap"), 3))
insert_ms("app", 2)
tests.append(("map_sum_2", get_sum("ap"), 5))

# Autocomplete tests
insert_ac, input_ac = autocomplete_system()
insert_ac("i love you", 5)
insert_ac("island", 3)
insert_ac("ironman", 2)
tests.append(("autocomplete", input_ac('i'), ["i love you", "island", "ironman"]))

# Stream checker tests
add_stream, query_stream = stream_checker()
add_stream("cd")
add_stream("f")
add_stream("kl")
tests.append(("stream_a", query_stream('a'), False))
tests.append(("stream_d", query_stream('d'), True))

# Add search word tests
add_search, search_word = add_search_word()
add_search("bad")
add_search("dad")
add_search("mad")
tests.append(("search_pad", search_word("pad"), False))
tests.append(("search_wildcard", search_word(".ad"), True))
tests.append(("search_bad", search_word("bad"), True))

# Concatenated words
tests.append(("concat_words", sorted(concatenated_words_trie(["cat","cats","catsdogcats","dog","dogcatsdog","hippopotamuses","rat","ratcatdogcat"])),
              sorted(["catsdogcats","dogcatsdog","ratcatdogcat"])))

# Palindrome pairs
tests.append(("palindrome_pairs", len(palindrome_pairs_trie(["abcd","dcba","lls","s","sssll"])), 4))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
