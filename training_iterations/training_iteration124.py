#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ULTRA TRAINING ITERATION 124                              ║
║                  Trie Variants & Autocomplete Systems                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from collections import defaultdict
import heapq

# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURE 1: Standard Trie with Word Count
# ═══════════════════════════════════════════════════════════════════════════════
class Trie:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = 0  # Words ending here
        self.prefix_count = 0  # Words with this prefix

    def insert(self, word):
        node = self
        for char in word:
            node.prefix_count += 1
            if char not in node.children:
                node.children[char] = Trie()
            node = node.children[char]
        node.prefix_count += 1
        node.is_end = True
        node.count += 1

    def search(self, word):
        node = self._find_node(word)
        return node is not None and node.is_end

    def starts_with(self, prefix):
        return self._find_node(prefix) is not None

    def count_prefix(self, prefix):
        node = self._find_node(prefix)
        return node.prefix_count if node else 0

    def _find_node(self, s):
        node = self
        for char in s:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURE 2: Autocomplete System with Frequency
# ═══════════════════════════════════════════════════════════════════════════════
class AutocompleteSystem:
    def __init__(self, sentences, times):
        self.root = {}
        self.frequencies = defaultdict(int)
        self.current_input = ""

        for sentence, freq in zip(sentences, times):
            self.frequencies[sentence] = freq
            self._insert(sentence)

    def _insert(self, sentence):
        node = self.root
        for char in sentence:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['#'] = sentence  # Mark end with sentence

    def _collect(self, node, results):
        if '#' in node:
            sentence = node['#']
            results.append((-self.frequencies[sentence], sentence))
        for char, child in node.items():
            if char != '#':
                self._collect(child, results)

    def input(self, c):
        if c == '#':
            # End of input, record sentence
            self.frequencies[self.current_input] += 1
            self._insert(self.current_input)
            self.current_input = ""
            return []

        self.current_input += c

        # Find node for current prefix
        node = self.root
        for char in self.current_input:
            if char not in node:
                return []
            node = node[char]

        # Collect all sentences with this prefix
        results = []
        self._collect(node, results)

        # Return top 3 by frequency (descending), then lexicographically
        heapq.heapify(results)
        top = []
        while results and len(top) < 3:
            _, sentence = heapq.heappop(results)
            top.append(sentence)

        return top

# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURE 3: XOR Trie for Maximum XOR Queries
# ═══════════════════════════════════════════════════════════════════════════════
class XORTrie:
    def __init__(self, max_bits=30):
        self.root = {}
        self.max_bits = max_bits

    def insert(self, num):
        node = self.root
        for i in range(self.max_bits - 1, -1, -1):
            bit = (num >> i) & 1
            if bit not in node:
                node[bit] = {}
            node = node[bit]

    def max_xor(self, num):
        if not self.root:
            return 0

        node = self.root
        result = 0

        for i in range(self.max_bits - 1, -1, -1):
            bit = (num >> i) & 1
            # Try to go opposite direction for max XOR
            opposite = 1 - bit

            if opposite in node:
                result |= (1 << i)
                node = node[opposite]
            elif bit in node:
                node = node[bit]
            else:
                break

        return result

# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURE 4: Compressed Trie (Radix Tree)
# ═══════════════════════════════════════════════════════════════════════════════
class RadixTree:
    def __init__(self):
        self.children = {}  # edge label -> (remaining_label, child_node)
        self.is_end = False
        self.value = None

    def insert(self, word, value=True):
        if not word:
            self.is_end = True
            self.value = value
            return

        first_char = word[0]

        if first_char not in self.children:
            # Create new edge with entire word
            new_node = RadixTree()
            new_node.is_end = True
            new_node.value = value
            self.children[first_char] = (word, new_node)
            return

        edge_label, child = self.children[first_char]

        # Find common prefix
        common_len = 0
        while common_len < len(word) and common_len < len(edge_label) and \
              word[common_len] == edge_label[common_len]:
            common_len += 1

        if common_len == len(edge_label):
            # Edge label is prefix of word
            child.insert(word[common_len:], value)
        else:
            # Split edge
            new_child = RadixTree()
            new_child.children[edge_label[common_len]] = (edge_label[common_len:], child)

            if common_len == len(word):
                new_child.is_end = True
                new_child.value = value
            else:
                new_leaf = RadixTree()
                new_leaf.is_end = True
                new_leaf.value = value
                new_child.children[word[common_len]] = (word[common_len:], new_leaf)

            self.children[first_char] = (edge_label[:common_len], new_child)

    def search(self, word):
        if not word:
            return self.is_end

        first_char = word[0]
        if first_char not in self.children:
            return False

        edge_label, child = self.children[first_char]

        if not word.startswith(edge_label):
            return False

        return child.search(word[len(edge_label):])

# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURE 5: Suffix Trie for Substring Queries
# ═══════════════════════════════════════════════════════════════════════════════
class SuffixTrieOptimized:
    def __init__(self, text):
        self.root = {}
        for i in range(len(text)):
            self._insert_suffix(text[i:])

    def _insert_suffix(self, suffix):
        node = self.root
        for char in suffix:
            if char not in node:
                node[char] = {}
            node = node[char]

    def contains_substring(self, pattern):
        node = self.root
        for char in pattern:
            if char not in node:
                return False
            node = node[char]
        return True

# ═══════════════════════════════════════════════════════════════════════════════
#                              TEST SUITE
# ═══════════════════════════════════════════════════════════════════════════════
def run_tests():
    tests = []

    # Test 1: Standard Trie
    trie = Trie()
    for word in ["apple", "app", "application", "banana"]:
        trie.insert(word)
    tests.append(("trie_search", trie.search("apple"), True))
    tests.append(("trie_prefix", trie.starts_with("app"), True))
    tests.append(("trie_count", trie.count_prefix("app"), 3))

    # Test 2: Autocomplete
    ac = AutocompleteSystem(["i love you", "island", "iroman", "i love leetcode"], [5, 3, 2, 2])
    results = ac.input('i')
    tests.append(("autocomplete", "i love you" in results, True))

    # Test 3: XOR Trie
    xor_trie = XORTrie()
    nums = [3, 10, 5, 25, 2, 8]
    for n in nums:
        xor_trie.insert(n)
    tests.append(("xor_max", xor_trie.max_xor(5), 28))  # 5 XOR 25 = 28

    # Test 4: Radix Tree
    radix = RadixTree()
    radix.insert("test")
    radix.insert("testing")
    radix.insert("team")
    tests.append(("radix_test", radix.search("test"), True))
    tests.append(("radix_testing", radix.search("testing"), True))
    tests.append(("radix_team", radix.search("team"), True))
    tests.append(("radix_tea", radix.search("tea"), False))

    # Test 5: Suffix Trie
    st = SuffixTrieOptimized("banana")
    tests.append(("suffix_ana", st.contains_substring("ana"), True))
    tests.append(("suffix_nan", st.contains_substring("nan"), True))
    tests.append(("suffix_xyz", st.contains_substring("xyz"), False))

    # Run all tests
    passed = 0
    print("\n" + "─" * 60)
    for name, result, expected in tests:
        if result == expected:
            passed += 1
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name}: got {result}, expected {expected}")

    print("─" * 60)
    print(f"\n  📊 Results: {passed}/{len(tests)} tests passed")
    return passed, len(tests)

if __name__ == "__main__":
    print(__doc__)
    passed, total = run_tests()
    if passed == total:
        print("\n  🎯 PERFECT SCORE!")
