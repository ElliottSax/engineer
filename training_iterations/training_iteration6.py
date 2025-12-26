def trie_insert_search():
    """Implements Trie with insert and search."""
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_end = False

    root = TrieNode()

    def insert(word):
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(word):
        node = root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    return insert, search

def max_depth_parentheses(s):
    """Returns maximum nesting depth of parentheses."""
    max_depth = current = 0
    for char in s:
        if char == '(':
            current += 1
            max_depth = max(max_depth, current)
        elif char == ')':
            current -= 1
    return max_depth

def remove_duplicates_sorted(nums):
    """Removes duplicates from sorted array in-place, returns new length."""
    if not nums:
        return 0
    write_idx = 1
    for i in range(1, len(nums)):
        if nums[i] != nums[i-1]:
            nums[write_idx] = nums[i]
            write_idx += 1
    return write_idx

def climbing_stairs(n):
    """Number of ways to climb n stairs (1 or 2 steps at a time)."""
    if n <= 2:
        return n
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b

def max_profit_stock(prices):
    """Maximum profit from buying and selling stock once."""
    if not prices:
        return 0
    min_price = prices[0]
    max_profit = 0
    for price in prices[1:]:
        max_profit = max(max_profit, price - min_price)
        min_price = min(min_price, price)
    return max_profit

def reverse_linked_list(head):
    """Reverses linked list (dict with val, next)."""
    prev = None
    current = head
    while current:
        next_node = current.get('next')
        current['next'] = prev
        prev = current
        current = next_node
    return prev

def merge_two_sorted_lists(l1, l2):
    """Merges two sorted linked lists."""
    dummy = {'val': 0, 'next': None}
    current = dummy
    while l1 and l2:
        if l1['val'] <= l2['val']:
            current['next'] = l1
            l1 = l1.get('next')
        else:
            current['next'] = l2
            l2 = l2.get('next')
        current = current['next']
    current['next'] = l1 if l1 else l2
    return dummy['next']

def single_number(nums):
    """Finds number that appears once (others appear twice)."""
    result = 0
    for num in nums:
        result ^= num
    return result

def majority_element(nums):
    """Finds element appearing more than n/2 times (Boyer-Moore)."""
    candidate = None
    count = 0
    for num in nums:
        if count == 0:
            candidate = num
        count += 1 if num == candidate else -1
    return candidate

def move_zeroes(nums):
    """Moves all zeroes to end while maintaining order of non-zeroes."""
    write_idx = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[write_idx], nums[i] = nums[i], nums[write_idx]
            write_idx += 1
    return nums

# Tests
insert, search = trie_insert_search()
insert("apple")
insert("app")
trie_test = search("apple") and search("app") and not search("appl")

tests = [
    ("trie", trie_test, True),
    ("max_depth_parens", max_depth_parentheses("(1+(2*3)+((8)/4))+1"), 3),
    ("remove_dups", remove_duplicates_sorted([1,1,2,2,3]), 3),
    ("climb_stairs", climbing_stairs(5), 8),
    ("max_profit", max_profit_stock([7,1,5,3,6,4]), 5),
    ("single_number", single_number([4,1,2,1,2]), 4),
    ("majority", majority_element([3,2,3]), 3),
    ("move_zeroes", move_zeroes([0,1,0,3,12]), [1,3,12,0,0]),
]

# Linked list tests
l1 = {'val': 1, 'next': {'val': 3, 'next': None}}
l2 = {'val': 2, 'next': {'val': 4, 'next': None}}
merged = merge_two_sorted_lists(l1, l2)
merged_vals = []
while merged:
    merged_vals.append(merged['val'])
    merged = merged.get('next')
tests.append(("merge_lists", merged_vals, [1,2,3,4]))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
