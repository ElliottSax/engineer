def copy_list_with_random_pointer(head):
    """Deep copies linked list with random pointer."""
    if not head:
        return None
    # Interleave copies
    current = head
    while current:
        copy = {'val': current['val'], 'next': current.get('next'), 'random': None}
        current['next'] = copy
        current = copy.get('next')
    # Set random pointers
    current = head
    while current:
        if current.get('random'):
            current['next']['random'] = current['random']['next']
        current = current['next'].get('next')
    # Separate lists
    dummy = {'next': None}
    copy_tail = dummy
    current = head
    while current:
        copy = current['next']
        current['next'] = copy.get('next')
        copy_tail['next'] = copy
        copy_tail = copy
        current = current['next']
    return dummy['next']

def detect_cycle(head):
    """Detects if linked list has cycle."""
    slow = fast = head
    while fast and fast.get('next'):
        slow = slow.get('next')
        fast = fast['next'].get('next')
        if slow == fast:
            return True
    return False

def find_cycle_start(head):
    """Finds start node of cycle in linked list."""
    slow = fast = head
    while fast and fast.get('next'):
        slow = slow.get('next')
        fast = fast['next'].get('next')
        if slow == fast:
            slow = head
            while slow != fast:
                slow = slow.get('next')
                fast = fast.get('next')
            return slow
    return None

def intersection_of_lists(headA, headB):
    """Finds intersection node of two lists."""
    if not headA or not headB:
        return None
    a, b = headA, headB
    while a != b:
        a = a.get('next') if a else headB
        b = b.get('next') if b else headA
    return a

def remove_nth_from_end(head, n):
    """Removes nth node from end."""
    dummy = {'next': head}
    slow = fast = dummy
    for _ in range(n + 1):
        fast = fast.get('next')
    while fast:
        slow = slow.get('next')
        fast = fast.get('next')
    slow['next'] = slow['next'].get('next')
    return dummy['next']

def add_two_numbers(l1, l2):
    """Adds two numbers represented as reversed linked lists."""
    dummy = {'val': 0, 'next': None}
    current = dummy
    carry = 0
    while l1 or l2 or carry:
        x = l1['val'] if l1 else 0
        y = l2['val'] if l2 else 0
        total = x + y + carry
        carry = total // 10
        current['next'] = {'val': total % 10, 'next': None}
        current = current['next']
        l1 = l1.get('next') if l1 else None
        l2 = l2.get('next') if l2 else None
    return dummy['next']

def odd_even_list(head):
    """Groups odd-indexed nodes before even-indexed."""
    if not head:
        return head
    odd = head
    even = even_head = head.get('next')
    while even and even.get('next'):
        odd['next'] = even['next']
        odd = odd['next']
        even['next'] = odd.get('next')
        even = even['next']
    odd['next'] = even_head
    return head

def reorder_list(head):
    """Reorders list: L0→Ln→L1→Ln-1→L2→Ln-2→..."""
    if not head or not head.get('next'):
        return head
    # Find middle
    slow = fast = head
    while fast.get('next') and fast['next'].get('next'):
        slow = slow.get('next')
        fast = fast['next'].get('next')
    # Reverse second half
    prev = None
    current = slow.get('next')
    slow['next'] = None
    while current:
        next_node = current.get('next')
        current['next'] = prev
        prev = current
        current = next_node
    # Merge two halves
    first, second = head, prev
    while second:
        tmp1, tmp2 = first.get('next'), second.get('next')
        first['next'] = second
        second['next'] = tmp1
        first, second = tmp1, tmp2
    return head

def partition_list(head, x):
    """Partitions list around value x."""
    before = before_head = {'val': 0, 'next': None}
    after = after_head = {'val': 0, 'next': None}
    while head:
        if head['val'] < x:
            before['next'] = head
            before = before['next']
        else:
            after['next'] = head
            after = after['next']
        head = head.get('next')
    after['next'] = None
    before['next'] = after_head['next']
    return before_head['next']

def sort_list(head):
    """Sorts linked list using merge sort."""
    if not head or not head.get('next'):
        return head
    # Find middle
    slow = fast = head
    prev = None
    while fast and fast.get('next'):
        prev = slow
        slow = slow.get('next')
        fast = fast['next'].get('next')
    prev['next'] = None
    # Sort halves
    left = sort_list(head)
    right = sort_list(slow)
    # Merge
    dummy = {'next': None}
    current = dummy
    while left and right:
        if left['val'] <= right['val']:
            current['next'] = left
            left = left.get('next')
        else:
            current['next'] = right
            right = right.get('next')
        current = current['next']
    current['next'] = left or right
    return dummy['next']

# Helper to create and traverse list
def make_list(vals):
    dummy = {'next': None}
    current = dummy
    for v in vals:
        current['next'] = {'val': v, 'next': None}
        current = current['next']
    return dummy['next']

def list_to_array(head):
    result = []
    while head:
        result.append(head['val'])
        head = head.get('next')
    return result

# Tests
tests = []

# Detect cycle test
cyclic = make_list([1, 2, 3, 4])
# Make it cyclic
node = cyclic
while node.get('next'):
    node = node['next']
node['next'] = cyclic['next']  # Points back to 2
tests.append(("detect_cycle", detect_cycle(cyclic), True))
tests.append(("detect_no_cycle", detect_cycle(make_list([1, 2, 3])), False))

# Find cycle start
tests.append(("find_cycle_start", find_cycle_start(cyclic)['val'], 2))

# Intersection test
shared = make_list([8, 4, 5])
a = {'val': 4, 'next': {'val': 1, 'next': shared}}
b = {'val': 5, 'next': {'val': 0, 'next': {'val': 1, 'next': shared}}}
tests.append(("intersection", intersection_of_lists(a, b)['val'], 8))

# Remove nth from end
tests.append(("remove_nth", list_to_array(remove_nth_from_end(make_list([1,2,3,4,5]), 2)), [1,2,3,5]))

# Add two numbers
l1 = make_list([2, 4, 3])  # 342
l2 = make_list([5, 6, 4])  # 465
tests.append(("add_numbers", list_to_array(add_two_numbers(l1, l2)), [7, 0, 8]))  # 807

# Odd even list
tests.append(("odd_even", list_to_array(odd_even_list(make_list([1,2,3,4,5]))), [1,3,5,2,4]))

# Reorder list
reorder = make_list([1,2,3,4,5])
reorder_list(reorder)
tests.append(("reorder", list_to_array(reorder), [1,5,2,4,3]))

# Partition list
tests.append(("partition", list_to_array(partition_list(make_list([1,4,3,2,5,2]), 3)), [1,2,2,4,3,5]))

# Sort list
tests.append(("sort", list_to_array(sort_list(make_list([4,2,1,3]))), [1,2,3,4]))

# Copy with random (simple test - check structure)
original = {'val': 1, 'next': {'val': 2, 'next': None, 'random': None}, 'random': None}
original['random'] = original['next']
original['next']['random'] = original
copied = copy_list_with_random_pointer(original)
tests.append(("copy_random", copied['val'] == 1 and copied['random']['val'] == 2, True))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
