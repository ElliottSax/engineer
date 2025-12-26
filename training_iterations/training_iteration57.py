class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def to_linked_list(arr):
    if not arr:
        return None
    head = ListNode(arr[0])
    curr = head
    for val in arr[1:]:
        curr.next = ListNode(val)
        curr = curr.next
    return head

def to_array(head):
    result = []
    while head:
        result.append(head.val)
        head = head.next
    return result

def reverse_linked_list(head):
    """Reverse a linked list."""
    prev = None
    while head:
        next_node = head.next
        head.next = prev
        prev = head
        head = next_node
    return prev

def reverse_between(head, left, right):
    """Reverse from position left to right."""
    dummy = ListNode(0, head)
    prev = dummy

    for _ in range(left - 1):
        prev = prev.next

    curr = prev.next
    for _ in range(right - left):
        next_node = curr.next
        curr.next = next_node.next
        next_node.next = prev.next
        prev.next = next_node

    return dummy.next

def merge_two_sorted(l1, l2):
    """Merge two sorted linked lists."""
    dummy = ListNode()
    curr = dummy

    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next

    curr.next = l1 or l2
    return dummy.next

def remove_nth_from_end(head, n):
    """Remove nth node from end."""
    dummy = ListNode(0, head)
    fast = slow = dummy

    for _ in range(n + 1):
        fast = fast.next

    while fast:
        fast = fast.next
        slow = slow.next

    slow.next = slow.next.next
    return dummy.next

def detect_cycle(head):
    """Return node where cycle begins."""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next
            return slow
    return None

def reorder_list(head):
    """Reorder: L0→Ln→L1→Ln-1→L2→Ln-2→..."""
    if not head or not head.next:
        return head

    # Find middle
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next

    # Reverse second half
    prev = None
    curr = slow.next
    slow.next = None
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node

    # Merge
    first, second = head, prev
    while second:
        tmp1, tmp2 = first.next, second.next
        first.next = second
        second.next = tmp1
        first, second = tmp1, tmp2

    return head

def add_two_numbers(l1, l2):
    """Add two numbers represented as linked lists (reversed)."""
    dummy = ListNode()
    curr = dummy
    carry = 0

    while l1 or l2 or carry:
        val = carry
        if l1:
            val += l1.val
            l1 = l1.next
        if l2:
            val += l2.val
            l2 = l2.next
        carry, val = divmod(val, 10)
        curr.next = ListNode(val)
        curr = curr.next

    return dummy.next

def partition_list(head, x):
    """Partition list around x."""
    before = before_head = ListNode()
    after = after_head = ListNode()

    while head:
        if head.val < x:
            before.next = head
            before = before.next
        else:
            after.next = head
            after = after.next
        head = head.next

    after.next = None
    before.next = after_head.next
    return before_head.next

def copy_random_list(head):
    """Copy list with random pointers."""
    if not head:
        return None

    # Create interleaved list
    curr = head
    while curr:
        copy = ListNode(curr.val)
        copy.next = curr.next
        curr.next = copy
        curr = copy.next

    # Copy random pointers
    curr = head
    while curr:
        if hasattr(curr, 'random') and curr.random:
            curr.next.random = curr.random.next
        curr = curr.next.next

    # Separate lists
    dummy = ListNode()
    copy_curr = dummy
    curr = head
    while curr:
        copy_curr.next = curr.next
        copy_curr = copy_curr.next
        curr.next = curr.next.next
        curr = curr.next

    return dummy.next

def swap_pairs(head):
    """Swap every two adjacent nodes."""
    dummy = ListNode(0, head)
    prev = dummy

    while prev.next and prev.next.next:
        first = prev.next
        second = prev.next.next
        prev.next = second
        first.next = second.next
        second.next = first
        prev = first

    return dummy.next

def reverse_k_group(head, k):
    """Reverse nodes in k-groups."""
    dummy = ListNode(0, head)
    prev = dummy

    while True:
        # Check if k nodes exist
        curr = prev
        for _ in range(k):
            curr = curr.next
            if not curr:
                return dummy.next

        # Reverse k nodes
        curr = prev.next
        for _ in range(k - 1):
            next_node = curr.next
            curr.next = next_node.next
            next_node.next = prev.next
            prev.next = next_node

        prev = curr

    return dummy.next

def odd_even_list(head):
    """Group odd-indexed then even-indexed nodes."""
    if not head:
        return None

    odd = head
    even = even_head = head.next

    while even and even.next:
        odd.next = even.next
        odd = odd.next
        even.next = odd.next
        even = even.next

    odd.next = even_head
    return head

# Tests
tests = []

# Reverse
tests.append(("reverse", to_array(reverse_linked_list(to_linked_list([1,2,3,4,5]))), [5,4,3,2,1]))

# Reverse between
tests.append(("reverse_between", to_array(reverse_between(to_linked_list([1,2,3,4,5]), 2, 4)), [1,4,3,2,5]))

# Merge sorted
tests.append(("merge", to_array(merge_two_sorted(to_linked_list([1,2,4]), to_linked_list([1,3,4]))), [1,1,2,3,4,4]))

# Remove nth
tests.append(("remove_nth", to_array(remove_nth_from_end(to_linked_list([1,2,3,4,5]), 2)), [1,2,3,5]))

# Reorder
ll = to_linked_list([1,2,3,4])
reorder_list(ll)
tests.append(("reorder", to_array(ll), [1,4,2,3]))

# Add numbers
tests.append(("add", to_array(add_two_numbers(to_linked_list([2,4,3]), to_linked_list([5,6,4]))), [7,0,8]))

# Partition
tests.append(("partition", to_array(partition_list(to_linked_list([1,4,3,2,5,2]), 3)), [1,2,2,4,3,5]))

# Swap pairs
tests.append(("swap", to_array(swap_pairs(to_linked_list([1,2,3,4]))), [2,1,4,3]))

# Reverse k-group
tests.append(("reverse_k", to_array(reverse_k_group(to_linked_list([1,2,3,4,5]), 2)), [2,1,4,3,5]))

# Odd-even
tests.append(("odd_even", to_array(odd_even_list(to_linked_list([1,2,3,4,5]))), [1,3,5,2,4]))

# Cycle detection - create a list with cycle
cycle_list = to_linked_list([3,2,0,-4])
tail = cycle_list
while tail.next:
    tail = tail.next
node_at_1 = cycle_list.next
tail.next = node_at_1
cycle_node = detect_cycle(cycle_list)
tests.append(("cycle", cycle_node.val if cycle_node else None, 2))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
