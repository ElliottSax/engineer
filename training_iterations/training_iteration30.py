def serialize_binary_tree(root):
    """Serializes binary tree to string."""
    if not root:
        return "null"
    result = []
    queue = [root]
    while queue:
        node = queue.pop(0)
        if node:
            result.append(str(node['val']))
            queue.append(node.get('left'))
            queue.append(node.get('right'))
        else:
            result.append("null")
    # Remove trailing nulls
    while result and result[-1] == "null":
        result.pop()
    return ",".join(result)

def deserialize_binary_tree(data):
    """Deserializes string to binary tree."""
    if data == "null" or not data:
        return None
    values = data.split(",")
    root = {'val': int(values[0]), 'left': None, 'right': None}
    queue = [root]
    i = 1
    while queue and i < len(values):
        node = queue.pop(0)
        if i < len(values) and values[i] != "null":
            node['left'] = {'val': int(values[i]), 'left': None, 'right': None}
            queue.append(node['left'])
        i += 1
        if i < len(values) and values[i] != "null":
            node['right'] = {'val': int(values[i]), 'left': None, 'right': None}
            queue.append(node['right'])
        i += 1
    return root

def validate_bst(root, min_val=float('-inf'), max_val=float('inf')):
    """Validates binary search tree."""
    if not root:
        return True
    if root['val'] <= min_val or root['val'] >= max_val:
        return False
    return (validate_bst(root.get('left'), min_val, root['val']) and
            validate_bst(root.get('right'), root['val'], max_val))

def recover_bst(root):
    """Recovers BST with two swapped nodes."""
    first = second = prev = None

    def inorder(node):
        nonlocal first, second, prev
        if not node:
            return
        inorder(node.get('left'))
        if prev and prev['val'] > node['val']:
            if not first:
                first = prev
            second = node
        prev = node
        inorder(node.get('right'))

    inorder(root)
    if first and second:
        first['val'], second['val'] = second['val'], first['val']
    return root

def trim_bst(root, low, high):
    """Trims BST to values in [low, high]."""
    if not root:
        return None
    if root['val'] < low:
        return trim_bst(root.get('right'), low, high)
    if root['val'] > high:
        return trim_bst(root.get('left'), low, high)
    root['left'] = trim_bst(root.get('left'), low, high)
    root['right'] = trim_bst(root.get('right'), low, high)
    return root

def delete_node_bst(root, key):
    """Deletes node from BST."""
    if not root:
        return None
    if key < root['val']:
        root['left'] = delete_node_bst(root.get('left'), key)
    elif key > root['val']:
        root['right'] = delete_node_bst(root.get('right'), key)
    else:
        if not root.get('left'):
            return root.get('right')
        if not root.get('right'):
            return root.get('left')
        # Find successor
        successor = root['right']
        while successor.get('left'):
            successor = successor['left']
        root['val'] = successor['val']
        root['right'] = delete_node_bst(root['right'], successor['val'])
    return root

def insert_bst(root, val):
    """Inserts value into BST."""
    if not root:
        return {'val': val, 'left': None, 'right': None}
    if val < root['val']:
        root['left'] = insert_bst(root.get('left'), val)
    else:
        root['right'] = insert_bst(root.get('right'), val)
    return root

def sorted_array_to_bst(nums):
    """Converts sorted array to height-balanced BST."""
    if not nums:
        return None
    mid = len(nums) // 2
    return {
        'val': nums[mid],
        'left': sorted_array_to_bst(nums[:mid]),
        'right': sorted_array_to_bst(nums[mid+1:])
    }

def bst_iterator():
    """BST iterator using controlled inorder traversal."""
    stack = []

    def init(root):
        push_left(root)

    def push_left(node):
        while node:
            stack.append(node)
            node = node.get('left')

    def has_next():
        return len(stack) > 0

    def get_next():
        node = stack.pop()
        push_left(node.get('right'))
        return node['val']

    return init, has_next, get_next

def inorder_successor_bst(root, p_val):
    """Finds inorder successor in BST."""
    successor = None
    while root:
        if p_val < root['val']:
            successor = root
            root = root.get('left')
        else:
            root = root.get('right')
    return successor

def convert_bst_to_greater(root):
    """Converts BST to Greater Sum Tree."""
    total = [0]

    def reverse_inorder(node):
        if not node:
            return
        reverse_inorder(node.get('right'))
        total[0] += node['val']
        node['val'] = total[0]
        reverse_inorder(node.get('left'))

    reverse_inorder(root)
    return root

# Helper
def tree_vals(root):
    """Gets values in level order."""
    if not root:
        return []
    result = []
    queue = [root]
    while queue:
        node = queue.pop(0)
        result.append(node['val'])
        if node.get('left'):
            queue.append(node['left'])
        if node.get('right'):
            queue.append(node['right'])
    return result

# Tests
# Build test BST
bst = {'val': 5, 'left': {'val': 3, 'left': {'val': 2, 'left': None, 'right': None}, 'right': {'val': 4, 'left': None, 'right': None}}, 'right': {'val': 7, 'left': {'val': 6, 'left': None, 'right': None}, 'right': {'val': 8, 'left': None, 'right': None}}}

tests = [
    ("validate_bst", validate_bst(bst), True),
    ("validate_bst_no", validate_bst({'val': 5, 'left': {'val': 6, 'left': None, 'right': None}, 'right': None}), False),
]

# Serialize/deserialize test
tree = {'val': 1, 'left': {'val': 2, 'left': None, 'right': None}, 'right': {'val': 3, 'left': {'val': 4, 'left': None, 'right': None}, 'right': {'val': 5, 'left': None, 'right': None}}}
serialized = serialize_binary_tree(tree)
deserialized = deserialize_binary_tree(serialized)
tests.append(("serialize", deserialized['val'] == 1 and deserialized['right']['val'] == 3, True))

# Trim BST test
trim_tree = {'val': 3, 'left': {'val': 0, 'left': None, 'right': {'val': 2, 'left': {'val': 1, 'left': None, 'right': None}, 'right': None}}, 'right': {'val': 4, 'left': None, 'right': None}}
trimmed = trim_bst(trim_tree, 1, 3)
tests.append(("trim", trimmed['val'], 3))
tests.append(("trim_left", trimmed['left']['val'], 2))

# Delete node test
delete_tree = {'val': 5, 'left': {'val': 3, 'left': {'val': 2, 'left': None, 'right': None}, 'right': {'val': 4, 'left': None, 'right': None}}, 'right': {'val': 6, 'left': None, 'right': {'val': 7, 'left': None, 'right': None}}}
deleted = delete_node_bst(delete_tree, 3)
tests.append(("delete", 3 not in tree_vals(deleted), True))

# Insert test
insert_tree = {'val': 4, 'left': {'val': 2, 'left': {'val': 1, 'left': None, 'right': None}, 'right': {'val': 3, 'left': None, 'right': None}}, 'right': {'val': 7, 'left': None, 'right': None}}
inserted = insert_bst(insert_tree, 5)
tests.append(("insert", 5 in tree_vals(inserted), True))

# Sorted array to BST
arr_tree = sorted_array_to_bst([-10, -3, 0, 5, 9])
tests.append(("arr_to_bst", arr_tree['val'], 0))

# BST Iterator
init_iter, has_next, get_next = bst_iterator()
iter_tree = {'val': 7, 'left': {'val': 3, 'left': None, 'right': None}, 'right': {'val': 15, 'left': {'val': 9, 'left': None, 'right': None}, 'right': {'val': 20, 'left': None, 'right': None}}}
init_iter(iter_tree)
tests.append(("iter_next", get_next(), 3))
tests.append(("iter_next_2", get_next(), 7))
tests.append(("iter_has", has_next(), True))

# Inorder successor
succ_tree = {'val': 5, 'left': {'val': 3, 'left': {'val': 2, 'left': None, 'right': None}, 'right': {'val': 4, 'left': None, 'right': None}}, 'right': {'val': 6, 'left': None, 'right': None}}
tests.append(("successor", inorder_successor_bst(succ_tree, 4)['val'], 5))

# Greater sum tree
gst = {'val': 4, 'left': {'val': 1, 'left': {'val': 0, 'left': None, 'right': None}, 'right': {'val': 2, 'left': None, 'right': {'val': 3, 'left': None, 'right': None}}}, 'right': {'val': 6, 'left': {'val': 5, 'left': None, 'right': None}, 'right': {'val': 7, 'left': None, 'right': {'val': 8, 'left': None, 'right': None}}}}
convert_bst_to_greater(gst)
tests.append(("greater_sum", gst['val'], 30))

# Recover BST
bad_bst = {'val': 3, 'left': {'val': 1, 'left': None, 'right': None}, 'right': {'val': 4, 'left': {'val': 2, 'left': None, 'right': None}, 'right': None}}
recover_bst(bad_bst)
tests.append(("recover", validate_bst(bad_bst), True))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
