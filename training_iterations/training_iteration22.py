def invert_binary_tree(root):
    """Inverts binary tree (mirror image)."""
    if not root:
        return None
    root['left'], root['right'] = invert_binary_tree(root.get('right')), invert_binary_tree(root.get('left'))
    return root

def symmetric_tree(root):
    """Checks if tree is symmetric."""
    def is_mirror(t1, t2):
        if not t1 and not t2:
            return True
        if not t1 or not t2:
            return False
        return (t1['val'] == t2['val'] and
                is_mirror(t1.get('left'), t2.get('right')) and
                is_mirror(t1.get('right'), t2.get('left')))
    return is_mirror(root, root)

def diameter_binary_tree(root):
    """Diameter of binary tree (longest path between any two nodes)."""
    diameter = [0]

    def depth(node):
        if not node:
            return 0
        left = depth(node.get('left'))
        right = depth(node.get('right'))
        diameter[0] = max(diameter[0], left + right)
        return max(left, right) + 1

    depth(root)
    return diameter[0]

def balanced_binary_tree(root):
    """Checks if tree is height-balanced."""
    def check(node):
        if not node:
            return 0
        left = check(node.get('left'))
        right = check(node.get('right'))
        if left == -1 or right == -1 or abs(left - right) > 1:
            return -1
        return max(left, right) + 1

    return check(root) != -1

def flatten_binary_tree(root):
    """Flattens binary tree to linked list (preorder)."""
    if not root:
        return None
    stack = [root]
    prev = None
    while stack:
        node = stack.pop()
        if prev:
            prev['left'] = None
            prev['right'] = node
        if node.get('right'):
            stack.append(node['right'])
        if node.get('left'):
            stack.append(node['left'])
        prev = node
    return root

def path_sum(root, target):
    """Checks if root-to-leaf path sums to target."""
    if not root:
        return False
    if not root.get('left') and not root.get('right'):
        return root['val'] == target
    target -= root['val']
    return path_sum(root.get('left'), target) or path_sum(root.get('right'), target)

def path_sum_ii(root, target):
    """All root-to-leaf paths that sum to target."""
    result = []

    def dfs(node, remaining, path):
        if not node:
            return
        path.append(node['val'])
        if not node.get('left') and not node.get('right') and remaining == node['val']:
            result.append(path[:])
        dfs(node.get('left'), remaining - node['val'], path)
        dfs(node.get('right'), remaining - node['val'], path)
        path.pop()

    dfs(root, target, [])
    return result

def construct_from_preorder_inorder(preorder, inorder):
    """Constructs binary tree from preorder and inorder traversal."""
    if not preorder or not inorder:
        return None
    root_val = preorder[0]
    root = {'val': root_val, 'left': None, 'right': None}
    mid = inorder.index(root_val)
    root['left'] = construct_from_preorder_inorder(preorder[1:mid+1], inorder[:mid])
    root['right'] = construct_from_preorder_inorder(preorder[mid+1:], inorder[mid+1:])
    return root

def binary_tree_right_side_view(root):
    """Values visible from right side."""
    if not root:
        return []
    from collections import deque
    result = []
    queue = deque([root])
    while queue:
        level_size = len(queue)
        for i in range(level_size):
            node = queue.popleft()
            if i == level_size - 1:
                result.append(node['val'])
            if node.get('left'):
                queue.append(node['left'])
            if node.get('right'):
                queue.append(node['right'])
    return result

def level_order_traversal(root):
    """Level order traversal of binary tree."""
    if not root:
        return []
    from collections import deque
    result = []
    queue = deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node['val'])
            if node.get('left'):
                queue.append(node['left'])
            if node.get('right'):
                queue.append(node['right'])
        result.append(level)
    return result

def zigzag_level_order(root):
    """Zigzag level order traversal."""
    if not root:
        return []
    from collections import deque
    result = []
    queue = deque([root])
    left_to_right = True
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node['val'])
            if node.get('left'):
                queue.append(node['left'])
            if node.get('right'):
                queue.append(node['right'])
        if not left_to_right:
            level.reverse()
        result.append(level)
        left_to_right = not left_to_right
    return result

def count_complete_tree_nodes(root):
    """Counts nodes in complete binary tree in O(log^2 n)."""
    if not root:
        return 0

    def get_depth(node, go_left):
        depth = 0
        while node:
            depth += 1
            node = node.get('left') if go_left else node.get('right')
        return depth

    left_depth = get_depth(root, True)
    right_depth = get_depth(root, False)

    if left_depth == right_depth:
        return (1 << left_depth) - 1
    return 1 + count_complete_tree_nodes(root.get('left')) + count_complete_tree_nodes(root.get('right'))

# Tests
# Build test trees
tree1 = {'val': 4, 'left': {'val': 2, 'left': {'val': 1, 'left': None, 'right': None}, 'right': {'val': 3, 'left': None, 'right': None}}, 'right': {'val': 7, 'left': {'val': 6, 'left': None, 'right': None}, 'right': {'val': 9, 'left': None, 'right': None}}}

symmetric = {'val': 1, 'left': {'val': 2, 'left': {'val': 3, 'left': None, 'right': None}, 'right': {'val': 4, 'left': None, 'right': None}}, 'right': {'val': 2, 'left': {'val': 4, 'left': None, 'right': None}, 'right': {'val': 3, 'left': None, 'right': None}}}

path_tree = {'val': 5, 'left': {'val': 4, 'left': {'val': 11, 'left': {'val': 7, 'left': None, 'right': None}, 'right': {'val': 2, 'left': None, 'right': None}}, 'right': None}, 'right': {'val': 8, 'left': {'val': 13, 'left': None, 'right': None}, 'right': {'val': 4, 'left': None, 'right': {'val': 1, 'left': None, 'right': None}}}}

# Invert tree and check
inverted = invert_binary_tree({'val': 4, 'left': {'val': 2, 'left': {'val': 1, 'left': None, 'right': None}, 'right': {'val': 3, 'left': None, 'right': None}}, 'right': {'val': 7, 'left': {'val': 6, 'left': None, 'right': None}, 'right': {'val': 9, 'left': None, 'right': None}}})

tests = [
    ("invert", inverted['left']['val'], 7),
    ("symmetric", symmetric_tree(symmetric), True),
    ("diameter", diameter_binary_tree(tree1), 4),
    ("balanced", balanced_binary_tree(tree1), True),
    ("path_sum", path_sum(path_tree, 22), True),
    ("path_sum_no", path_sum(path_tree, 5), False),
    ("path_sum_ii", path_sum_ii(path_tree, 22), [[5,4,11,2]]),
    ("right_side", binary_tree_right_side_view(tree1), [4, 7, 9]),
    ("level_order", level_order_traversal(tree1), [[4], [2, 7], [1, 3, 6, 9]]),
    ("zigzag", zigzag_level_order(tree1), [[4], [7, 2], [1, 3, 6, 9]]),
    ("count_complete", count_complete_tree_nodes({'val': 1, 'left': {'val': 2, 'left': {'val': 4, 'left': None, 'right': None}, 'right': {'val': 5, 'left': None, 'right': None}}, 'right': {'val': 3, 'left': {'val': 6, 'left': None, 'right': None}, 'right': None}}), 6),
]

# Construct from traversal test
preorder = [3,9,20,15,7]
inorder_arr = [9,3,15,20,7]
constructed = construct_from_preorder_inorder(preorder, inorder_arr)
tests.append(("construct", constructed['val'] == 3 and constructed['left']['val'] == 9, True))

# Flatten test
flat_tree = {'val': 1, 'left': {'val': 2, 'left': {'val': 3, 'left': None, 'right': None}, 'right': {'val': 4, 'left': None, 'right': None}}, 'right': {'val': 5, 'left': None, 'right': {'val': 6, 'left': None, 'right': None}}}
flatten_binary_tree(flat_tree)
vals = []
node = flat_tree
while node:
    vals.append(node['val'])
    node = node.get('right')
tests.append(("flatten", vals, [1,2,3,4,5,6]))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
