class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def build_tree(vals):
    """Build tree from level order list."""
    if not vals:
        return None
    root = TreeNode(vals[0])
    queue = [root]
    i = 1
    while queue and i < len(vals):
        node = queue.pop(0)
        if i < len(vals) and vals[i] is not None:
            node.left = TreeNode(vals[i])
            queue.append(node.left)
        i += 1
        if i < len(vals) and vals[i] is not None:
            node.right = TreeNode(vals[i])
            queue.append(node.right)
        i += 1
    return root

def inorder_traversal(root):
    """Inorder traversal iteratively."""
    result = []
    stack = []
    curr = root

    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        result.append(curr.val)
        curr = curr.right

    return result

def preorder_traversal(root):
    """Preorder traversal iteratively."""
    if not root:
        return []
    result = []
    stack = [root]

    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

    return result

def postorder_traversal(root):
    """Postorder traversal iteratively."""
    if not root:
        return []
    result = []
    stack = [root]

    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)

    return result[::-1]

def level_order(root):
    """Level order traversal."""
    if not root:
        return []
    result = []
    queue = [root]

    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.pop(0)
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)

    return result

def zigzag_level_order(root):
    """Zigzag level order traversal."""
    if not root:
        return []
    result = []
    queue = [root]
    left_to_right = True

    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.pop(0)
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        if not left_to_right:
            level.reverse()
        result.append(level)
        left_to_right = not left_to_right

    return result

def vertical_order(root):
    """Vertical order traversal."""
    if not root:
        return []
    from collections import defaultdict
    col_map = defaultdict(list)
    queue = [(root, 0, 0)]  # node, col, row

    while queue:
        node, col, row = queue.pop(0)
        col_map[col].append((row, node.val))
        if node.left:
            queue.append((node.left, col - 1, row + 1))
        if node.right:
            queue.append((node.right, col + 1, row + 1))

    result = []
    for col in sorted(col_map.keys()):
        result.append([val for _, val in sorted(col_map[col])])

    return result

def build_from_preorder_inorder(preorder, inorder):
    """Build tree from preorder and inorder."""
    if not preorder:
        return None

    root_val = preorder[0]
    root = TreeNode(root_val)
    mid = inorder.index(root_val)

    root.left = build_from_preorder_inorder(preorder[1:mid+1], inorder[:mid])
    root.right = build_from_preorder_inorder(preorder[mid+1:], inorder[mid+1:])

    return root

def build_from_inorder_postorder(inorder, postorder):
    """Build tree from inorder and postorder."""
    if not postorder:
        return None

    root_val = postorder[-1]
    root = TreeNode(root_val)
    mid = inorder.index(root_val)

    root.left = build_from_inorder_postorder(inorder[:mid], postorder[:mid])
    root.right = build_from_inorder_postorder(inorder[mid+1:], postorder[mid:-1])

    return root

def build_bst_from_preorder(preorder):
    """Build BST from preorder traversal."""
    if not preorder:
        return None

    def build(min_val, max_val):
        nonlocal idx
        if idx >= len(preorder):
            return None
        if not (min_val < preorder[idx] < max_val):
            return None

        val = preorder[idx]
        idx += 1
        node = TreeNode(val)
        node.left = build(min_val, val)
        node.right = build(val, max_val)
        return node

    idx = 0
    return build(float('-inf'), float('inf'))

def right_side_view(root):
    """Right side view of binary tree."""
    if not root:
        return []
    result = []
    queue = [root]

    while queue:
        for i in range(len(queue)):
            node = queue.pop(0)
            if i == 0:
                result.append(node.val)
            if node.right:
                queue.append(node.right)
            if node.left:
                queue.append(node.left)

    return result

def boundary_traversal(root):
    """Boundary of binary tree."""
    if not root:
        return []

    def is_leaf(node):
        return not node.left and not node.right

    def left_boundary(node, result):
        while node:
            if not is_leaf(node):
                result.append(node.val)
            node = node.left if node.left else node.right

    def right_boundary(node, result):
        stack = []
        while node:
            if not is_leaf(node):
                stack.append(node.val)
            node = node.right if node.right else node.left
        result.extend(reversed(stack))

    def leaves(node, result):
        if not node:
            return
        if is_leaf(node):
            result.append(node.val)
        leaves(node.left, result)
        leaves(node.right, result)

    result = [root.val]
    if is_leaf(root):
        return result

    left_boundary(root.left, result)
    leaves(root.left, result)
    leaves(root.right, result)
    right_boundary(root.right, result)

    return result

def flatten_to_linked_list(root):
    """Flatten tree to linked list in-place."""
    if not root:
        return None

    stack = [root]
    prev = None

    while stack:
        curr = stack.pop()
        if prev:
            prev.right = curr
            prev.left = None
        if curr.right:
            stack.append(curr.right)
        if curr.left:
            stack.append(curr.left)
        prev = curr

    return root

# Tests
tests = []

# Basic traversals
tree1 = build_tree([1, 2, 3, 4, 5, None, 6])
tests.append(("inorder", inorder_traversal(tree1), [4, 2, 5, 1, 3, 6]))
tests.append(("preorder", preorder_traversal(tree1), [1, 2, 4, 5, 3, 6]))
tests.append(("postorder", postorder_traversal(tree1), [4, 5, 2, 6, 3, 1]))
tests.append(("level_order", level_order(tree1), [[1], [2, 3], [4, 5, 6]]))

# Zigzag
tree2 = build_tree([3, 9, 20, None, None, 15, 7])
tests.append(("zigzag", zigzag_level_order(tree2), [[3], [20, 9], [15, 7]]))

# Vertical order
tree3 = build_tree([1, 2, 3, 4, 5, 6, 7])
tests.append(("vertical", vertical_order(tree3), [[4], [2], [1, 5, 6], [3], [7]]))

# Build from traversals
built = build_from_preorder_inorder([3,9,20,15,7], [9,3,15,20,7])
tests.append(("build_pre_in", level_order(built), [[3], [9, 20], [15, 7]]))

built2 = build_from_inorder_postorder([9,3,15,20,7], [9,15,7,20,3])
tests.append(("build_in_post", level_order(built2), [[3], [9, 20], [15, 7]]))

# BST from preorder
bst = build_bst_from_preorder([8,5,1,7,10,12])
tests.append(("bst_preorder", inorder_traversal(bst), [1, 5, 7, 8, 10, 12]))

# Right side view
tree4 = build_tree([1, 2, 3, None, 5, None, 4])
tests.append(("right_view", right_side_view(tree4), [1, 3, 4]))

# Flatten
tree5 = build_tree([1, 2, 5, 3, 4, None, 6])
flatten_to_linked_list(tree5)
flattened = []
while tree5:
    flattened.append(tree5.val)
    tree5 = tree5.right
tests.append(("flatten", flattened, [1, 2, 3, 4, 5, 6]))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
