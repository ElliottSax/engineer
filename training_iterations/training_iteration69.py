class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def build_tree(vals):
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

def max_path_sum(root):
    """Maximum path sum in binary tree."""
    max_sum = [float('-inf')]

    def dfs(node):
        if not node:
            return 0
        left = max(0, dfs(node.left))
        right = max(0, dfs(node.right))
        max_sum[0] = max(max_sum[0], left + right + node.val)
        return max(left, right) + node.val

    dfs(root)
    return max_sum[0]

def diameter_of_tree(root):
    """Diameter of binary tree (longest path)."""
    diameter = [0]

    def dfs(node):
        if not node:
            return 0
        left = dfs(node.left)
        right = dfs(node.right)
        diameter[0] = max(diameter[0], left + right)
        return max(left, right) + 1

    dfs(root)
    return diameter[0]

def lowest_common_ancestor(root, p, q):
    """LCA of two nodes."""
    if not root or root.val == p or root.val == q:
        return root.val if root else None

    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)

    if left and right:
        return root.val
    return left if left else right

def binary_tree_cameras(root):
    """Minimum cameras to monitor all nodes."""
    result = [0]
    NOT_COVERED = 0
    HAS_CAMERA = 1
    COVERED = 2

    def dfs(node):
        if not node:
            return COVERED

        left = dfs(node.left)
        right = dfs(node.right)

        if left == NOT_COVERED or right == NOT_COVERED:
            result[0] += 1
            return HAS_CAMERA

        if left == HAS_CAMERA or right == HAS_CAMERA:
            return COVERED

        return NOT_COVERED

    if dfs(root) == NOT_COVERED:
        result[0] += 1

    return result[0]

def count_good_nodes(root):
    """Count nodes where path from root has no greater value."""
    def dfs(node, max_val):
        if not node:
            return 0
        count = 1 if node.val >= max_val else 0
        max_val = max(max_val, node.val)
        return count + dfs(node.left, max_val) + dfs(node.right, max_val)

    return dfs(root, float('-inf'))

def sum_root_to_leaf_numbers(root):
    """Sum of all root-to-leaf numbers."""
    def dfs(node, current):
        if not node:
            return 0
        current = current * 10 + node.val
        if not node.left and not node.right:
            return current
        return dfs(node.left, current) + dfs(node.right, current)

    return dfs(root, 0)

def path_sum_iii(root, target_sum):
    """Count paths that sum to target (can start anywhere)."""
    from collections import defaultdict
    count = [0]
    prefix_sums = defaultdict(int)
    prefix_sums[0] = 1

    def dfs(node, current_sum):
        if not node:
            return

        current_sum += node.val
        count[0] += prefix_sums[current_sum - target_sum]

        prefix_sums[current_sum] += 1
        dfs(node.left, current_sum)
        dfs(node.right, current_sum)
        prefix_sums[current_sum] -= 1

    dfs(root, 0)
    return count[0]

def house_robber_iii(root):
    """Maximum robbery without robbing adjacent nodes."""
    def dfs(node):
        if not node:
            return (0, 0)  # (with_node, without_node)

        left = dfs(node.left)
        right = dfs(node.right)

        with_node = node.val + left[1] + right[1]
        without_node = max(left) + max(right)

        return (with_node, without_node)

    return max(dfs(root))

def distribute_coins(root):
    """Minimum moves to distribute coins."""
    moves = [0]

    def dfs(node):
        if not node:
            return 0
        left = dfs(node.left)
        right = dfs(node.right)
        moves[0] += abs(left) + abs(right)
        return node.val + left + right - 1

    dfs(root)
    return moves[0]

def longest_zigzag_path(root):
    """Longest zigzag path in tree."""
    max_len = [0]

    def dfs(node, direction, length):
        if not node:
            return
        max_len[0] = max(max_len[0], length)
        if direction == 'left':
            dfs(node.left, 'right', length + 1)
            dfs(node.right, 'left', 1)
        else:
            dfs(node.right, 'left', length + 1)
            dfs(node.left, 'right', 1)

    if root:
        dfs(root.left, 'right', 1)
        dfs(root.right, 'left', 1)
    return max_len[0]

def all_possible_fbt(n):
    """All structurally unique full binary trees with n nodes."""
    if n % 2 == 0:
        return []
    if n == 1:
        return [TreeNode(0)]

    result = []
    for left_count in range(1, n, 2):
        right_count = n - 1 - left_count
        for left in all_possible_fbt(left_count):
            for right in all_possible_fbt(right_count):
                root = TreeNode(0)
                root.left = left
                root.right = right
                result.append(root)

    return result

# Tests
tests = []

# Max path sum
tree1 = build_tree([-10, 9, 20, None, None, 15, 7])
tests.append(("max_path", max_path_sum(tree1), 42))

# Diameter
tree2 = build_tree([1, 2, 3, 4, 5])
tests.append(("diameter", diameter_of_tree(tree2), 3))

# LCA
tree3 = build_tree([3, 5, 1, 6, 2, 0, 8, None, None, 7, 4])
tests.append(("lca", lowest_common_ancestor(tree3, 5, 1), 3))

# Good nodes
tree4 = build_tree([3, 1, 4, 3, None, 1, 5])
tests.append(("good_nodes", count_good_nodes(tree4), 4))

# Sum root to leaf
tree5 = build_tree([1, 2, 3])
tests.append(("sum_root_leaf", sum_root_to_leaf_numbers(tree5), 25))

# Path sum III
tree6 = build_tree([10, 5, -3, 3, 2, None, 11, 3, -2, None, 1])
tests.append(("path_sum_3", path_sum_iii(tree6, 8), 3))

# House robber III
tree7 = build_tree([3, 2, 3, None, 3, None, 1])
tests.append(("robber_3", house_robber_iii(tree7), 7))

# Distribute coins
tree8 = build_tree([3, 0, 0])
tests.append(("coins", distribute_coins(tree8), 2))

# Zigzag
tree9 = build_tree([1, 1, 1, None, 1, None, None, 1, 1, None, 1])
tests.append(("zigzag", longest_zigzag_path(tree9), 4))

# All possible FBT
tests.append(("fbt", len(all_possible_fbt(7)), 5))

# Cameras
tree10 = build_tree([0, 0, None, 0, 0])
tests.append(("cameras", binary_tree_cameras(tree10), 1))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
