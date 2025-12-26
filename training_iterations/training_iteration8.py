def word_ladder(begin_word, end_word, word_list):
    """Minimum transformations from begin to end (change 1 letter at a time)."""
    from collections import deque
    word_set = set(word_list)
    if end_word not in word_set:
        return 0
    queue = deque([(begin_word, 1)])
    visited = {begin_word}
    while queue:
        word, level = queue.popleft()
        if word == end_word:
            return level
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                new_word = word[:i] + c + word[i+1:]
                if new_word in word_set and new_word not in visited:
                    visited.add(new_word)
                    queue.append((new_word, level + 1))
    return 0

def course_schedule(num_courses, prerequisites):
    """Can finish all courses? (detect cycle in prerequisite graph)."""
    graph = {i: [] for i in range(num_courses)}
    for course, prereq in prerequisites:
        graph[prereq].append(course)
    visited = [0] * num_courses  # 0=unvisited, 1=visiting, 2=visited
    def dfs(course):
        if visited[course] == 1:
            return False  # Cycle
        if visited[course] == 2:
            return True
        visited[course] = 1
        for neighbor in graph[course]:
            if not dfs(neighbor):
                return False
        visited[course] = 2
        return True
    return all(dfs(i) for i in range(num_courses))

def find_peak_element(nums):
    """Finds index of any peak element using binary search."""
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[mid + 1]:
            right = mid
        else:
            left = mid + 1
    return left

def kth_smallest_bst(root, k):
    """Finds kth smallest element in BST using inorder traversal."""
    stack = []
    current = root
    count = 0
    while stack or current:
        while current:
            stack.append(current)
            current = current.get('left')
        current = stack.pop()
        count += 1
        if count == k:
            return current['val']
        current = current.get('right')
    return -1

def lowest_common_ancestor(root, p, q):
    """Finds LCA of two nodes in binary tree."""
    if root is None or root['val'] == p or root['val'] == q:
        return root
    left = lowest_common_ancestor(root.get('left'), p, q)
    right = lowest_common_ancestor(root.get('right'), p, q)
    if left and right:
        return root
    return left if left else right

def max_path_sum_tree(root):
    """Maximum path sum in binary tree."""
    max_sum = [float('-inf')]
    def helper(node):
        if not node:
            return 0
        left = max(0, helper(node.get('left')))
        right = max(0, helper(node.get('right')))
        max_sum[0] = max(max_sum[0], left + right + node['val'])
        return max(left, right) + node['val']
    helper(root)
    return max_sum[0]

def reconstruct_itinerary(tickets):
    """Reconstructs itinerary from tickets starting at JFK."""
    from collections import defaultdict
    graph = defaultdict(list)
    for src, dst in sorted(tickets, reverse=True):
        graph[src].append(dst)
    route = []
    def dfs(airport):
        while graph[airport]:
            dfs(graph[airport].pop())
        route.append(airport)
    dfs("JFK")
    return route[::-1]

def sliding_window_max(nums, k):
    """Maximum in each sliding window of size k."""
    from collections import deque
    dq = deque()
    result = []
    for i, num in enumerate(nums):
        while dq and nums[dq[-1]] < num:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result

def alien_dictionary(words):
    """Derives alien alphabet order from sorted words."""
    from collections import defaultdict, deque
    graph = defaultdict(set)
    in_degree = {c: 0 for word in words for c in word}
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        min_len = min(len(w1), len(w2))
        if len(w1) > len(w2) and w1[:min_len] == w2[:min_len]:
            return ""  # Invalid
        for j in range(min_len):
            if w1[j] != w2[j]:
                if w2[j] not in graph[w1[j]]:
                    graph[w1[j]].add(w2[j])
                    in_degree[w2[j]] += 1
                break
    queue = deque([c for c in in_degree if in_degree[c] == 0])
    result = []
    while queue:
        c = queue.popleft()
        result.append(c)
        for neighbor in graph[c]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return "".join(result) if len(result) == len(in_degree) else ""

# Tests
tests = [
    ("word_ladder", word_ladder("hit", "cog", ["hot","dot","dog","lot","log","cog"]), 5),
    ("course_yes", course_schedule(2, [[1,0]]), True),
    ("course_no", course_schedule(2, [[1,0],[0,1]]), False),
    ("peak_elem", find_peak_element([1,2,1,3,5,6,4]) in [1, 5], True),
    ("sliding_max", sliding_window_max([1,3,-1,-3,5,3,6,7], 3), [3,3,5,5,6,7]),
    ("itinerary", reconstruct_itinerary([["MU","LHR"],["JFK","MU"],["LHR","SFO"],["SFO","JFK"]]), ["JFK","MU","LHR","SFO","JFK"]),
    ("alien_dict", alien_dictionary(["wrt","wrf","er","ett","rftt"]), "wertf"),
]

# BST test
bst = {'val': 3, 'left': {'val': 1, 'left': None, 'right': {'val': 2, 'left': None, 'right': None}}, 'right': {'val': 4, 'left': None, 'right': None}}
tests.append(("kth_smallest", kth_smallest_bst(bst, 2), 2))

# Max path sum test
tree = {'val': -10, 'left': {'val': 9, 'left': None, 'right': None}, 'right': {'val': 20, 'left': {'val': 15, 'left': None, 'right': None}, 'right': {'val': 7, 'left': None, 'right': None}}}
tests.append(("max_path_sum", max_path_sum_tree(tree), 42))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
