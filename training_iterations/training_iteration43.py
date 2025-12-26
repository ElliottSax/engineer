def is_bipartite(graph):
    """Checks if undirected graph is bipartite."""
    n = len(graph)
    color = [-1] * n

    def bfs(start):
        from collections import deque
        queue = deque([start])
        color[start] = 0
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if color[neighbor] == -1:
                    color[neighbor] = 1 - color[node]
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    return False
        return True

    for i in range(n):
        if color[i] == -1 and not bfs(i):
            return False
    return True

def flower_planting(n, paths):
    """Assign 4 flower types so no adjacent gardens have same type."""
    from collections import defaultdict
    graph = defaultdict(list)
    for a, b in paths:
        graph[a].append(b)
        graph[b].append(a)

    result = [0] * n
    for i in range(1, n + 1):
        used = {result[j - 1] for j in graph[i]}
        for flower in range(1, 5):
            if flower not in used:
                result[i - 1] = flower
                break
    return result

def minimum_height_trees(n, edges):
    """Find root nodes that minimize tree height."""
    if n <= 2:
        return list(range(n))
    from collections import defaultdict
    graph = defaultdict(set)
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)

    leaves = [i for i in range(n) if len(graph[i]) == 1]
    remaining = n

    while remaining > 2:
        remaining -= len(leaves)
        new_leaves = []
        for leaf in leaves:
            neighbor = graph[leaf].pop()
            graph[neighbor].remove(leaf)
            if len(graph[neighbor]) == 1:
                new_leaves.append(neighbor)
        leaves = new_leaves

    return leaves

def find_eventual_safe_nodes(graph):
    """Find all nodes that lead to terminal nodes."""
    n = len(graph)
    color = [0] * n  # 0: white, 1: gray, 2: black

    def dfs(node):
        if color[node] > 0:
            return color[node] == 2
        color[node] = 1
        for neighbor in graph[node]:
            if not dfs(neighbor):
                return False
        color[node] = 2
        return True

    return [i for i in range(n) if dfs(i)]

def keys_and_rooms(rooms):
    """Can we visit all rooms starting from room 0?"""
    visited = {0}
    stack = [0]
    while stack:
        room = stack.pop()
        for key in rooms[room]:
            if key not in visited:
                visited.add(key)
                stack.append(key)
    return len(visited) == len(rooms)

def find_town_judge(n, trust):
    """Find town judge (trusted by all, trusts no one)."""
    if n == 1:
        return 1
    trust_count = [0] * (n + 1)
    for a, b in trust:
        trust_count[a] -= 1
        trust_count[b] += 1
    for i in range(1, n + 1):
        if trust_count[i] == n - 1:
            return i
    return -1

def find_center_star(edges):
    """Find center of star graph."""
    if edges[0][0] in edges[1]:
        return edges[0][0]
    return edges[0][1]

def minimum_vertices_reach_all(n, edges):
    """Minimum vertices to reach all nodes in DAG."""
    has_incoming = [False] * n
    for _, to in edges:
        has_incoming[to] = True
    return [i for i in range(n) if not has_incoming[i]]

def find_redundant_directed_connection(edges):
    """Find edge causing invalid tree in directed graph."""
    n = len(edges)
    parent = [0] * (n + 1)
    candidate1 = candidate2 = None

    # Check for node with two parents
    for u, v in edges:
        if parent[v] > 0:
            candidate1 = [parent[v], v]
            candidate2 = [u, v]
        else:
            parent[v] = u

    # Union-Find to detect cycle
    root = list(range(n + 1))

    def find(x):
        if root[x] != x:
            root[x] = find(root[x])
        return root[x]

    for u, v in edges:
        if [u, v] == candidate2:
            continue
        pu, pv = find(u), find(v)
        if pu == pv:
            return candidate1 if candidate1 else [u, v]
        root[pu] = pv

    return candidate2

def max_network_rank(n, roads):
    """Maximum sum of degrees of two connected cities."""
    degree = [0] * n
    connected = set()
    for a, b in roads:
        degree[a] += 1
        degree[b] += 1
        connected.add((min(a, b), max(a, b)))

    max_rank = 0
    for i in range(n):
        for j in range(i + 1, n):
            rank = degree[i] + degree[j]
            if (i, j) in connected:
                rank -= 1
            max_rank = max(max_rank, rank)
    return max_rank

# Tests
tests = [
    ("bipartite", is_bipartite([[1,3],[0,2],[1,3],[0,2]]), True),
    ("bipartite_no", is_bipartite([[1,2,3],[0,2],[0,1,3],[0,2]]), False),
    ("flowers", flower_planting(3, [[1,2],[2,3],[3,1]]), [1, 2, 3]),
    ("mht", sorted(minimum_height_trees(4, [[1,0],[1,2],[1,3]])), [1]),
    ("mht_2", sorted(minimum_height_trees(6, [[3,0],[3,1],[3,2],[3,4],[5,4]])), [3, 4]),
    ("safe_nodes", find_eventual_safe_nodes([[1,2],[2,3],[5],[0],[5],[],[]]), [2, 4, 5, 6]),
    ("keys_rooms", keys_and_rooms([[1],[2],[3],[]]), True),
    ("keys_rooms_no", keys_and_rooms([[1,3],[3,0,1],[2],[0]]), False),
    ("judge", find_town_judge(3, [[1,3],[2,3]]), 3),
    ("judge_no", find_town_judge(3, [[1,3],[2,3],[3,1]]), -1),
    ("star_center", find_center_star([[1,2],[2,3],[4,2]]), 2),
    ("min_vertices", sorted(minimum_vertices_reach_all(6, [[0,1],[0,2],[2,5],[3,4],[4,2]])), [0, 3]),
    ("redundant_dir", find_redundant_directed_connection([[1,2],[1,3],[2,3]]), [2, 3]),
    ("network_rank", max_network_rank(4, [[0,1],[0,3],[1,2],[1,3]]), 4),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
