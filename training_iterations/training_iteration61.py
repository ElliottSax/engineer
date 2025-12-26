from collections import defaultdict, deque

def course_schedule(numCourses, prerequisites):
    """Check if can finish all courses."""
    graph = defaultdict(list)
    in_degree = [0] * numCourses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    count = 0

    while queue:
        node = queue.popleft()
        count += 1
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return count == numCourses

def course_schedule_order(numCourses, prerequisites):
    """Return valid course order."""
    graph = defaultdict(list)
    in_degree = [0] * numCourses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return order if len(order) == numCourses else []

def alien_dictionary(words):
    """Derive alphabet order from sorted alien words."""
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

    return ''.join(result) if len(result) == len(in_degree) else ""

def minimum_height_trees(n, edges):
    """Find roots of minimum height trees."""
    if n == 1:
        return [0]

    graph = defaultdict(set)
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)

    leaves = [i for i in range(n) if len(graph[i]) == 1]

    while n > 2:
        n -= len(leaves)
        new_leaves = []
        for leaf in leaves:
            neighbor = graph[leaf].pop()
            graph[neighbor].remove(leaf)
            if len(graph[neighbor]) == 1:
                new_leaves.append(neighbor)
        leaves = new_leaves

    return leaves

def parallel_courses(n, relations):
    """Minimum semesters to complete all courses."""
    graph = defaultdict(list)
    in_degree = [0] * (n + 1)

    for prereq, course in relations:
        graph[prereq].append(course)
        in_degree[course] += 1

    queue = deque([(i, 1) for i in range(1, n + 1) if in_degree[i] == 0])
    completed = 0
    semesters = 0

    while queue:
        node, semester = queue.popleft()
        completed += 1
        semesters = max(semesters, semester)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append((neighbor, semester + 1))

    return semesters if completed == n else -1

def longest_path_dag(n, edges):
    """Longest path in directed acyclic graph."""
    graph = defaultdict(list)
    in_degree = [0] * n

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    dist = [0] * n
    queue = deque([i for i in range(n) if in_degree[i] == 0])

    while queue:
        u = queue.popleft()
        for v in graph[u]:
            dist[v] = max(dist[v], dist[u] + 1)
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    return max(dist)

def all_ancestors(n, edges):
    """Find all ancestors for each node."""
    graph = defaultdict(list)
    in_degree = [0] * n

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    ancestors = [set() for _ in range(n)]
    queue = deque([i for i in range(n) if in_degree[i] == 0])

    while queue:
        u = queue.popleft()
        for v in graph[u]:
            ancestors[v].add(u)
            ancestors[v].update(ancestors[u])
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    return [sorted(list(a)) for a in ancestors]

def is_dag(n, edges):
    """Check if directed graph is acyclic."""
    graph = defaultdict(list)
    in_degree = [0] * n

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    queue = deque([i for i in range(n) if in_degree[i] == 0])
    count = 0

    while queue:
        node = queue.popleft()
        count += 1
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return count == n

def eventual_safe_nodes(graph):
    """Find nodes that lead to terminal nodes."""
    n = len(graph)
    reverse_graph = defaultdict(list)
    out_degree = [0] * n

    for u in range(n):
        out_degree[u] = len(graph[u])
        for v in graph[u]:
            reverse_graph[v].append(u)

    queue = deque([i for i in range(n) if out_degree[i] == 0])
    safe = [False] * n

    while queue:
        node = queue.popleft()
        safe[node] = True
        for neighbor in reverse_graph[node]:
            out_degree[neighbor] -= 1
            if out_degree[neighbor] == 0:
                queue.append(neighbor)

    return [i for i in range(n) if safe[i]]

def sort_items_by_groups(n, m, group, beforeItems):
    """Sort items respecting group and item dependencies."""
    # Assign ungrouped items to new groups
    next_group = m
    for i in range(n):
        if group[i] == -1:
            group[i] = next_group
            next_group += 1

    # Build item graph and group graph
    item_graph = defaultdict(list)
    item_in_degree = [0] * n
    group_graph = defaultdict(set)
    group_in_degree = defaultdict(int)

    for i in range(n):
        for before in beforeItems[i]:
            item_graph[before].append(i)
            item_in_degree[i] += 1
            if group[before] != group[i]:
                if i not in group_graph[group[before]]:
                    group_graph[group[before]].add(group[i])

    # Build in-degrees for groups
    all_groups = set(group)
    for g in all_groups:
        group_in_degree[g] = 0
    for g in group_graph:
        for ng in group_graph[g]:
            group_in_degree[ng] += 1

    # Topological sort groups
    group_order = []
    queue = deque([g for g in all_groups if group_in_degree[g] == 0])
    while queue:
        g = queue.popleft()
        group_order.append(g)
        for ng in group_graph[g]:
            group_in_degree[ng] -= 1
            if group_in_degree[ng] == 0:
                queue.append(ng)

    if len(group_order) != len(all_groups):
        return []

    # Items per group
    items_in_group = defaultdict(list)
    for i in range(n):
        items_in_group[group[i]].append(i)

    # Topological sort within each group
    def topo_sort_items(items):
        local_in = {i: 0 for i in items}
        for i in items:
            for j in item_graph[i]:
                if j in local_in:
                    local_in[j] += 1
        queue = deque([i for i in items if local_in[i] == 0])
        order = []
        while queue:
            i = queue.popleft()
            order.append(i)
            for j in item_graph[i]:
                if j in local_in:
                    local_in[j] -= 1
                    if local_in[j] == 0:
                        queue.append(j)
        return order if len(order) == len(items) else None

    result = []
    for g in group_order:
        sorted_items = topo_sort_items(items_in_group[g])
        if sorted_items is None:
            return []
        result.extend(sorted_items)

    return result

# Tests
tests = [
    ("can_finish", course_schedule(2, [[1,0]]), True),
    ("cant_finish", course_schedule(2, [[1,0],[0,1]]), False),
    ("order", course_schedule_order(4, [[1,0],[2,0],[3,1],[3,2]]), [0, 1, 2, 3]),
    ("alien", alien_dictionary(["wrt","wrf","er","ett","rftt"]), "wertf"),
    ("mht", sorted(minimum_height_trees(6, [[3,0],[3,1],[3,2],[3,4],[5,4]])), [3, 4]),
    ("parallel", parallel_courses(3, [[1,3],[2,3]]), 2),
    ("longest", longest_path_dag(4, [(0,1),(0,2),(1,3),(2,3)]), 2),
    ("ancestors", all_ancestors(5, [(0,1),(0,2),(1,3),(2,3),(3,4)]), [[],[0],[0],[0,1,2],[0,1,2,3]]),
    ("is_dag", is_dag(3, [(0,1),(1,2)]), True),
    ("not_dag", is_dag(3, [(0,1),(1,2),(2,0)]), False),
    ("safe_nodes", eventual_safe_nodes([[1,2],[2,3],[5],[0],[5],[],[]]), [2, 4, 5, 6]),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
