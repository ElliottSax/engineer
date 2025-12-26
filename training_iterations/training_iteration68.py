from collections import defaultdict, deque
import heapq

def prims_mst(n, edges):
    """Prim's algorithm for minimum spanning tree."""
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((w, v))
        graph[v].append((w, u))

    visited = set()
    total = 0
    heap = [(0, 0)]  # (weight, node)

    while heap and len(visited) < n:
        weight, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        total += weight
        for w, neighbor in graph[node]:
            if neighbor not in visited:
                heapq.heappush(heap, (w, neighbor))

    return total if len(visited) == n else -1

def kruskals_mst(n, edges):
    """Kruskal's algorithm for minimum spanning tree."""
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True

    edges.sort(key=lambda x: x[2])
    total = 0
    count = 0

    for u, v, w in edges:
        if union(u, v):
            total += w
            count += 1
            if count == n - 1:
                break

    return total if count == n - 1 else -1

def min_cost_connect_cities(n, connections):
    """Minimum cost to connect all cities."""
    return kruskals_mst(n, connections)

def critical_connections(n, connections):
    """Find bridges in graph."""
    graph = defaultdict(list)
    for u, v in connections:
        graph[u].append(v)
        graph[v].append(u)

    disc = [0] * n
    low = [0] * n
    bridges = []
    time = [1]

    def dfs(node, parent):
        disc[node] = low[node] = time[0]
        time[0] += 1

        for neighbor in graph[node]:
            if neighbor == parent:
                continue
            if disc[neighbor] == 0:
                dfs(neighbor, node)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] > disc[node]:
                    bridges.append([node, neighbor])
            else:
                low[node] = min(low[node], disc[neighbor])

    for i in range(n):
        if disc[i] == 0:
            dfs(i, -1)

    return bridges

def redundant_connection(edges):
    """Find edge causing cycle in tree + 1 edge."""
    n = len(edges)
    parent = list(range(n + 1))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    for u, v in edges:
        pu, pv = find(u), find(v)
        if pu == pv:
            return [u, v]
        parent[pu] = pv

    return []

def min_spanning_forest_k_trees(n, edges, k):
    """Minimum cost to have exactly k connected components."""
    edges.sort(key=lambda x: x[2])
    parent = list(range(n))
    components = n

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    total = 0
    for u, v, w in edges:
        if components == k:
            break
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pu] = pv
            total += w
            components -= 1

    return total if components == k else -1

def accounts_merge(accounts):
    """Merge accounts with common emails."""
    email_to_name = {}
    graph = defaultdict(set)

    for account in accounts:
        name = account[0]
        for email in account[1:]:
            email_to_name[email] = name
            graph[account[1]].add(email)
            graph[email].add(account[1])

    visited = set()
    result = []

    def dfs(email, emails):
        if email in visited:
            return
        visited.add(email)
        emails.append(email)
        for neighbor in graph[email]:
            dfs(neighbor, emails)

    for email in email_to_name:
        if email not in visited:
            emails = []
            dfs(email, emails)
            result.append([email_to_name[email]] + sorted(emails))

    return sorted(result)

def similar_string_groups(strs):
    """Number of groups of similar strings."""
    n = len(strs)
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def similar(s1, s2):
        diff = 0
        for c1, c2 in zip(s1, s2):
            if c1 != c2:
                diff += 1
            if diff > 2:
                return False
        return diff == 0 or diff == 2

    for i in range(n):
        for j in range(i + 1, n):
            if similar(strs[i], strs[j]):
                pi, pj = find(i), find(j)
                if pi != pj:
                    parent[pi] = pj

    return len(set(find(i) for i in range(n)))

def swim_in_water(grid):
    """Minimum time to swim from (0,0) to (n-1,n-1)."""
    n = len(grid)
    dist = [[float('inf')] * n for _ in range(n)]
    dist[0][0] = grid[0][0]
    heap = [(grid[0][0], 0, 0)]

    while heap:
        t, r, c = heapq.heappop(heap)
        if r == n - 1 and c == n - 1:
            return t
        if t > dist[r][c]:
            continue
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                new_t = max(t, grid[nr][nc])
                if new_t < dist[nr][nc]:
                    dist[nr][nc] = new_t
                    heapq.heappush(heap, (new_t, nr, nc))

    return dist[n-1][n-1]

def min_cost_to_make_valid_path(grid):
    """Minimum cost to make valid path from (0,0) to (n-1,m-1)."""
    m, n = len(grid), len(grid[0])
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up (1-indexed in problem)
    dist = [[float('inf')] * n for _ in range(m)]
    dist[0][0] = 0
    dq = deque([(0, 0, 0)])  # cost, row, col

    while dq:
        cost, r, c = dq.popleft()
        if cost > dist[r][c]:
            continue
        for i, (dr, dc) in enumerate(dirs):
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n:
                new_cost = cost + (0 if grid[r][c] == i + 1 else 1)
                if new_cost < dist[nr][nc]:
                    dist[nr][nc] = new_cost
                    if grid[r][c] == i + 1:
                        dq.appendleft((new_cost, nr, nc))
                    else:
                        dq.append((new_cost, nr, nc))

    return dist[m-1][n-1]

# Tests
edges_mst = [(0, 1, 4), (0, 7, 8), (1, 2, 8), (1, 7, 11), (2, 3, 7), (2, 5, 4), (2, 8, 2),
             (3, 4, 9), (3, 5, 14), (4, 5, 10), (5, 6, 2), (6, 7, 1), (6, 8, 6), (7, 8, 7)]

tests = [
    ("prims", prims_mst(9, edges_mst), 37),
    ("kruskals", kruskals_mst(9, edges_mst), 37),
    ("connect_cities", min_cost_connect_cities(3, [(0, 1, 5), (0, 2, 6), (1, 2, 1)]), 6),
    ("bridges", len(critical_connections(4, [[0,1],[1,2],[2,0],[1,3]])), 1),
    ("redundant", redundant_connection([[1,2],[1,3],[2,3]]), [2, 3]),
    ("accounts", len(accounts_merge([["John","j1@mail.com","j2@mail.com"],
                                      ["John","j1@mail.com","j3@mail.com"],
                                      ["Mary","mary@mail.com"]])), 2),
    ("similar", similar_string_groups(["tars","rats","arts","star"]), 2),
    ("swim", swim_in_water([[0,2],[1,3]]), 3),
    ("valid_path", min_cost_to_make_valid_path([[1,1,1,1],[2,2,2,2],[1,1,1,1],[2,2,2,2]]), 3),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
