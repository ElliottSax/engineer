def network_delay_time(times, n, k):
    """Time for signal to reach all nodes (Dijkstra)."""
    import heapq
    from collections import defaultdict
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))
    dist = {k: 0}
    heap = [(0, k)]
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist.get(node, float('inf')):
            continue
        for neighbor, weight in graph[node]:
            new_dist = d + weight
            if new_dist < dist.get(neighbor, float('inf')):
                dist[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor))
    return max(dist.values()) if len(dist) == n else -1

def cheapest_flights_k_stops(n, flights, src, dst, k):
    """Cheapest flight with at most k stops."""
    prices = [float('inf')] * n
    prices[src] = 0
    for _ in range(k + 1):
        temp = prices[:]
        for u, v, w in flights:
            if prices[u] != float('inf'):
                temp[v] = min(temp[v], prices[u] + w)
        prices = temp
    return prices[dst] if prices[dst] != float('inf') else -1

def minimum_spanning_tree_cost(n, edges):
    """Minimum cost to connect all nodes (Kruskal's)."""
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
    cost = 0
    edges_used = 0
    for u, v, w in edges:
        if union(u, v):
            cost += w
            edges_used += 1
            if edges_used == n - 1:
                break
    return cost if edges_used == n - 1 else -1

def find_redundant_connection(edges):
    """Finds edge that creates cycle in undirected graph."""
    parent = list(range(len(edges) + 1))

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

def critical_connections(n, connections):
    """Finds bridges in graph (Tarjan's algorithm)."""
    from collections import defaultdict
    graph = defaultdict(list)
    for u, v in connections:
        graph[u].append(v)
        graph[v].append(u)

    disc = [0] * n
    low = [0] * n
    time = [1]
    bridges = []

    def dfs(node, parent):
        disc[node] = low[node] = time[0]
        time[0] += 1
        for neighbor in graph[node]:
            if disc[neighbor] == 0:
                dfs(neighbor, node)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] > disc[node]:
                    bridges.append([node, neighbor])
            elif neighbor != parent:
                low[node] = min(low[node], disc[neighbor])

    dfs(0, -1)
    return bridges

def accounts_merge(accounts):
    """Merges accounts with same email."""
    from collections import defaultdict
    parent = {}

    def find(x):
        if parent.setdefault(x, x) != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    email_to_name = {}
    for account in accounts:
        name = account[0]
        for email in account[1:]:
            email_to_name[email] = name
            union(account[1], email)

    groups = defaultdict(list)
    for email in email_to_name:
        groups[find(email)].append(email)

    return [[email_to_name[emails[0]]] + sorted(emails) for emails in groups.values()]

def evaluate_division(equations, values, queries):
    """Evaluates division given equations."""
    from collections import defaultdict
    graph = defaultdict(dict)
    for (a, b), val in zip(equations, values):
        graph[a][b] = val
        graph[b][a] = 1 / val

    def dfs(start, end, visited):
        if start not in graph or end not in graph:
            return -1.0
        if start == end:
            return 1.0
        visited.add(start)
        for neighbor, val in graph[start].items():
            if neighbor not in visited:
                result = dfs(neighbor, end, visited)
                if result != -1.0:
                    return val * result
        return -1.0

    return [dfs(a, b, set()) for a, b in queries]

def clone_graph_bfs(node):
    """Clones graph using BFS."""
    if not node:
        return None
    from collections import deque
    cloned = {node['val']: {'val': node['val'], 'neighbors': []}}
    queue = deque([node])
    while queue:
        current = queue.popleft()
        for neighbor in current.get('neighbors', []):
            if neighbor['val'] not in cloned:
                cloned[neighbor['val']] = {'val': neighbor['val'], 'neighbors': []}
                queue.append(neighbor)
            cloned[current['val']]['neighbors'].append(cloned[neighbor['val']])
    return cloned[node['val']]

def all_paths_source_target(graph):
    """All paths from node 0 to node n-1."""
    n = len(graph)
    result = []

    def dfs(node, path):
        if node == n - 1:
            result.append(path[:])
            return
        for neighbor in graph[node]:
            path.append(neighbor)
            dfs(neighbor, path)
            path.pop()

    dfs(0, [0])
    return result

# Tests
tests = [
    ("network_delay", network_delay_time([[2,1,1],[2,3,1],[3,4,1]], 4, 2), 2),
    ("network_delay_no", network_delay_time([[1,2,1]], 2, 2), -1),
    ("cheapest_flights", cheapest_flights_k_stops(3, [[0,1,100],[1,2,100],[0,2,500]], 0, 2, 1), 200),
    ("cheapest_flights_0", cheapest_flights_k_stops(3, [[0,1,100],[1,2,100],[0,2,500]], 0, 2, 0), 500),
    ("mst", minimum_spanning_tree_cost(4, [[0,1,1],[1,2,2],[2,3,3],[0,3,4]]), 6),
    ("redundant", find_redundant_connection([[1,2],[1,3],[2,3]]), [2,3]),
    ("critical", sorted([sorted(e) for e in critical_connections(4, [[0,1],[1,2],[2,0],[1,3]])]), [[1,3]]),
    ("all_paths", all_paths_source_target([[1,2],[3],[3],[]]), [[0,1,3],[0,2,3]]),
]

# Evaluate division test
equations = [["a","b"],["b","c"]]
values = [2.0, 3.0]
queries = [["a","c"],["b","a"],["a","e"]]
results = evaluate_division(equations, values, queries)
tests.append(("eval_div", results, [6.0, 0.5, -1.0]))

# Accounts merge test
accounts = [["John","johnsmith@mail.com","john_newyork@mail.com"],
            ["John","johnsmith@mail.com","john00@mail.com"],
            ["Mary","mary@mail.com"]]
merged = accounts_merge(accounts)
tests.append(("accounts_len", len(merged), 2))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
