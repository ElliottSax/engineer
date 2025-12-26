import heapq
from collections import defaultdict, deque

def dijkstra(n, edges, src):
    """Single source shortest path with non-negative weights."""
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))

    dist = [float('inf')] * n
    dist[src] = 0
    heap = [(0, src)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))

    return dist

def bellman_ford(n, edges, src):
    """Single source shortest path with negative weights."""
    dist = [float('inf')] * n
    dist[src] = 0

    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    # Check negative cycle
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            return None  # Negative cycle

    return dist

def floyd_warshall(n, edges):
    """All pairs shortest path."""
    dist = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        dist[u][v] = w

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] != float('inf') and dist[k][j] != float('inf'):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist

def network_delay_time(times, n, k):
    """Time for signal to reach all nodes from k."""
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))

    dist = dijkstra_general(n + 1, graph, k)
    result = max(dist[1:])
    return result if result != float('inf') else -1

def dijkstra_general(n, graph, src):
    dist = [float('inf')] * n
    dist[src] = 0
    heap = [(0, src)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))

    return dist

def cheapest_flight_k_stops(n, flights, src, dst, k):
    """Cheapest flight with at most k stops."""
    dist = [[float('inf')] * n for _ in range(k + 2)]
    dist[0][src] = 0

    for i in range(1, k + 2):
        dist[i] = dist[i - 1][:]
        for u, v, w in flights:
            if dist[i - 1][u] != float('inf'):
                dist[i][v] = min(dist[i][v], dist[i - 1][u] + w)

    return dist[k + 1][dst] if dist[k + 1][dst] != float('inf') else -1

def path_with_minimum_effort(heights):
    """Minimum effort path in grid."""
    m, n = len(heights), len(heights[0])
    dist = [[float('inf')] * n for _ in range(m)]
    dist[0][0] = 0
    heap = [(0, 0, 0)]  # effort, row, col

    while heap:
        effort, r, c = heapq.heappop(heap)
        if r == m - 1 and c == n - 1:
            return effort
        if effort > dist[r][c]:
            continue
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n:
                new_effort = max(effort, abs(heights[nr][nc] - heights[r][c]))
                if new_effort < dist[nr][nc]:
                    dist[nr][nc] = new_effort
                    heapq.heappush(heap, (new_effort, nr, nc))

    return dist[m-1][n-1]

def swim_in_rising_water(grid):
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

def minimum_cost_to_reach_destination(n, roads, appleCost, k):
    """Minimum cost to buy apple from any city."""
    graph = defaultdict(list)
    for u, v, w in roads:
        graph[u].append((v, w))
        graph[v].append((u, w))

    min_cost = [float('inf')] * (n + 1)

    for start in range(1, n + 1):
        dist = [float('inf')] * (n + 1)
        dist[start] = 0
        heap = [(0, start)]

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            # Buy apple here and return
            cost = appleCost[u - 1] + d * (k + 1)
            min_cost[start] = min(min_cost[start], cost)

            for v, w in graph[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    heapq.heappush(heap, (dist[v], v))

    return min_cost[1:]

def reconstruct_path(n, edges, src, dst):
    """Reconstruct shortest path from src to dst."""
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))

    dist = [float('inf')] * n
    parent = [-1] * n
    dist[src] = 0
    heap = [(0, src)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                parent[v] = u
                heapq.heappush(heap, (dist[v], v))

    if dist[dst] == float('inf'):
        return []

    path = []
    curr = dst
    while curr != -1:
        path.append(curr)
        curr = parent[curr]
    return path[::-1]

def second_shortest_path(n, edges, src, dst):
    """Second shortest path (not strictly shortest)."""
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    dist = [[float('inf'), float('inf')] for _ in range(n)]
    dist[src][0] = 0
    heap = [(0, src, 0)]  # dist, node, which_shortest

    while heap:
        d, u, idx = heapq.heappop(heap)
        if d > dist[u][idx]:
            continue
        for v, w in graph[u]:
            new_dist = d + w
            if new_dist < dist[v][0]:
                dist[v][1] = dist[v][0]
                dist[v][0] = new_dist
                heapq.heappush(heap, (new_dist, v, 0))
            elif dist[v][0] < new_dist < dist[v][1]:
                dist[v][1] = new_dist
                heapq.heappush(heap, (new_dist, v, 1))

    return dist[dst][1] if dist[dst][1] != float('inf') else -1

# Tests
tests = [
    ("dijkstra", dijkstra(5, [(0,1,10),(0,2,3),(1,2,1),(2,1,4),(2,3,2),(3,4,7),(1,3,5)], 0),
     [0, 7, 3, 5, 12]),
    ("bellman_ford", bellman_ford(5, [(0,1,4),(0,2,2),(1,2,3),(2,1,1),(1,3,2),(2,4,5),(3,4,1)], 0),
     [0, 3, 2, 5, 6]),
    ("floyd", floyd_warshall(3, [(0,1,5),(1,2,3),(0,2,10)])[0][2], 8),
    ("network_delay", network_delay_time([[2,1,1],[2,3,1],[3,4,1]], 4, 2), 2),
    ("cheapest_k", cheapest_flight_k_stops(4, [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]], 0, 3, 1), 700),
    ("min_effort", path_with_minimum_effort([[1,2,2],[3,8,2],[5,3,5]]), 2),
    ("swim", swim_in_rising_water([[0,2],[1,3]]), 3),
    ("reconstruct", reconstruct_path(4, [(0,1,1),(1,2,2),(0,2,4),(2,3,1)], 0, 3), [0, 1, 2, 3]),
    ("second_short", second_shortest_path(5, [(0,1,1),(1,2,1),(0,2,3),(2,3,1),(3,4,1)], 0, 4), 5),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
