#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ULTRA TRAINING ITERATION 125                              ║
║               Graph Shortest Paths - Complete Collection                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import heapq
from collections import defaultdict, deque

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 1: Dijkstra's Algorithm (Single Source, Non-negative weights)
# ═══════════════════════════════════════════════════════════════════════════════
def dijkstra(n, edges, source):
    """O((V + E) log V) shortest paths from source."""
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))

    dist = [float('inf')] * n
    dist[source] = 0
    pq = [(0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))

    return dist

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 2: Bellman-Ford (Handles negative weights)
# ═══════════════════════════════════════════════════════════════════════════════
def bellman_ford(n, edges, source):
    """O(VE) shortest paths, detects negative cycles."""
    dist = [float('inf')] * n
    dist[source] = 0

    # Relax all edges V-1 times
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    # Check for negative cycles
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            return None  # Negative cycle detected

    return dist

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 3: Floyd-Warshall (All Pairs)
# ═══════════════════════════════════════════════════════════════════════════════
def floyd_warshall(n, edges):
    """O(V³) all-pairs shortest paths."""
    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]

    for i in range(n):
        dist[i][i] = 0

    for u, v, w in edges:
        dist[u][v] = min(dist[u][v], w)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] != INF and dist[k][j] != INF:
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 4: 0-1 BFS (Binary edge weights)
# ═══════════════════════════════════════════════════════════════════════════════
def bfs_01(n, edges, source):
    """O(V + E) for graphs with edge weights 0 or 1."""
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))

    dist = [float('inf')] * n
    dist[source] = 0
    dq = deque([source])

    while dq:
        u = dq.popleft()
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                if w == 0:
                    dq.appendleft(v)
                else:
                    dq.append(v)

    return dist

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 5: Johnson's Algorithm (All Pairs, handles negative)
# ═══════════════════════════════════════════════════════════════════════════════
def johnson(n, edges):
    """O(VE log V) all-pairs shortest paths with negative edges."""
    # Add virtual vertex connected to all with weight 0
    new_edges = edges + [(n, i, 0) for i in range(n)]

    # Run Bellman-Ford from virtual vertex
    h = bellman_ford(n + 1, new_edges, n)
    if h is None:
        return None  # Negative cycle

    # Reweight edges
    reweighted = []
    for u, v, w in edges:
        reweighted.append((u, v, w + h[u] - h[v]))

    # Run Dijkstra from each vertex
    dist = []
    for i in range(n):
        d = dijkstra(n, reweighted, i)
        # Restore original distances
        original = []
        for j in range(n):
            if d[j] == float('inf'):
                original.append(float('inf'))
            else:
                original.append(d[j] - h[i] + h[j])
        dist.append(original)

    return dist

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 6: A* Search (Heuristic-guided)
# ═══════════════════════════════════════════════════════════════════════════════
def a_star(n, edges, source, target, heuristic):
    """A* pathfinding with heuristic function."""
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))

    dist = [float('inf')] * n
    dist[source] = 0
    pq = [(heuristic(source), 0, source)]  # (f, g, node)

    while pq:
        f, g, u = heapq.heappop(pq)
        if u == target:
            return dist[target]
        if g > dist[u]:
            continue
        for v, w in graph[u]:
            new_g = g + w
            if new_g < dist[v]:
                dist[v] = new_g
                heapq.heappush(pq, (new_g + heuristic(v), new_g, v))

    return dist[target]

# ═══════════════════════════════════════════════════════════════════════════════
# ALGORITHM 7: K Shortest Paths
# ═══════════════════════════════════════════════════════════════════════════════
def k_shortest_paths(n, edges, source, target, k):
    """Find k shortest paths using Yen's algorithm."""
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))

    def dijkstra_path(blocked_edges, blocked_nodes):
        dist = {source: 0}
        pq = [(0, source, [source])]

        while pq:
            d, u, path = heapq.heappop(pq)
            if u == target:
                return d, path
            if d > dist.get(u, float('inf')):
                continue
            for v, w in graph[u]:
                if v in blocked_nodes:
                    continue
                if (u, v) in blocked_edges:
                    continue
                new_d = d + w
                if new_d < dist.get(v, float('inf')):
                    dist[v] = new_d
                    heapq.heappush(pq, (new_d, v, path + [v]))

        return float('inf'), []

    # Find first shortest path
    d, path = dijkstra_path(set(), set())
    if not path:
        return []

    results = [(d, path)]
    candidates = []

    for i in range(k - 1):
        last_path = results[-1][1]
        for j in range(len(last_path) - 1):
            spur_node = last_path[j]
            root_path = last_path[:j + 1]

            blocked_edges = set()
            for dist_p, p in results:
                if p[:j + 1] == root_path:
                    blocked_edges.add((p[j], p[j + 1]))

            blocked_nodes = set(root_path[:-1])

            spur_dist, spur_path = dijkstra_path(blocked_edges, blocked_nodes)
            if spur_path:
                total_dist = sum(
                    next((w for v2, w in graph[root_path[x]] if v2 == root_path[x + 1]), 0)
                    for x in range(j)
                ) + spur_dist
                full_path = root_path[:-1] + spur_path
                heapq.heappush(candidates, (total_dist, full_path))

        if not candidates:
            break

        next_best = heapq.heappop(candidates)
        results.append(next_best)

    return results

# ═══════════════════════════════════════════════════════════════════════════════
#                              TEST SUITE
# ═══════════════════════════════════════════════════════════════════════════════
def run_tests():
    tests = []

    # Test graph
    edges = [(0, 1, 4), (0, 2, 1), (2, 1, 2), (1, 3, 1), (2, 3, 5)]

    # Test 1: Dijkstra
    dist = dijkstra(4, edges, 0)
    tests.append(("dijkstra_0", dist[0], 0))
    tests.append(("dijkstra_3", dist[3], 4))  # 0->2->1->3

    # Test 2: Bellman-Ford
    dist_bf = bellman_ford(4, edges, 0)
    tests.append(("bellman", dist_bf[3], 4))

    # Test 3: Bellman-Ford with negative cycle
    neg_edges = [(0, 1, 1), (1, 2, -1), (2, 0, -1)]
    tests.append(("neg_cycle", bellman_ford(3, neg_edges, 0), None))

    # Test 4: Floyd-Warshall
    dist_fw = floyd_warshall(4, edges)
    tests.append(("floyd_0_3", dist_fw[0][3], 4))
    tests.append(("floyd_2_3", dist_fw[2][3], 3))

    # Test 5: 0-1 BFS
    edges_01 = [(0, 1, 0), (0, 2, 1), (1, 3, 1), (2, 3, 0)]
    dist_01 = bfs_01(4, edges_01, 0)
    tests.append(("bfs01", dist_01[3], 1))

    # Test 6: A*
    def h(v):
        return 0  # No heuristic = Dijkstra
    dist_astar = a_star(4, edges, 0, 3, h)
    tests.append(("astar", dist_astar, 4))

    # Test 7: K shortest paths
    ksp = k_shortest_paths(4, edges, 0, 3, 2)
    tests.append(("ksp_count", len(ksp), 2))

    # Run all tests
    passed = 0
    print("\n" + "─" * 60)
    for name, result, expected in tests:
        if result == expected:
            passed += 1
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name}: got {result}, expected {expected}")

    print("─" * 60)
    print(f"\n  📊 Results: {passed}/{len(tests)} tests passed")
    return passed, len(tests)

if __name__ == "__main__":
    print(__doc__)
    passed, total = run_tests()
    if passed == total:
        print("\n  🎯 PERFECT SCORE!")
