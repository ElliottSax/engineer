def largest_component_by_common_factor(nums):
    """Largest connected component where edges are common factors > 1."""
    from collections import defaultdict

    def factorize(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    parent = {}
    rank = {}

    def find(x):
        if x not in parent:
            parent[x] = x
            rank[x] = 0
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1

    factor_to_idx = {}
    for i, num in enumerate(nums):
        factors = factorize(num)
        for f in factors:
            if f in factor_to_idx:
                union(i, factor_to_idx[f])
            factor_to_idx[f] = i

    count = defaultdict(int)
    for i in range(len(nums)):
        count[find(i)] += 1
    return max(count.values()) if count else 0

def regions_by_slashes(grid):
    """Counts regions divided by / and \\ in grid."""
    n = len(grid)
    parent = list(range(4 * n * n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(n):
            root = 4 * (i * n + j)
            char = grid[i][j]
            # Connect triangles within cell
            if char == '/':
                union(root + 0, root + 3)
                union(root + 1, root + 2)
            elif char == '\\':
                union(root + 0, root + 1)
                union(root + 2, root + 3)
            else:
                union(root + 0, root + 1)
                union(root + 1, root + 2)
                union(root + 2, root + 3)
            # Connect to adjacent cells
            if i + 1 < n:
                union(root + 2, root + 4 * n + 0)
            if j + 1 < n:
                union(root + 1, root + 4 + 3)

    return len(set(find(i) for i in range(4 * n * n)))

def satisfiability_equations(equations):
    """Checks if equality/inequality equations are satisfiable."""
    parent = {chr(i): chr(i) for i in range(ord('a'), ord('z') + 1)}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    # Process equalities first
    for eq in equations:
        if eq[1] == '=':
            union(eq[0], eq[3])

    # Check inequalities
    for eq in equations:
        if eq[1] == '!':
            if find(eq[0]) == find(eq[3]):
                return False
    return True

def similar_string_groups(strs):
    """Counts groups of similar strings (differ by at most 2 positions)."""
    def similar(s1, s2):
        diff = sum(c1 != c2 for c1, c2 in zip(s1, s2))
        return diff == 0 or diff == 2

    n = len(strs)
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            if similar(strs[i], strs[j]):
                union(i, j)

    return len(set(find(i) for i in range(n)))

def swim_in_rising_water(grid):
    """Minimum time to swim from top-left to bottom-right."""
    import heapq
    n = len(grid)
    heap = [(grid[0][0], 0, 0)]
    visited = [[False] * n for _ in range(n)]
    visited[0][0] = True

    while heap:
        t, r, c = heapq.heappop(heap)
        if r == n - 1 and c == n - 1:
            return t
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and not visited[nr][nc]:
                visited[nr][nc] = True
                heapq.heappush(heap, (max(t, grid[nr][nc]), nr, nc))
    return -1

def path_with_minimum_effort(heights):
    """Minimum effort path from top-left to bottom-right."""
    import heapq
    m, n = len(heights), len(heights[0])
    efforts = [[float('inf')] * n for _ in range(m)]
    efforts[0][0] = 0
    heap = [(0, 0, 0)]

    while heap:
        effort, r, c = heapq.heappop(heap)
        if r == m - 1 and c == n - 1:
            return effort
        if effort > efforts[r][c]:
            continue
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n:
                new_effort = max(effort, abs(heights[nr][nc] - heights[r][c]))
                if new_effort < efforts[nr][nc]:
                    efforts[nr][nc] = new_effort
                    heapq.heappush(heap, (new_effort, nr, nc))
    return efforts[m-1][n-1]

def minimize_malware_spread(graph, initial):
    """Minimizes malware spread by removing one initial node."""
    n = len(graph)
    parent = list(range(n))
    size = [1] * n

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            if size[px] < size[py]:
                px, py = py, px
            parent[py] = px
            size[px] += size[py]

    for i in range(n):
        for j in range(i + 1, n):
            if graph[i][j]:
                union(i, j)

    from collections import Counter
    initial_set = set(initial)
    # Count initial nodes per component
    comp_count = Counter(find(i) for i in initial)

    # Find component with exactly one initial node and max size
    best = min(initial)
    max_save = 0
    for node in initial:
        root = find(node)
        if comp_count[root] == 1:
            if size[root] > max_save or (size[root] == max_save and node < best):
                max_save = size[root]
                best = node
    return best

# Tests
tests = [
    ("largest_component", largest_component_by_common_factor([4,6,15,35]), 4),
    ("largest_component_2", largest_component_by_common_factor([20,50,9,63]), 2),
    ("regions", regions_by_slashes([" /","/ "]), 2),
    ("regions_2", regions_by_slashes([" /","/  "]), 1),
    ("equations", satisfiability_equations(["a==b","b!=a"]), False),
    ("equations_2", satisfiability_equations(["a==b","b==c","a==c"]), True),
    ("similar_groups", similar_string_groups(["tars","rats","arts","star"]), 2),
    ("swim", swim_in_rising_water([[0,2],[1,3]]), 3),
    ("swim_2", swim_in_rising_water([[0,1,2,3,4],[24,23,22,21,5],[12,13,14,15,16],[11,17,18,19,20],[10,9,8,7,6]]), 16),
    ("min_effort", path_with_minimum_effort([[1,2,2],[3,8,2],[5,3,5]]), 2),
    ("malware", minimize_malware_spread([[1,1,0],[1,1,0],[0,0,1]], [0,1]), 0),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
