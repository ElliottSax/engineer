def word_ladder_ii(begin_word, end_word, word_list):
    """All shortest transformation sequences."""
    from collections import defaultdict, deque
    word_set = set(word_list)
    if end_word not in word_set:
        return []

    # BFS to find shortest paths
    layer = defaultdict(list)  # word -> list of predecessors
    current_layer = {begin_word}
    found = False

    while current_layer and not found:
        word_set -= current_layer
        next_layer = defaultdict(list)
        for word in current_layer:
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    new_word = word[:i] + c + word[i+1:]
                    if new_word in word_set:
                        next_layer[new_word].append(word)
                        if new_word == end_word:
                            found = True
        current_layer = set(next_layer.keys())
        for k, v in next_layer.items():
            layer[k].extend(v)

    # Backtrack to find all paths
    result = []

    def backtrack(word, path):
        if word == begin_word:
            result.append(path[::-1])
            return
        for pred in layer[word]:
            backtrack(pred, path + [pred])

    if found:
        backtrack(end_word, [end_word])
    return result

def find_itinerary(tickets):
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

def pacific_atlantic_water_flow(heights):
    """Cells that can flow to both Pacific and Atlantic."""
    if not heights:
        return []
    m, n = len(heights), len(heights[0])

    def bfs(starts):
        visited = set(starts)
        queue = list(starts)
        for r, c in queue:
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < m and 0 <= nc < n and
                    (nr, nc) not in visited and
                    heights[nr][nc] >= heights[r][c]):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return visited

    pacific = [(0, j) for j in range(n)] + [(i, 0) for i in range(1, m)]
    atlantic = [(m-1, j) for j in range(n)] + [(i, n-1) for i in range(m-1)]

    pac_reach = bfs(pacific)
    atl_reach = bfs(atlantic)

    return sorted([[r, c] for r, c in pac_reach & atl_reach])

def bus_routes(routes, source, target):
    """Minimum buses to reach target from source."""
    from collections import defaultdict, deque
    if source == target:
        return 0

    stop_to_buses = defaultdict(set)
    for i, route in enumerate(routes):
        for stop in route:
            stop_to_buses[stop].add(i)

    visited_stops = {source}
    visited_buses = set()
    queue = deque([(source, 0)])

    while queue:
        stop, buses = queue.popleft()
        for bus in stop_to_buses[stop]:
            if bus in visited_buses:
                continue
            visited_buses.add(bus)
            for next_stop in routes[bus]:
                if next_stop == target:
                    return buses + 1
                if next_stop not in visited_stops:
                    visited_stops.add(next_stop)
                    queue.append((next_stop, buses + 1))
    return -1

def sequence_reconstruction(nums, sequences):
    """Checks if nums is uniquely reconstructable from sequences."""
    from collections import defaultdict, deque
    n = len(nums)
    if n == 0:
        return len(sequences) == 0

    graph = defaultdict(set)
    in_degree = defaultdict(int)
    nodes = set()

    for seq in sequences:
        nodes.update(seq)
        for i in range(len(seq) - 1):
            if seq[i + 1] not in graph[seq[i]]:
                graph[seq[i]].add(seq[i + 1])
                in_degree[seq[i + 1]] += 1

    if nodes != set(nums):
        return False

    queue = deque([x for x in nodes if in_degree[x] == 0])
    result = []

    while queue:
        if len(queue) > 1:
            return False
        node = queue.popleft()
        result.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result == nums

def loud_and_rich(richer, quiet):
    """For each person, find quietest person at least as rich."""
    from collections import defaultdict
    n = len(quiet)
    graph = defaultdict(list)
    for a, b in richer:
        graph[b].append(a)

    answer = [-1] * n

    def dfs(node):
        if answer[node] != -1:
            return answer[node]
        answer[node] = node
        for richer_person in graph[node]:
            candidate = dfs(richer_person)
            if quiet[candidate] < quiet[answer[node]]:
                answer[node] = candidate
        return answer[node]

    for i in range(n):
        dfs(i)
    return answer

def is_graph_bipartite(graph):
    """Checks if graph is bipartite using BFS."""
    from collections import deque
    n = len(graph)
    color = [-1] * n
    for start in range(n):
        if color[start] == -1:
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

def possible_bipartition(n, dislikes):
    """Can we split into two groups where no two in same group dislike each other?"""
    from collections import defaultdict, deque
    graph = defaultdict(list)
    for a, b in dislikes:
        graph[a].append(b)
        graph[b].append(a)

    color = {}
    for node in range(1, n + 1):
        if node not in color:
            queue = deque([node])
            color[node] = 0
            while queue:
                curr = queue.popleft()
                for neighbor in graph[curr]:
                    if neighbor not in color:
                        color[neighbor] = 1 - color[curr]
                        queue.append(neighbor)
                    elif color[neighbor] == color[curr]:
                        return False
    return True

# Tests
tests = [
    ("word_ladder_ii", len(word_ladder_ii("hit", "cog", ["hot","dot","dog","lot","log","cog"])), 2),
    ("itinerary", find_itinerary([["MU","LHR"],["JFK","MU"],["LHR","SFO"],["SFO","JFK"]]),
     ["JFK","MU","LHR","SFO","JFK"]),
    ("pacific_atlantic", len(pacific_atlantic_water_flow([[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]])), 7),
    ("bus_routes", bus_routes([[1,2,7],[3,6,7]], 1, 6), 2),
    ("bus_routes_same", bus_routes([[1,2,7],[3,6,7]], 7, 7), 0),
    ("seq_reconstruct", sequence_reconstruction([1,2,3], [[1,2],[1,3],[2,3]]), True),
    ("seq_reconstruct_no", sequence_reconstruction([1,2,3], [[1,2],[1,3]]), False),
    ("loud_rich", loud_and_rich([[1,0],[2,1],[3,1],[3,7],[4,3],[5,3],[6,3]], [3,2,5,4,6,1,7,0]),
     [5,5,2,5,4,5,6,7]),
    ("bipartite", is_graph_bipartite([[1,3],[0,2],[1,3],[0,2]]), True),
    ("bipartite_no", is_graph_bipartite([[1,2,3],[0,2],[0,1,3],[0,2]]), False),
    ("partition", possible_bipartition(4, [[1,2],[1,3],[2,4]]), True),
    ("partition_no", possible_bipartition(3, [[1,2],[1,3],[2,3]]), False),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
