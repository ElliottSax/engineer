from collections import defaultdict
import heapq

def alien_order(words):
    """Derive alien dictionary order."""
    graph = defaultdict(set)
    in_degree = {c: 0 for word in words for c in word}

    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        min_len = min(len(w1), len(w2))
        if len(w1) > len(w2) and w1[:min_len] == w2[:min_len]:
            return ""
        for j in range(min_len):
            if w1[j] != w2[j]:
                if w2[j] not in graph[w1[j]]:
                    graph[w1[j]].add(w2[j])
                    in_degree[w2[j]] += 1
                break

    from collections import deque
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

def reconstruct_itinerary(tickets):
    """Reconstruct itinerary from tickets."""
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

def word_ladder(begin_word, end_word, word_list):
    """Shortest transformation sequence length."""
    if end_word not in word_list:
        return 0

    word_set = set(word_list)
    from collections import deque
    queue = deque([(begin_word, 1)])
    visited = {begin_word}

    while queue:
        word, length = queue.popleft()
        if word == end_word:
            return length
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                new_word = word[:i] + c + word[i+1:]
                if new_word in word_set and new_word not in visited:
                    visited.add(new_word)
                    queue.append((new_word, length + 1))

    return 0

def word_ladder_ii(begin_word, end_word, word_list):
    """All shortest transformation sequences."""
    if end_word not in word_list:
        return []

    word_set = set(word_list)
    from collections import deque

    # BFS to find shortest path length and build parent graph
    parents = defaultdict(set)
    queue = deque([begin_word])
    visited = {begin_word: 0}
    found = False
    level = 0

    while queue and not found:
        level += 1
        level_size = len(queue)
        for _ in range(level_size):
            word = queue.popleft()
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    new_word = word[:i] + c + word[i+1:]
                    if new_word == end_word:
                        found = True
                        parents[new_word].add(word)
                    if new_word in word_set:
                        if new_word not in visited:
                            visited[new_word] = level
                            queue.append(new_word)
                            parents[new_word].add(word)
                        elif visited[new_word] == level:
                            parents[new_word].add(word)

    # Backtrack to find all paths
    if not found:
        return []

    result = []
    def backtrack(word, path):
        if word == begin_word:
            result.append([begin_word] + path[::-1])
            return
        for parent in parents[word]:
            backtrack(parent, path + [word])

    backtrack(end_word, [])
    return result

def evaluate_division(equations, values, queries):
    """Evaluate division queries."""
    graph = defaultdict(dict)
    for (a, b), val in zip(equations, values):
        graph[a][b] = val
        graph[b][a] = 1 / val

    def dfs(src, dst, visited):
        if src not in graph or dst not in graph:
            return -1.0
        if dst in graph[src]:
            return graph[src][dst]
        visited.add(src)
        for neighbor, val in graph[src].items():
            if neighbor not in visited:
                result = dfs(neighbor, dst, visited)
                if result != -1.0:
                    return val * result
        return -1.0

    return [dfs(a, b, set()) for a, b in queries]

def shortest_path_visiting_all_nodes(graph):
    """Shortest path to visit all nodes."""
    n = len(graph)
    target = (1 << n) - 1

    from collections import deque
    queue = deque([(i, 1 << i, 0) for i in range(n)])
    visited = {(i, 1 << i) for i in range(n)}

    while queue:
        node, mask, dist = queue.popleft()
        if mask == target:
            return dist
        for neighbor in graph[node]:
            new_mask = mask | (1 << neighbor)
            if (neighbor, new_mask) not in visited:
                visited.add((neighbor, new_mask))
                queue.append((neighbor, new_mask, dist + 1))

    return -1

def find_cheapest_price(n, flights, src, dst, k):
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

# Tests
tests = [
    ("alien", alien_order(["wrt","wrf","er","ett","rftt"]), "wertf"),
    ("alien_invalid", alien_order(["z","x","z"]), ""),
    ("itinerary", reconstruct_itinerary([["MU","LHR"],["JFK","MU"],["SFO","SJC"],["LHR","SFO"]]),
     ["JFK","MU","LHR","SFO","SJC"]),
    ("ladder", word_ladder("hit", "cog", ["hot","dot","dog","lot","log","cog"]), 5),
    ("ladder_no", word_ladder("hit", "cog", ["hot","dot","dog","lot","log"]), 0),
    ("ladder_ii", len(word_ladder_ii("hit", "cog", ["hot","dot","dog","lot","log","cog"])), 2),
    ("division", evaluate_division([["a","b"],["b","c"]], [2.0,3.0], [["a","c"],["b","a"],["a","e"]]),
     [6.0, 0.5, -1.0]),
    ("all_nodes", shortest_path_visiting_all_nodes([[1,2,3],[0],[0],[0]]), 4),
    ("cheapest", find_cheapest_price(4, [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]], 0, 3, 1), 700),
]

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
