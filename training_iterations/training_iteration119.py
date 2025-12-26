# ULTRA: Advanced Heuristic Search Algorithms

import heapq
from collections import deque

# ULTRA: A* Search
def a_star(start, goal, neighbors_fn, heuristic_fn):
    """A* pathfinding algorithm."""
    open_set = [(heuristic_fn(start, goal), 0, start, [start])]
    closed_set = set()

    while open_set:
        f, g, current, path = heapq.heappop(open_set)

        if current == goal:
            return path, g

        if current in closed_set:
            continue
        closed_set.add(current)

        for neighbor, cost in neighbors_fn(current):
            if neighbor in closed_set:
                continue
            new_g = g + cost
            new_f = new_g + heuristic_fn(neighbor, goal)
            heapq.heappush(open_set, (new_f, new_g, neighbor, path + [neighbor]))

    return None, float('inf')

# ULTRA: IDA* (Iterative Deepening A*)
def ida_star(start, goal, neighbors_fn, heuristic_fn):
    """IDA* - A* with iterative deepening to save memory."""
    def search(path, g, bound):
        node = path[-1]
        f = g + heuristic_fn(node, goal)

        if f > bound:
            return f, None
        if node == goal:
            return f, path[:]

        min_bound = float('inf')
        for neighbor, cost in neighbors_fn(node):
            if neighbor not in path:  # Avoid cycles
                path.append(neighbor)
                t, result = search(path, g + cost, bound)
                if result is not None:
                    return t, result
                min_bound = min(min_bound, t)
                path.pop()

        return min_bound, None

    bound = heuristic_fn(start, goal)
    path = [start]

    while True:
        t, result = search(path, 0, bound)
        if result is not None:
            return result, t
        if t == float('inf'):
            return None, float('inf')
        bound = t

# ULTRA: Bidirectional BFS
def bidirectional_bfs(start, goal, neighbors_fn):
    """Bidirectional BFS for unweighted graphs."""
    if start == goal:
        return [start], 0

    forward = {start: [start]}
    backward = {goal: [goal]}
    forward_queue = deque([start])
    backward_queue = deque([goal])

    while forward_queue and backward_queue:
        # Forward step
        if forward_queue:
            current = forward_queue.popleft()
            for neighbor, _ in neighbors_fn(current):
                if neighbor in backward:
                    # Found meeting point
                    path = forward[current] + backward[neighbor][::-1]
                    return path, len(path) - 1
                if neighbor not in forward:
                    forward[neighbor] = forward[current] + [neighbor]
                    forward_queue.append(neighbor)

        # Backward step
        if backward_queue:
            current = backward_queue.popleft()
            for neighbor, _ in neighbors_fn(current):
                if neighbor in forward:
                    # Found meeting point
                    path = forward[neighbor] + backward[current][::-1]
                    return path, len(path) - 1
                if neighbor not in backward:
                    backward[neighbor] = backward[current] + [neighbor]
                    backward_queue.append(neighbor)

    return None, float('inf')

# ULTRA: Jump Point Search (Simplified for grid)
def jump_point_search(grid, start, goal):
    """JPS for uniform-cost grid pathfinding."""
    rows, cols = len(grid), len(grid[0]) if grid else 0

    def is_valid(r, c):
        return 0 <= r < rows and 0 <= c < cols and grid[r][c] == 0

    def heuristic(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def jump(r, c, dr, dc, parent):
        nr, nc = r + dr, c + dc
        if not is_valid(nr, nc):
            return None

        if (nr, nc) == goal:
            return (nr, nc)

        # Diagonal movement
        if dr != 0 and dc != 0:
            # Check forced neighbors
            if (is_valid(nr - dr, nc + dc) and not is_valid(nr - dr, nc)) or \
               (is_valid(nr + dr, nc - dc) and not is_valid(nr, nc - dc)):
                return (nr, nc)
            # Recursive horizontal/vertical
            if jump(nr, nc, dr, 0, (nr, nc)) is not None or \
               jump(nr, nc, 0, dc, (nr, nc)) is not None:
                return (nr, nc)
        else:
            # Horizontal/vertical movement
            if dr != 0:
                if (is_valid(nr, nc + 1) and not is_valid(r, nc + 1)) or \
                   (is_valid(nr, nc - 1) and not is_valid(r, nc - 1)):
                    return (nr, nc)
            else:
                if (is_valid(nr + 1, nc) and not is_valid(nr + 1, c)) or \
                   (is_valid(nr - 1, nc) and not is_valid(nr - 1, c)):
                    return (nr, nc)

        return jump(nr, nc, dr, dc, (nr, nc))

    # Standard A* with jump points
    open_set = [(heuristic(start), 0, start, [start])]
    visited = {start: 0}

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                  (1, 1), (1, -1), (-1, 1), (-1, -1)]

    while open_set:
        f, g, current, path = heapq.heappop(open_set)

        if current == goal:
            return path, g

        if g > visited.get(current, float('inf')):
            continue

        for dr, dc in directions:
            jp = jump(current[0], current[1], dr, dc, current)
            if jp is not None:
                new_g = g + abs(jp[0] - current[0]) + abs(jp[1] - current[1])
                if new_g < visited.get(jp, float('inf')):
                    visited[jp] = new_g
                    new_f = new_g + heuristic(jp)
                    heapq.heappush(open_set, (new_f, new_g, jp, path + [jp]))

    return None, float('inf')

# ULTRA: Beam Search
def beam_search(start, goal, neighbors_fn, heuristic_fn, beam_width=3):
    """Beam search - keeps only top k nodes at each level."""
    beam = [(heuristic_fn(start, goal), start, [start])]

    while beam:
        candidates = []
        for _, current, path in beam:
            if current == goal:
                return path, len(path) - 1

            for neighbor, cost in neighbors_fn(current):
                if neighbor not in path:
                    h = heuristic_fn(neighbor, goal)
                    candidates.append((h, neighbor, path + [neighbor]))

        if not candidates:
            break

        # Keep top beam_width candidates
        candidates.sort(key=lambda x: x[0])
        beam = candidates[:beam_width]

    return None, float('inf')

# ULTRA: Hill Climbing with Random Restarts
def hill_climbing_restarts(initial_fn, neighbors_fn, evaluate_fn, max_restarts=10, max_steps=100):
    """Hill climbing with random restarts."""
    best_solution = None
    best_score = float('-inf')

    for _ in range(max_restarts):
        current = initial_fn()
        current_score = evaluate_fn(current)

        for _ in range(max_steps):
            neighbors = neighbors_fn(current)
            if not neighbors:
                break

            best_neighbor = max(neighbors, key=evaluate_fn)
            neighbor_score = evaluate_fn(best_neighbor)

            if neighbor_score <= current_score:
                break

            current = best_neighbor
            current_score = neighbor_score

        if current_score > best_score:
            best_score = current_score
            best_solution = current

    return best_solution, best_score

# Tests
tests = []

# Grid for testing
grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0]
]

def grid_neighbors(pos):
    r, c = pos
    result = []
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 5 and 0 <= nc < 5 and grid[nr][nc] == 0:
            result.append(((nr, nc), 1))
    return result

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# A*
path, cost = a_star((0, 0), (4, 4), grid_neighbors, manhattan)
tests.append(("astar_found", path is not None, True))
tests.append(("astar_cost", cost <= 8, True))

# IDA*
path_ida, cost_ida = ida_star((0, 0), (4, 4), grid_neighbors, manhattan)
tests.append(("idastar_found", path_ida is not None, True))

# Bidirectional BFS
path_bi, cost_bi = bidirectional_bfs((0, 0), (4, 4), grid_neighbors)
tests.append(("bidir_found", path_bi is not None, True))

# JPS
path_jps, cost_jps = jump_point_search(grid, (0, 0), (4, 4))
tests.append(("jps_found", path_jps is not None, True))

# Beam Search
path_beam, cost_beam = beam_search((0, 0), (4, 4), grid_neighbors, manhattan, beam_width=5)
tests.append(("beam_found", path_beam is not None, True))

# Hill Climbing (simple example: maximize sum of binary array)
import random
random.seed(42)

def random_binary():
    return [random.randint(0, 1) for _ in range(10)]

def flip_neighbors(arr):
    result = []
    for i in range(len(arr)):
        new_arr = arr[:]
        new_arr[i] = 1 - new_arr[i]
        result.append(new_arr)
    return result

def count_ones(arr):
    return sum(arr)

solution, score = hill_climbing_restarts(random_binary, flip_neighbors, count_ones, max_restarts=5)
tests.append(("hill_climb", score, 10))  # Should find all 1s

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
