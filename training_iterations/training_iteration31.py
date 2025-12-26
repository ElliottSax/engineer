def minimum_genetic_mutation(start, end, bank):
    """Minimum mutations from start to end gene."""
    from collections import deque
    bank_set = set(bank)
    if end not in bank_set:
        return -1
    genes = 'ACGT'
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        gene, mutations = queue.popleft()
        if gene == end:
            return mutations
        for i in range(len(gene)):
            for g in genes:
                if g != gene[i]:
                    new_gene = gene[:i] + g + gene[i+1:]
                    if new_gene in bank_set and new_gene not in visited:
                        visited.add(new_gene)
                        queue.append((new_gene, mutations + 1))
    return -1

def open_the_lock(deadends, target):
    """Minimum moves to open lock from 0000."""
    from collections import deque
    dead = set(deadends)
    if '0000' in dead:
        return -1
    queue = deque([('0000', 0)])
    visited = {'0000'}
    while queue:
        state, moves = queue.popleft()
        if state == target:
            return moves
        for i in range(4):
            digit = int(state[i])
            for d in [-1, 1]:
                new_digit = (digit + d) % 10
                new_state = state[:i] + str(new_digit) + state[i+1:]
                if new_state not in dead and new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, moves + 1))
    return -1

def sliding_puzzle(board):
    """Minimum moves to solve sliding puzzle."""
    from collections import deque
    target = "123450"
    start = ''.join(str(n) for row in board for n in row)
    if start == target:
        return 0
    moves = {0: [1, 3], 1: [0, 2, 4], 2: [1, 5],
             3: [0, 4], 4: [1, 3, 5], 5: [2, 4]}
    queue = deque([(start, start.index('0'), 0)])
    visited = {start}
    while queue:
        state, zero_pos, steps = queue.popleft()
        for new_pos in moves[zero_pos]:
            state_list = list(state)
            state_list[zero_pos], state_list[new_pos] = state_list[new_pos], state_list[zero_pos]
            new_state = ''.join(state_list)
            if new_state == target:
                return steps + 1
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, new_pos, steps + 1))
    return -1

def minimum_jumps(forbidden, a, b, x):
    """Minimum jumps to reach position x."""
    from collections import deque
    forbidden_set = set(forbidden)
    max_pos = max(max(forbidden) + a + b if forbidden else 0, x) + b
    queue = deque([(0, 0, False)])  # position, jumps, can_go_back
    visited = {(0, False)}
    while queue:
        pos, jumps, went_back = queue.popleft()
        if pos == x:
            return jumps
        # Forward jump
        forward = pos + a
        if forward <= max_pos and forward not in forbidden_set and (forward, False) not in visited:
            visited.add((forward, False))
            queue.append((forward, jumps + 1, False))
        # Backward jump (only if didn't just go back)
        if not went_back:
            backward = pos - b
            if backward > 0 and backward not in forbidden_set and (backward, True) not in visited:
                visited.add((backward, True))
                queue.append((backward, jumps + 1, True))
    return -1

def cut_off_trees(forest):
    """Minimum steps to cut all trees in order of height."""
    from collections import deque
    if not forest or forest[0][0] == 0:
        return -1

    m, n = len(forest), len(forest[0])
    trees = sorted((h, r, c) for r, row in enumerate(forest) for c, h in enumerate(row) if h > 1)

    def bfs(sr, sc, tr, tc):
        if sr == tr and sc == tc:
            return 0
        queue = deque([(sr, sc, 0)])
        visited = {(sr, sc)}
        while queue:
            r, c, dist = queue.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and (nr, nc) not in visited and forest[nr][nc] != 0:
                    if nr == tr and nc == tc:
                        return dist + 1
                    visited.add((nr, nc))
                    queue.append((nr, nc, dist + 1))
        return -1

    total = 0
    sr, sc = 0, 0
    for _, tr, tc in trees:
        steps = bfs(sr, sc, tr, tc)
        if steps == -1:
            return -1
        total += steps
        sr, sc = tr, tc
    return total

def shortest_path_alternating_colors(n, red_edges, blue_edges):
    """Shortest path with alternating colors."""
    from collections import defaultdict, deque
    graph = {'red': defaultdict(list), 'blue': defaultdict(list)}
    for u, v in red_edges:
        graph['red'][u].append(v)
    for u, v in blue_edges:
        graph['blue'][u].append(v)

    result = [-1] * n
    # BFS with state (node, last_color)
    queue = deque([(0, None, 0)])  # node, last_color, dist
    visited = {(0, None)}

    while queue:
        node, last_color, dist = queue.popleft()
        if result[node] == -1:
            result[node] = dist
        for color in ['red', 'blue']:
            if color != last_color:
                for neighbor in graph[color][node]:
                    if (neighbor, color) not in visited:
                        visited.add((neighbor, color))
                        queue.append((neighbor, color, dist + 1))
    return result

def min_pushes_to_move_box(grid):
    """Minimum pushes to move box to target."""
    from collections import deque
    m, n = len(grid), len(grid[0])
    player = box = target = None
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 'S':
                player = (i, j)
            elif grid[i][j] == 'B':
                box = (i, j)
            elif grid[i][j] == 'T':
                target = (i, j)

    def can_reach(start, end, box_pos):
        if start == end:
            return True
        queue = deque([start])
        visited = {start, box_pos}
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and (nr, nc) not in visited and grid[nr][nc] != '#':
                    if (nr, nc) == end:
                        return True
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return False

    # BFS with state (box_pos, player_pos, pushes)
    queue = deque([(box, player, 0)])
    visited = {(box, player)}

    while queue:
        (br, bc), (pr, pc), pushes = queue.popleft()
        if (br, bc) == target:
            return pushes
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            # New box position
            nbr, nbc = br + dr, bc + dc
            # Player needs to be on opposite side to push
            player_pos = (br - dr, bc - dc)
            if (0 <= nbr < m and 0 <= nbc < n and grid[nbr][nbc] != '#' and
                0 <= player_pos[0] < m and 0 <= player_pos[1] < n and
                grid[player_pos[0]][player_pos[1]] != '#'):
                if can_reach((pr, pc), player_pos, (br, bc)):
                    new_state = ((nbr, nbc), (br, bc))
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append(((nbr, nbc), (br, bc), pushes + 1))
    return -1

# Tests
tests = [
    ("mutation", minimum_genetic_mutation("AACCGGTT", "AAACGGTA", ["AACCGGTA","AACCGCTA","AAACGGTA"]), 2),
    ("mutation_no", minimum_genetic_mutation("AAAAACCC", "AACCCCCC", ["AAAACCCC","AAACCCCC","AACCCCCC"]), 3),
    ("lock", open_the_lock(["0201","0101","0102","1212","2002"], "0202"), 6),
    ("lock_dead", open_the_lock(["8888"], "0009"), 1),
    ("puzzle", sliding_puzzle([[1,2,3],[4,0,5]]), 1),
    ("puzzle_2", sliding_puzzle([[4,1,2],[5,0,3]]), 5),
    ("jump", minimum_jumps([14,4,18,1,15], 3, 15, 9), 3),
    ("cut_trees", cut_off_trees([[1,2,3],[0,0,4],[7,6,5]]), 6),
    ("alt_colors", shortest_path_alternating_colors(3, [[0,1],[1,2]], []), [0,1,-1]),
]

# Box pushing test
grid = [["#","#","#","#","#","#"],
        ["#","T","#","#","#","#"],
        ["#",".",".","B",".","#"],
        ["#",".","#","#",".","#"],
        ["#",".",".",".","S","#"],
        ["#","#","#","#","#","#"]]
tests.append(("box_push", min_pushes_to_move_box(grid), 3))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
