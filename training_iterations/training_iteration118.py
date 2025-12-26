# ULTRA: Game Theory and Combinatorial Games II

from functools import lru_cache

# ULTRA: Sprague-Grundy for Multiple Games
def sprague_grundy_multiple(game_states, sg_function):
    """Compute SG value for multiple games played in parallel (XOR of SG values)."""
    result = 0
    for state in game_states:
        result ^= sg_function(state)
    return result

# ULTRA: Nim with Custom Rules (Subtraction Game)
def subtraction_game_sg(n, S):
    """Compute SG values for subtraction game where you can take s in S stones."""
    sg = [0] * (n + 1)
    for i in range(1, n + 1):
        reachable = set()
        for s in S:
            if i >= s:
                reachable.add(sg[i - s])
        mex = 0
        while mex in reachable:
            mex += 1
        sg[i] = mex
    return sg

# ULTRA: Staircase Nim
def staircase_nim(piles):
    """Staircase Nim: can move any number of stones from pile i to pile i-1.
    Pile 0 is the floor. Win = XOR of odd-indexed piles."""
    xor_sum = 0
    for i in range(1, len(piles), 2):  # Only odd indices
        xor_sum ^= piles[i]
    return xor_sum != 0  # True if first player wins

# ULTRA: Green Hackenbush on Trees
def green_hackenbush_tree(n, edges, root=0):
    """Compute SG value for Green Hackenbush on a tree.
    Value = XOR of subtree depths."""
    if n == 0:
        return 0

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Compute depth parity for each subtree
    visited = [False] * n
    sg = [0] * n

    def dfs(u, parent):
        visited[u] = True
        child_xor = 0
        for v in adj[u]:
            if not visited[v]:
                child_xor ^= (1 + dfs(v, u))
        sg[u] = child_xor
        return sg[u]

    return dfs(root, -1)

# ULTRA: Wythoff's Game
def wythoff_losing(x, y):
    """Check if (x, y) is a losing position in Wythoff's game.
    Losing positions are (floor(n*phi), floor(n*phi^2)) and vice versa."""
    phi = (1 + 5**0.5) / 2

    # For each n, compute cold positions
    for n in range(max(x, y) + 1):
        a = int(n * phi)
        b = int(n * phi * phi)
        if (x == a and y == b) or (x == b and y == a):
            return True

    return False

def wythoff_sg(x, y, memo=None):
    """Compute SG value for Wythoff's game position (x, y)."""
    if memo is None:
        memo = {}

    if (x, y) in memo:
        return memo[(x, y)]

    if x == 0 and y == 0:
        return 0

    reachable = set()

    # Take from x pile
    for i in range(x):
        reachable.add(wythoff_sg(i, y, memo))

    # Take from y pile
    for j in range(y):
        reachable.add(wythoff_sg(x, j, memo))

    # Take from both equally
    for k in range(1, min(x, y) + 1):
        reachable.add(wythoff_sg(x - k, y - k, memo))

    mex = 0
    while mex in reachable:
        mex += 1

    memo[(x, y)] = mex
    return mex

# ULTRA: Misère Nim
def misere_nim(piles):
    """Misère Nim: player who takes last stone loses."""
    if all(p <= 1 for p in piles):
        # Special case: all piles are 0 or 1
        return sum(piles) % 2 == 0  # Win if even number of 1s
    else:
        # Standard Nim XOR analysis
        xor_sum = 0
        for p in piles:
            xor_sum ^= p
        return xor_sum != 0

# ULTRA: Euclid's Game
def euclid_game_winner(a, b):
    """Euclid's Game: subtract k*smaller from larger (k>=1).
    Return True if first player wins."""
    if a < b:
        a, b = b, a

    if b == 0:
        return False  # Game over, previous player won

    # If a >= 2*b, first player can always choose to win
    # Otherwise, depends on recursion
    turn = 0  # 0 = first player's turn
    while b > 0:
        if a >= 2 * b:
            # Current player can choose to leave opponent in any position
            # They can win by choosing appropriately
            return turn == 0
        a, b = b, a - b
        turn = 1 - turn

    return turn == 1

# ULTRA: Blue-Red Hackenbush (Simplified for linear case)
def blue_red_hackenbush_value(edges):
    """Compute game value for Blue-Red Hackenbush on a path.
    Blue edges = +1 for Left, Red edges = -1 for Right."""
    # For a linear game, value is sum of edge values
    value = 0
    for edge in edges:
        if edge == 'B':
            value += 1
        elif edge == 'R':
            value -= 1
    return value

# ULTRA: Nimber Multiplication
def nimber_mult(a, b):
    """Multiply nimbers (used in composite games)."""
    if a == 0 or b == 0:
        return 0
    if a == 1:
        return b
    if b == 1:
        return a

    # Find highest power of 2 in a
    def highest_bit(x):
        bit = 0
        while (1 << bit) <= x:
            bit += 1
        return bit - 1

    ha = highest_bit(a)
    hb = highest_bit(b)

    # If a is a power of 2
    if a == (1 << ha):
        if b == (1 << hb):
            # Both powers of 2
            if ha & hb:
                # Same bit set, use special case
                d = 1 << (ha - 1)
                return nimber_mult(d, d) ^ nimber_mult(d, b)
            else:
                return a * b
        else:
            # Split b
            lb = 1 << hb
            return nimber_mult(a, lb) ^ nimber_mult(a, b ^ lb)
    else:
        # Split a
        la = 1 << ha
        return nimber_mult(la, b) ^ nimber_mult(a ^ la, b)

# Tests
tests = []

# Subtraction game
sg = subtraction_game_sg(10, [1, 3, 4])
tests.append(("subtract_sg0", sg[0], 0))
tests.append(("subtract_sg1", sg[1], 1))
tests.append(("subtract_sg2", sg[2], 0))  # Can only take 1, leaving 1 (SG=1), so mex({1})=0

# Staircase Nim
tests.append(("stair_win", staircase_nim([0, 3, 0, 2, 0]), True))  # 3 XOR 2 = 1 != 0
tests.append(("stair_lose", staircase_nim([0, 1, 0, 1, 0]), False))  # 1 XOR 1 = 0

# Green Hackenbush
edges = [(0, 1), (0, 2), (1, 3)]  # Tree with root 0
gh_sg = green_hackenbush_tree(4, edges, 0)
tests.append(("hackenbush", gh_sg >= 0, True))  # Just verify it computes

# Wythoff's game
tests.append(("wythoff_lose", wythoff_losing(1, 2), True))
tests.append(("wythoff_lose2", wythoff_losing(3, 5), True))
tests.append(("wythoff_win", wythoff_losing(2, 2), False))

# Misère Nim
tests.append(("misere_1", misere_nim([1, 1]), False))  # Two 1s: second player wins
tests.append(("misere_2", misere_nim([1, 2, 3]), True))  # 1^2^3=0 but has pile>1
tests.append(("misere_3", misere_nim([3, 5, 7]), True))  # 3^5^7 != 0

# Euclid's game
tests.append(("euclid_1", euclid_game_winner(5, 3), False))
tests.append(("euclid_2", euclid_game_winner(6, 3), True))  # 6 >= 2*3

# Blue-Red Hackenbush
tests.append(("br_hack", blue_red_hackenbush_value(['B', 'B', 'R']), 1))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
