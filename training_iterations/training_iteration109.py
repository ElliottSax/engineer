# ULTRA: Random and Probabilistic Algorithms

import random
from collections import Counter
import heapq

# ULTRA: Randomized Quick Select
def quickselect(arr, k):
    """Find k-th smallest element in O(n) expected time."""
    if not arr or k < 1 or k > len(arr):
        return None

    arr = arr[:]

    def partition(left, right, pivot_idx):
        pivot = arr[pivot_idx]
        arr[pivot_idx], arr[right] = arr[right], arr[pivot_idx]
        store_idx = left
        for i in range(left, right):
            if arr[i] < pivot:
                arr[store_idx], arr[i] = arr[i], arr[store_idx]
                store_idx += 1
        arr[store_idx], arr[right] = arr[right], arr[store_idx]
        return store_idx

    def select(left, right, k):
        if left == right:
            return arr[left]
        pivot_idx = random.randint(left, right)
        pivot_idx = partition(left, right, pivot_idx)
        if k == pivot_idx:
            return arr[k]
        elif k < pivot_idx:
            return select(left, pivot_idx - 1, k)
        else:
            return select(pivot_idx + 1, right, k)

    return select(0, len(arr) - 1, k - 1)

# ULTRA: Median of Medians (Deterministic Linear Selection)
def median_of_medians(arr, k):
    """Deterministic O(n) selection algorithm."""
    if not arr or k < 1 or k > len(arr):
        return None

    def select(items, k):
        if len(items) <= 5:
            return sorted(items)[k - 1]

        # Divide into groups of 5
        chunks = [items[i:i+5] for i in range(0, len(items), 5)]
        medians = [sorted(chunk)[len(chunk) // 2] for chunk in chunks]

        # Recursively find median of medians
        pivot = select(medians, (len(medians) + 1) // 2)

        # Partition
        lows = [x for x in items if x < pivot]
        highs = [x for x in items if x > pivot]
        pivots = [x for x in items if x == pivot]

        if k <= len(lows):
            return select(lows, k)
        elif k <= len(lows) + len(pivots):
            return pivot
        else:
            return select(highs, k - len(lows) - len(pivots))

    return select(arr[:], k)

# ULTRA: Randomized Min-Cut (Karger's Algorithm)
def karger_min_cut(n, edges, iterations=100):
    """Karger's randomized min-cut algorithm."""
    def contract():
        parent = list(range(n))
        rank = [0] * n

        def find(x):
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

        remaining = n
        edge_list = edges[:]
        random.shuffle(edge_list)

        for u, v in edge_list:
            if remaining == 2:
                break
            pu, pv = find(u), find(v)
            if pu != pv:
                union(pu, pv)
                remaining -= 1

        # Count cut edges
        cut_edges = 0
        for u, v in edges:
            if find(u) != find(v):
                cut_edges += 1

        return cut_edges

    min_cut = float('inf')
    for _ in range(iterations):
        min_cut = min(min_cut, contract())

    return min_cut

# ULTRA: Freivald's Algorithm (Matrix Multiplication Verification)
def freivalds_verify(A, B, C, k=20):
    """Verify if A * B = C with probability 1 - 2^(-k)."""
    n = len(A)

    for _ in range(k):
        # Random vector
        r = [random.randint(0, 1) for _ in range(n)]

        # Compute B * r
        Br = [sum(B[i][j] * r[j] for j in range(n)) for i in range(n)]

        # Compute A * (B * r)
        ABr = [sum(A[i][j] * Br[j] for j in range(n)) for i in range(n)]

        # Compute C * r
        Cr = [sum(C[i][j] * r[j] for j in range(n)) for i in range(n)]

        if ABr != Cr:
            return False

    return True

# ULTRA: Monte Carlo Pi Estimation
def monte_carlo_pi(samples):
    """Estimate pi using Monte Carlo method."""
    inside = 0
    for _ in range(samples):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1:
            inside += 1
    return 4 * inside / samples

# ULTRA: Las Vegas Algorithm - Randomized Primality Test
def miller_rabin_las_vegas(n, k=10):
    """Miller-Rabin primality test."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False

    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    def check(a):
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            return True
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                return True
        return False

    # Deterministic witnesses for small n
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for a in witnesses:
        if a >= n:
            continue
        if not check(a):
            return False

    return True

# ULTRA: Simulated Annealing for TSP
def simulated_annealing_tsp(dist, initial_temp=1000, cooling_rate=0.995, min_temp=1):
    """Simulated annealing for TSP."""
    n = len(dist)
    if n <= 1:
        return list(range(n)), 0

    def tour_length(tour):
        return sum(dist[tour[i]][tour[(i + 1) % n]] for i in range(n))

    def swap_2opt(tour, i, j):
        new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
        return new_tour

    # Initial tour
    current_tour = list(range(n))
    random.shuffle(current_tour)
    current_length = tour_length(current_tour)
    best_tour = current_tour[:]
    best_length = current_length

    temp = initial_temp

    while temp > min_temp:
        # Generate neighbor
        i, j = sorted(random.sample(range(n), 2))
        new_tour = swap_2opt(current_tour, i, j)
        new_length = tour_length(new_tour)

        # Accept or reject
        delta = new_length - current_length
        if delta < 0 or random.random() < pow(2.71828, -delta / temp):
            current_tour = new_tour
            current_length = new_length
            if current_length < best_length:
                best_tour = current_tour[:]
                best_length = current_length

        temp *= cooling_rate

    return best_tour, best_length

# ULTRA: Genetic Algorithm for Optimization
def genetic_algorithm_max(fitness_fn, n_bits, pop_size=50, generations=100, mutation_rate=0.01):
    """Simple genetic algorithm for maximization."""
    def random_chromosome():
        return [random.randint(0, 1) for _ in range(n_bits)]

    def crossover(p1, p2):
        point = random.randint(1, n_bits - 1)
        return p1[:point] + p2[point:]

    def mutate(chromosome):
        return [1 - b if random.random() < mutation_rate else b for b in chromosome]

    def select(population, fitness_scores):
        total = sum(fitness_scores)
        if total == 0:
            return random.choice(population)
        r = random.random() * total
        cumsum = 0
        for chrom, score in zip(population, fitness_scores):
            cumsum += score
            if cumsum >= r:
                return chrom
        return population[-1]

    # Initialize population
    population = [random_chromosome() for _ in range(pop_size)]

    for _ in range(generations):
        fitness_scores = [fitness_fn(chrom) for chrom in population]

        new_population = []
        for _ in range(pop_size):
            p1 = select(population, fitness_scores)
            p2 = select(population, fitness_scores)
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    # Return best
    best = max(population, key=fitness_fn)
    return best, fitness_fn(best)

# Tests
tests = []

# Quick Select
random.seed(42)
arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
tests.append(("quickselect_3", quickselect(arr, 3), 2))
tests.append(("quickselect_6", quickselect(arr, 6), 4))

# Median of Medians
tests.append(("mom_3", median_of_medians(arr, 3), 2))
tests.append(("mom_6", median_of_medians(arr, 6), 4))

# Karger Min-Cut
random.seed(42)
edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
tests.append(("karger", karger_min_cut(4, edges, 50) >= 1, True))

# Freivald's
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
C = [[19, 22], [43, 50]]  # Correct
tests.append(("freivalds_correct", freivalds_verify(A, B, C), True))
C_wrong = [[1, 2], [3, 4]]
tests.append(("freivalds_wrong", freivalds_verify(A, B, C_wrong), False))

# Monte Carlo Pi
random.seed(42)
pi_est = monte_carlo_pi(10000)
tests.append(("monte_carlo_pi", 3.0 < pi_est < 3.3, True))

# Miller-Rabin
tests.append(("miller_prime", miller_rabin_las_vegas(104729), True))
tests.append(("miller_composite", miller_rabin_las_vegas(104730), False))

# Simulated Annealing
random.seed(42)
dist_tsp = [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]
_, length = simulated_annealing_tsp(dist_tsp)
tests.append(("sa_tsp", length <= 100, True))  # Reasonable tour

# Genetic Algorithm
random.seed(42)
def onemax(chrom):
    return sum(chrom)
best, score = genetic_algorithm_max(onemax, 10)
tests.append(("genetic", score >= 5, True))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
