# ULTRA: Advanced Computational Geometry II

import math
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])

# ULTRA: Line Intersection
def line_intersection(p1, p2, p3, p4):
    """Find intersection of lines (p1-p2) and (p3-p4)."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None  # Parallel or coincident

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)

    return (x, y)

# ULTRA: Segment Intersection Check
def segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def on_segment(p, q, r):
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)

    if d1 != d2 and d3 != d4:
        return True

    # Collinear cases
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    if abs(cross(p1, p2, p3)) < 1e-10 and on_segment(p1, p3, p2):
        return True
    if abs(cross(p1, p2, p4)) < 1e-10 and on_segment(p1, p4, p2):
        return True
    if abs(cross(p3, p4, p1)) < 1e-10 and on_segment(p3, p1, p4):
        return True
    if abs(cross(p3, p4, p2)) < 1e-10 and on_segment(p3, p2, p4):
        return True

    return False

# ULTRA: Point in Convex Polygon (Binary Search)
def point_in_convex_polygon(polygon, point):
    """Check if point is inside convex polygon (O(log n))."""
    n = len(polygon)
    if n < 3:
        return False

    # Assume polygon[0] is a corner, check which triangle sector
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Check if point is on correct side of first and last edge
    if cross(polygon[0], polygon[1], point) < 0:
        return False
    if cross(polygon[0], polygon[-1], point) > 0:
        return False

    # Binary search for the sector
    lo, hi = 1, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if cross(polygon[0], polygon[mid], point) >= 0:
            lo = mid
        else:
            hi = mid

    # Check if point is inside triangle
    return cross(polygon[lo], polygon[hi], point) >= 0

# ULTRA: Closest Pair of Points (Divide and Conquer)
def closest_pair(points):
    """Find closest pair of points in O(n log n)."""
    def dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def closest_pair_rec(px, py):
        n = len(px)
        if n <= 3:
            min_dist = float('inf')
            for i in range(n):
                for j in range(i + 1, n):
                    d = dist(px[i], px[j])
                    if d < min_dist:
                        min_dist = d
            return min_dist

        mid = n // 2
        mid_x = px[mid][0]

        # Split
        pyl = [p for p in py if p[0] <= mid_x]
        pyr = [p for p in py if p[0] > mid_x]

        dl = closest_pair_rec(px[:mid], pyl)
        dr = closest_pair_rec(px[mid:], pyr)
        d = min(dl, dr)

        # Strip
        strip = [p for p in py if abs(p[0] - mid_x) < d]

        for i in range(len(strip)):
            j = i + 1
            while j < len(strip) and strip[j][1] - strip[i][1] < d:
                d = min(d, dist(strip[i], strip[j]))
                j += 1

        return d

    if len(points) < 2:
        return float('inf')

    px = sorted(points, key=lambda p: p[0])
    py = sorted(points, key=lambda p: p[1])

    return closest_pair_rec(px, py)

# ULTRA: Rotating Calipers (Convex Hull Diameter)
def convex_hull_diameter(hull):
    """Find diameter of convex hull using rotating calipers."""
    n = len(hull)
    if n < 2:
        return 0
    if n == 2:
        return math.sqrt((hull[0][0] - hull[1][0])**2 + (hull[0][1] - hull[1][1])**2)

    def dist_sq(p1, p2):
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    k = 1
    while abs(cross(hull[0], hull[1], hull[(k + 1) % n])) > \
          abs(cross(hull[0], hull[1], hull[k])):
        k += 1

    max_dist = 0
    j = k
    for i in range(k + 1):
        max_dist = max(max_dist, dist_sq(hull[i], hull[j]))
        while abs(cross(hull[i], hull[(i + 1) % n], hull[(j + 1) % n])) > \
              abs(cross(hull[i], hull[(i + 1) % n], hull[j])):
            j = (j + 1) % n
            max_dist = max(max_dist, dist_sq(hull[i], hull[j]))

    return math.sqrt(max_dist)

# ULTRA: Minkowski Sum of Two Convex Polygons
def minkowski_sum(P, Q):
    """Compute Minkowski sum of two convex polygons."""
    def reorder(polygon):
        # Start from bottom-leftmost point
        n = len(polygon)
        start = min(range(n), key=lambda i: (polygon[i][1], polygon[i][0]))
        return polygon[start:] + polygon[:start]

    def edge_vectors(polygon):
        n = len(polygon)
        return [(polygon[(i + 1) % n][0] - polygon[i][0],
                 polygon[(i + 1) % n][1] - polygon[i][1])
                for i in range(n)]

    def angle(v):
        return math.atan2(v[1], v[0])

    P = reorder(P)
    Q = reorder(Q)

    edges_p = edge_vectors(P)
    edges_q = edge_vectors(Q)

    # Merge edges by angle
    result = [(P[0][0] + Q[0][0], P[0][1] + Q[0][1])]

    i, j = 0, 0
    while i < len(edges_p) or j < len(edges_q):
        if i >= len(edges_p):
            edge = edges_q[j]
            j += 1
        elif j >= len(edges_q):
            edge = edges_p[i]
            i += 1
        elif angle(edges_p[i]) < angle(edges_q[j]):
            edge = edges_p[i]
            i += 1
        else:
            edge = edges_q[j]
            j += 1

        last = result[-1]
        result.append((last[0] + edge[0], last[1] + edge[1]))

    result.pop()  # Remove duplicate of first point
    return result

# ULTRA: Half-Plane Intersection
def half_plane_intersection(lines):
    """Find intersection of half-planes (each line defines a half-plane to its left)."""
    # Each line is ((x1, y1), (x2, y2))
    if not lines:
        return []

    INF = 1e9

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def line_intersect(l1, l2):
        return line_intersection(l1[0], l1[1], l2[0], l2[1])

    def on_left(line, point):
        return cross(line[0], line[1], point) > 0

    # Sort by angle
    def angle(line):
        dx = line[1][0] - line[0][0]
        dy = line[1][1] - line[0][1]
        return math.atan2(dy, dx)

    lines = sorted(lines, key=angle)

    # Incremental half-plane intersection
    dq = []  # deque of lines
    points = []  # deque of intersection points

    for line in lines:
        while len(points) > 0 and not on_left(line, points[-1]):
            dq.pop()
            points.pop()
        while len(points) > 0 and not on_left(line, points[0]):
            dq.pop(0)
            points.pop(0)

        if dq:
            p = line_intersect(dq[-1], line)
            if p:
                points.append(p)

        dq.append(line)

    # Check last with first
    while len(points) > 1 and not on_left(dq[0], points[-1]):
        dq.pop()
        points.pop()

    if len(dq) >= 2:
        p = line_intersect(dq[-1], dq[0])
        if p:
            points.append(p)

    return points

# Tests
tests = []

# Line intersection
p = line_intersection((0, 0), (1, 1), (0, 1), (1, 0))
tests.append(("line_inter", (round(p[0], 2), round(p[1], 2)) if p else None, (0.5, 0.5)))

# Segment intersection
tests.append(("seg_inter_yes", segments_intersect((0, 0), (1, 1), (0, 1), (1, 0)), True))
tests.append(("seg_inter_no", segments_intersect((0, 0), (1, 1), (2, 2), (3, 3)), False))

# Point in convex polygon
square = [(0, 0), (2, 0), (2, 2), (0, 2)]
tests.append(("in_convex_yes", point_in_convex_polygon(square, (1, 1)), True))
tests.append(("in_convex_no", point_in_convex_polygon(square, (3, 3)), False))

# Closest pair
pts = [(0, 0), (1, 0), (2, 0), (3, 0), (1.1, 0.1)]
cp = closest_pair(pts)
tests.append(("closest_pair", round(cp, 2), 0.14))

# Convex hull diameter
hull = [(0, 0), (3, 0), (3, 4), (0, 4)]
diam = convex_hull_diameter(hull)
tests.append(("hull_diam", round(diam, 1), 5.0))

# Minkowski sum
P = [(0, 0), (1, 0), (1, 1), (0, 1)]
Q = [(0, 0), (1, 0), (0, 1)]
msum = minkowski_sum(P, Q)
tests.append(("minkowski_len", len(msum), 6))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
