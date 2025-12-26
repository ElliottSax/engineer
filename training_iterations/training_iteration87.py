# EXTREME: Computational Geometry

from math import sqrt, atan2, pi
from functools import cmp_to_key

# HARD: Convex Hull (Graham Scan)
def convex_hull(points):
    """Graham scan for convex hull."""
    if len(points) < 3:
        return points[:]

    # Find bottom-most point
    points = sorted(points, key=lambda p: (p[1], p[0]))
    start = points[0]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def polar_angle(p):
        return atan2(p[1] - start[1], p[0] - start[0])

    def dist(p):
        return (p[0] - start[0])**2 + (p[1] - start[1])**2

    # Sort by polar angle
    rest = sorted(points[1:], key=lambda p: (polar_angle(p), -dist(p)))

    hull = [start]
    for p in rest:
        while len(hull) > 1 and cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)

    return hull

# HARD: Closest Pair of Points
def closest_pair(points):
    """Find minimum distance between any two points."""
    def dist(p1, p2):
        return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def closest_split(px, py, delta):
        mid_x = px[len(px) // 2][0]
        sy = [p for p in py if abs(p[0] - mid_x) < delta]
        best = delta
        for i in range(len(sy)):
            for j in range(i + 1, min(i + 7, len(sy))):
                d = dist(sy[i], sy[j])
                if d < best:
                    best = d
        return best

    def closest_rec(px, py):
        n = len(px)
        if n <= 3:
            best = float('inf')
            for i in range(n):
                for j in range(i + 1, n):
                    best = min(best, dist(px[i], px[j]))
            return best

        mid = n // 2
        lx = px[:mid]
        rx = px[mid:]
        mid_x = px[mid][0]
        ly = [p for p in py if p[0] <= mid_x]
        ry = [p for p in py if p[0] > mid_x]

        dl = closest_rec(lx, ly)
        dr = closest_rec(rx, ry)
        delta = min(dl, dr)

        return closest_split(px, py, delta)

    px = sorted(points, key=lambda p: p[0])
    py = sorted(points, key=lambda p: p[1])
    return closest_rec(px, py)

# HARD: Line Segment Intersection
def segments_intersect(p1, p2, p3, p4):
    """Check if line segments (p1,p2) and (p3,p4) intersect."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(p, q, r):
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # Collinear
        return 1 if val > 0 else 2  # Clockwise or counterclockwise

    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, p3, p2):
        return True
    if o2 == 0 and on_segment(p1, p4, p2):
        return True
    if o3 == 0 and on_segment(p3, p1, p4):
        return True
    if o4 == 0 and on_segment(p3, p2, p4):
        return True

    return False

# HARD: Point in Polygon
def point_in_polygon(point, polygon):
    """Check if point is inside polygon using ray casting."""
    x, y = point
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside

# HARD: Polygon Area
def polygon_area(vertices):
    """Calculate area of polygon using shoelace formula."""
    n = len(vertices)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2

# HARD: Rotating Calipers - Diameter of Convex Hull
def diameter_convex_hull(points):
    """Find maximum distance between any two points on convex hull."""
    hull = convex_hull(points)
    n = len(hull)
    if n < 2:
        return 0
    if n == 2:
        return sqrt((hull[0][0] - hull[1][0])**2 + (hull[0][1] - hull[1][1])**2)

    def dist_sq(p1, p2):
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

    k = 1
    max_dist = 0
    for i in range(n):
        while True:
            curr = dist_sq(hull[i], hull[k])
            next_k = (k + 1) % n
            next_dist = dist_sq(hull[i], hull[next_k])
            if next_dist > curr:
                k = next_k
            else:
                break
        max_dist = max(max_dist, dist_sq(hull[i], hull[k]))

    return sqrt(max_dist)

# HARD: Circle-Line Intersection
def circle_line_intersection(cx, cy, r, x1, y1, x2, y2):
    """Find intersection points of circle and line."""
    dx = x2 - x1
    dy = y2 - y1
    fx = x1 - cx
    fy = y1 - cy

    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r

    discriminant = b * b - 4 * a * c

    if discriminant < 0:
        return []

    discriminant = sqrt(discriminant)
    t1 = (-b - discriminant) / (2 * a)
    t2 = (-b + discriminant) / (2 * a)

    points = []
    if 0 <= t1 <= 1:
        points.append((x1 + t1 * dx, y1 + t1 * dy))
    if 0 <= t2 <= 1 and abs(t1 - t2) > 1e-9:
        points.append((x1 + t2 * dx, y1 + t2 * dy))

    return points

# Tests
tests = []

# Convex Hull
hull = convex_hull([(0, 0), (1, 1), (2, 2), (0, 2), (2, 0), (1, 0)])
tests.append(("hull_len", len(hull), 4))

# Closest Pair
tests.append(("closest", round(closest_pair([(0,0), (1,1), (2,2), (3,3), (1,0)]), 2), 1.0))

# Segment Intersection
tests.append(("seg_yes", segments_intersect((0,0), (4,4), (0,4), (4,0)), True))
tests.append(("seg_no", segments_intersect((0,0), (1,1), (2,2), (3,3)), False))
tests.append(("seg_touch", segments_intersect((0,0), (2,2), (1,1), (3,0)), True))

# Point in Polygon
square = [(0,0), (4,0), (4,4), (0,4)]
tests.append(("pip_in", point_in_polygon((2,2), square), True))
tests.append(("pip_out", point_in_polygon((5,5), square), False))

# Polygon Area
tests.append(("area_square", polygon_area([(0,0), (4,0), (4,4), (0,4)]), 16.0))
tests.append(("area_tri", polygon_area([(0,0), (4,0), (2,3)]), 6.0))

# Diameter
tests.append(("diameter", round(diameter_convex_hull([(0,0), (3,0), (3,4), (0,4)]), 2), 5.0))

# Circle-Line Intersection
pts = circle_line_intersection(0, 0, 5, -10, 0, 10, 0)
tests.append(("circle_line", len(pts), 2))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
