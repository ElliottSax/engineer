import math

def distance(p1, p2):
    """Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def cross_product(o, a, b):
    """Cross product of vectors OA and OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def convex_hull(points):
    """Convex hull using Andrew's monotone chain algorithm."""
    points = sorted(set(map(tuple, points)))
    if len(points) <= 2:
        return points

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]

def polygon_area(vertices):
    """Area of polygon using shoelace formula."""
    n = len(vertices)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2

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

def closest_pair_of_points(points):
    """Find minimum distance between any two points."""
    def closest_pair_rec(pts_x, pts_y):
        n = len(pts_x)
        if n <= 3:
            return min_brute_force(pts_x)

        mid = n // 2
        mid_point = pts_x[mid]

        pts_y_left = [p for p in pts_y if p[0] <= mid_point[0]]
        pts_y_right = [p for p in pts_y if p[0] > mid_point[0]]

        d_left = closest_pair_rec(pts_x[:mid], pts_y_left)
        d_right = closest_pair_rec(pts_x[mid:], pts_y_right)
        d = min(d_left, d_right)

        # Check strip
        strip = [p for p in pts_y if abs(p[0] - mid_point[0]) < d]
        for i in range(len(strip)):
            for j in range(i + 1, min(i + 7, len(strip))):
                d = min(d, distance(strip[i], strip[j]))
        return d

    def min_brute_force(pts):
        min_dist = float('inf')
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                min_dist = min(min_dist, distance(pts[i], pts[j]))
        return min_dist

    pts_x = sorted(points, key=lambda p: p[0])
    pts_y = sorted(points, key=lambda p: p[1])
    return closest_pair_rec(pts_x, pts_y)

def line_intersection(l1, l2):
    """Find intersection of two lines given by two points each."""
    x1, y1, x2, y2 = l1[0][0], l1[0][1], l1[1][0], l1[1][1]
    x3, y3, x4, y4 = l2[0][0], l2[0][1], l2[1][0], l2[1][1]

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # Parallel

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    return (x, y)

def segments_intersect(s1, s2):
    """Check if two line segments intersect."""
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    A, B = s1
    C, D = s2
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def point_on_segment(p, seg):
    """Check if point lies on segment."""
    a, b = seg
    cross = (p[1] - a[1]) * (b[0] - a[0]) - (p[0] - a[0]) * (b[1] - a[1])
    if abs(cross) > 1e-9:
        return False
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])

def rotate_point(p, angle, origin=(0, 0)):
    """Rotate point around origin by angle (radians)."""
    ox, oy = origin
    px, py = p
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    qx = ox + cos_a * (px - ox) - sin_a * (py - oy)
    qy = oy + sin_a * (px - ox) + cos_a * (py - oy)
    return (qx, qy)

def triangle_area(p1, p2, p3):
    """Area of triangle given three points."""
    return abs(cross_product(p1, p2, p3)) / 2

def circle_line_intersection(center, radius, p1, p2):
    """Find intersection points of circle and line."""
    cx, cy = center
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    fx, fy = p1[0] - cx, p1[1] - cy

    a = dx*dx + dy*dy
    b = 2 * (fx*dx + fy*dy)
    c = fx*fx + fy*fy - radius*radius

    discriminant = b*b - 4*a*c
    if discriminant < 0:
        return []
    elif discriminant == 0:
        t = -b / (2*a)
        return [(p1[0] + t*dx, p1[1] + t*dy)]
    else:
        t1 = (-b - math.sqrt(discriminant)) / (2*a)
        t2 = (-b + math.sqrt(discriminant)) / (2*a)
        return [(p1[0] + t1*dx, p1[1] + t1*dy), (p1[0] + t2*dx, p1[1] + t2*dy)]

def is_point_in_circle(point, center, radius):
    """Check if point is inside or on circle."""
    return distance(point, center) <= radius + 1e-9

def largest_rectangle_in_histogram(heights):
    """Largest rectangle area in histogram."""
    stack = []
    max_area = 0
    heights = heights + [0]
    for i, h in enumerate(heights):
        start = i
        while stack and stack[-1][1] > h:
            idx, height = stack.pop()
            max_area = max(max_area, height * (i - idx))
            start = idx
        stack.append((start, h))
    return max_area

# Tests
tests = [
    ("distance", round(distance((0, 0), (3, 4)), 5), 5.0),
    ("cross_prod", cross_product((0, 0), (1, 0), (0, 1)), 1),
    ("convex_hull", len(convex_hull([(0,0), (1,1), (2,2), (0,2), (2,0), (1,0)])), 4),
    ("polygon_area", polygon_area([(0,0), (4,0), (4,3), (0,3)]), 12.0),
    ("point_in_poly", point_in_polygon((2, 2), [(0,0), (4,0), (4,4), (0,4)]), True),
    ("point_out_poly", point_in_polygon((5, 2), [(0,0), (4,0), (4,4), (0,4)]), False),
    ("closest_pair", round(closest_pair_of_points([(0,0), (1,1), (3,3), (5,5)]), 5), round(math.sqrt(2), 5)),
    ("line_inter", line_intersection(((0, 0), (2, 2)), ((0, 2), (2, 0))), (1.0, 1.0)),
    ("seg_intersect", segments_intersect(((0, 0), (2, 2)), ((0, 2), (2, 0))), True),
    ("seg_no_inter", segments_intersect(((0, 0), (1, 1)), ((2, 2), (3, 3))), False),
    ("point_on_seg", point_on_segment((1, 1), ((0, 0), (2, 2))), True),
    ("triangle_area", triangle_area((0, 0), (4, 0), (0, 3)), 6.0),
    ("circle_line", len(circle_line_intersection((0, 0), 5, (-10, 0), (10, 0))), 2),
    ("point_in_circle", is_point_in_circle((3, 4), (0, 0), 5), True),
    ("histogram", largest_rectangle_in_histogram([2, 1, 5, 6, 2, 3]), 10),
]

# Rotation test
rotated = rotate_point((1, 0), math.pi / 2)
tests.append(("rotate", (round(rotated[0], 5), round(rotated[1], 5)), (0.0, 1.0)))

passed = 0
for name, result, expected in tests:
    if result == expected:
        passed += 1
        print(f"✅ {name}")
    else:
        print(f"❌ {name}: got {result}, expected {expected}")

print(f"\n{passed}/{len(tests)} tests passed")
