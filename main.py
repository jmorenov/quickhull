import numpy as np
import math
from matplotlib import pyplot as plt


def calculate_distances_between_points(points):
    distances = []
    for i in range(0, len(points)):
        for j in range(i+1, len(points)):
            distances.append({
                'a': points[i],
                'b': points[j],
                'distance': math.dist(points[i], points[j])
            })

    return distances


def calculate_distances_points_to_line(points, line):
    distances = []
    for point in points:
        distances.append({
            'point': point,
            'distance': calculate_distance_point_to_line(point, line)
        })

    return distances


def find_max_distance(distances):
    return max(distances, key=lambda x:x['distance'])


def calculate_line(point_a, point_b):
    m = (point_a[1] - point_b[1])/(point_a[0] - point_b[0])
    n = point_a[1] - m * point_a[0]

    return [m, n]


def draw_points(points, color='black'):
    x, y = points.T
    plt.scatter(x, y, color=color)

    return plt


def draw_line(line, color='red'):
    x = np.linspace(-1, 1, 100)
    y = line[0] * x + line[1]
    plt.plot(x, y, '-r', label='Line', color=color)


def calculate_area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1)
                + x3 * (y1 - y2)) / 2.0)


def is_point_inside_triangle(triangle, point):
    x1 = triangle[0][0]
    y1 = triangle[0][1]
    x2 = triangle[1][0]
    y2 = triangle[1][1]
    x3 = triangle[2][0]
    y3 = triangle[2][1]
    x = point[0]
    y = point[1]

    # Calculate area of triangle ABC
    A = calculate_area(x1, y1, x2, y2, x3, y3)

    # Calculate area of triangle PBC
    A1 = calculate_area(x, y, x2, y2, x3, y3)

    # Calculate area of triangle PAC
    A2 = calculate_area(x1, y1, x, y, x3, y3)

    # Calculate area of triangle PAB
    A3 = calculate_area(x1, y1, x2, y2, x, y)

    if A == A1 + A2 + A3:
        return True
    else:
        return False


def calculate_distance_point_to_line(point, line):
    m = line[0]
    n = line[1]
    x = point[0]
    y = point[1]

    return abs((m*x - y + n) / (math.sqrt(m*m + 1)))


def find_points_outside_triangle(triangle, points):
    return list(filter(lambda point: is_point_inside_triangle(triangle, point) is False, points))


def is_point_right_to_line(a, b, p):
    """
    Cross product
    Where a = line point 1; b = line point 2; c = point to check against.
    If the formula is equal to 0, the points are colinear.
    If the line is horizontal, then this returns true if the point is above the line.
    """
    threshold = 1e-9
    if ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) <= -threshold:
        return True
    else:
        return False


def find_points_right_to_line(a, b, points):
    return list(filter(lambda point: is_point_right_to_line(a, b, point) is True, points))


def quick_hull(points):

    convex_hull = []

    coords_max = find_max_x_y(points)
    coords_min = find_min_x_y(points)

    a = coords_max[0]
    b = coords_min[0]

    s1 = find_points_right_to_line(a, b, points)
    s2 = find_points_right_to_line(b, a, points)

    convex_hull.extend(find_hull(s1, a, b))
    convex_hull.extend(find_hull(s2, b, a))

    draw_points(points)
    draw_points(np.array(convex_hull), color='red')

    xs, ys = zip(*convex_hull)
    plt.plot(xs, ys)
    plt.show()


def draw(a, b, s1, s2):
    if (len(s1) > 0):
        draw_points(np.array(s1), color='green')
    if (len(s2) > 0):
        draw_points(np.array(s2), color='yellow')
    draw_line(np.array(calculate_line(a, b)), color='red')
    plt.show()


def find_hull(points, p, q):
    convex_hull = []

    if len(points) == 0:
        return convex_hull
    else:
        line = calculate_line(p, q)
        max_list_distances_to_line = calculate_distances_points_to_line(points, line)
        max_point_to_line = find_max_distance(max_list_distances_to_line)['point']

        triangle = [p, max_point_to_line, q]
        points = find_points_outside_triangle(triangle, points)

        s1 = find_points_right_to_line(p, max_point_to_line, points)
        s1_1 = find_points_right_to_line(max_point_to_line, p, points)
        s2 = find_points_right_to_line(max_point_to_line, q, points)
        s2_2 = find_points_right_to_line(q, max_point_to_line, points)

        # draw(p, max_point_to_line, s1, s1_1)
        # draw(max_point_to_line, q, s2, s2_2)

        convex_hull.extend([p])
        convex_hull.extend(find_hull(s1, p, max_point_to_line))
        convex_hull.extend([max_point_to_line])

        convex_hull.extend([max_point_to_line])
        convex_hull.extend(find_hull(s2, max_point_to_line, q))
        convex_hull.extend([q])

        return convex_hull


def find_max_x_y(points):
    indexes_max = points.argmax(axis=0)

    return [points[indexes_max[0]], points[indexes_max[1]]]


def find_min_x_y(points):
    indexes_min = points.argmin(axis=0)

    return [points[indexes_min[0]], points[indexes_min[1]]]


def read_points_from_file(file_path):
    with open(file_path) as file:
        point_list = np.array([line.strip().split('\t') for line in file.readlines()], dtype=float)

    return point_list


if __name__ == '__main__':
    cloud_points = read_points_from_file('cloud_points.txt')
    quick_hull(cloud_points)
