# from scipy.spatial import ConvexHull
# from lab1 import Point2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D, art3d
from config import CONFIG
from pyhull.convex_hull import qconvex as qh
import math as m


col = [
    [0, 1, 0, 0.3],
    [1, 0, 0, 0.3],
    [0, 0, 1, 0.3],
    [0, 0.6, 0.6, 0.3],
    [0.6, 0.6, 0, 0.3],
    [0.6, 0, 0.6, 0.3]
]
edge_col = [
    [0, 0.7, 0, 0.8],
    [0.7, 0, 0, 0.8],
    [0, 0, 0.7, 0.8],
    [0, 0.4, 0.4, 0.8],
    [0.4, 0.4, 0, 0.8],
    [0.4, 0, 0.4, 0.8]
]


def draw(list_points, poly, title):
    fig = plt.figure()
    ax = Axes3D(fig)
    xlim = [0, 1]
    ylim = [0, 1]
    zlim = [0, 1]
    for j, points in enumerate(list_points):
        xlim = [min(np.min(points[:, 0]) - 2, xlim[0]), max(xlim[1], np.max(points[:, 0]) + 2)]
        ylim = [min(np.min(points[:, 1]) - 2, ylim[0]), max(ylim[1], np.max(points[:, 1]) + 2)]
        zlim = [min(np.min(points[:, 2]) - 2, zlim[0]), max(zlim[1], np.max(points[:, 2]) + 2)]

        for pp in poly:
            p = art3d.Poly3DCollection([points[pp, :]])
            p.set_color(col[j])
            p.set_edgecolor(edge_col[j])
            ax.add_collection3d(p)
    left = min(xlim[0], ylim[0], zlim[0])
    right = max(xlim[1], ylim[1], zlim[1])
    ax.set_xlim3d(left, right)
    ax.set_ylim3d(left, right)
    ax.set_zlim3d(left, right)
    ax.set_aspect('equal')
    plt.title(title)
    plt.show()


def convex(points):
    hull = qh('i Qi', points)
    poly = []
    for s in hull[1:]:
        s = [int(i) for i in s.split(' ')]
        poly.append(s)
    return poly


def transform(points, T):
    new_points = []

    for point in points:
        new_point = np.matmul(np.array(list(point) + [1]), T)
        new_point = [int(new_point[i] / new_point[3]) for i in range(3)]
        new_points.append(np.array(new_point))

    return np.array(new_points)


def shift_matrix(b, c, d, f, g, h):
    return np.array([np.array([1, b, c, 0]),
                     np.array([d, 1, f, 0]),
                     np.array([g, h, 1, 0]),
                     np.array([0, 0, 0, 1])
                     ])


def rotate_matrix1(theta):
    return np.array([[1, 0, 0, 0], [0, m.cos(theta), m.sin(theta), 0],
                     [0, -m.sin(theta), m.cos(theta), 0], [0, 0, 0, 1]])

def rotate_matrix2(psy):
    return np.array([[m.cos(psy), m.sin(psy), 0, 0], [-m.sin(psy), m.cos(psy), 0, 0],
                     [0, 0, 1, 0], [0, 0, 0, 1]])


def rotate_matrix3(phi):
    return np.array([[m.cos(phi), 0,  -m.sin(phi), 0], [0, 1, 0, 0],
                     [m.sin(phi), 0, m.cos(phi), 0], [0, 0, 0, 1]])


if __name__ == "__main__":
    p = []
    points = 10 * np.array(CONFIG['points1'])
    poly = convex(points)

    T = np.diag([1.0/2, 1.0/3, 1, 1])
    print(T)
    new_points = transform(points, T)
    draw([points, new_points], poly, 'Scaling by axes')

    T = np.diag([1, 1, 1, 0.5])
    print(T)
    new_points = transform(points, T)
    draw([points, new_points], poly, 'General scaling')

    T = shift_matrix(-0.85, 0.25, -0.75, 0.7, 0.5, 1)
    print(T)
    new_points = transform(points, T)
    draw([points, new_points], poly, 'Shifts')

    T = rotate_matrix1(270 * m.pi/180)
    print(T)
    new_points = transform(points, T)
    draw([points, new_points], poly, 'Rotate 1')

    T = rotate_matrix3(90 * m.pi / 180)
    print(T)
    new_points = transform(points, T)
    draw([points, new_points], poly, 'Rotate 3')

    T = np.array([
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ])
    print(T)
    new_points = transform(points, T)
    draw([points, new_points], poly, 'Rotate combined 1')

    T = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ])
    print(T)
    new_points = transform(points, T)
    draw([points, new_points], poly, 'Rotate combined 2')

    T = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    print(T)
    new_points = transform(points, T)
    draw([points, new_points], poly, 'Mirror')

    T = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [10, -3, -3, 1]
    ])
    print(T)
    new_points = transform(points, T)
    draw([points, new_points], poly, 'Transferring')

    T = np.array([
        [1, 0, 0, 0],
        [0, 0.866, 0.5, 0],
        [0, -0.5, 0.866, 0],
        [0, 0.951, -0.549, 1]
    ])
    print(T)
    new_points = transform(points, T)
    draw([points, new_points], poly, 'The only relative rotate')


    points = np.array([
        [2, 1, 2],
        [3, 1, 2],
        [3, 1.5, 2],
        [2.5, 2, 2],
        [2, 2, 2],
        [2, 1, 1],
        [3, 1, 1],
        [3, 2, 1],
        [2, 2, 1],
        [3, 2, 1.5]
    ])
    poly = convex(points)

    T = np.array([
        [2 / m.sqrt(6), 0, 1 / m.sqrt(3), 0],
        [-1 / m.sqrt(6), 1 / m.sqrt(2), 1 / m.sqrt(3), 0],
        [-1 / m.sqrt(6), -1 / m.sqrt(2), 1 / m.sqrt(3), 0],
        [-2 / m.sqrt(6), 0, -4 / m.sqrt(3), 1]
    ])
    print(T)
    new_points = transform(10 * points, T)
    draw([10 * points, new_points], poly, 'Relative rotate 1')

    T = np.array([
        [0.805, -0.311, 0.506, 0],
        [0.506, 0.805, -0.311, 0],
        [-0.311, 0.506, 0.805, 0],
        [0.195, 0.311, -0.506, 1]
    ])
    print(T)
    new_points = transform(10 * points, T)
    draw([10 * points, new_points], poly, 'Relative rotate 2')
