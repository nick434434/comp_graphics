import numpy as np
import cv2
import matplotlib.pyplot as plt
import math as m


def ellipse(a, b, n):
    delta_theta = 2 * m.pi / (n - 1)
    start = [a * m.cos(0), b * m.sin(0)]

    points = [start]

    cos = m.cos(delta_theta)
    sin = m.sin(delta_theta)

    for i in range(1, n):
        xi, yi = points[i-1]
        new_point = [xi*cos - 1.0 * a / b * yi*sin, 1.0 * b / a * xi*sin + yi*cos]
        points.append(new_point)

    return np.array(points)



def parabola(a, begin, end, n):
    th_min = m.sqrt(1.0 * begin / a)
    th_max = m.sqrt(1.0 * end / a)
    delta_theta = (th_max - th_min) * m.pi / (n - 1)
    start = [a * th_min * th_min, 2*a*th_min]

    points = [start]

    for i in range(1, n):
        xi, yi = points[i-1]
        new_point = [xi + delta_theta*(yi + a*delta_theta), yi + 2*a*delta_theta]
        points.append(new_point)

    points_rev = [[p[0], -p[1]] for p in reversed(points)]
    return np.array(points_rev + points)


def hyperbola(a, b, begin, end, n):
    th_min = m.acosh(1.0 * begin / a)
    th_max = m.acosh(1.0 * end / a)
    delta_theta = (th_max - th_min) / (n - 1)
    start = [a * m.cosh(th_min), b * m.sinh(th_min)]

    points = [start]

    cosh = m.cosh(delta_theta)
    sinh = m.sinh(delta_theta)

    for i in range(1, n):
        xi, yi = points[i - 1]
        new_point = [xi*cosh + yi*a/b*sinh, a/b*xi*sinh + yi*cosh]
        points.append(new_point)

    points_rev = [[p[0], -p[1]] for p in reversed(points)]
    points = points_rev + points
    points_rev = [[-p[0], p[1]] for p in reversed(points)]

    return np.array(points + points_rev)


def draw(points, window_name):
    if window_name is 'Ellipse':
        points = 40 * np.array(points)
    else:
        points = 10 * np.array(points)
    minimum = np.min(np.min(points))
    points -= minimum
    points = points.astype(np.int32)
    img = 255*np.ones((400, 400, 3))
    for i in range(0, len(points) - 1):
        if window_name == 'Ellipse' or len(points) / 2 != i+1:
            cv2.line(img, (points[i][0], points[i][1]), (points[i+1][0], points[i+1][1]), (255, 0, 0))
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(window_name)
    plt.show()
    return img


def shift_matrix(x, y):
    return np.array([[1, 0, 0], [0, 1, 0], [-x, -y, 1]])


def rotate_matrix(phi):
    return np.array([[m.cos(phi), -m.sin(phi), 0], [m.sin(phi), m.cos(phi), 0], [0, 0, 1]])


def reflect_matrix(x, y):
    a = np.zeros((3, 3))
    a[0, 0] = x
    a[1, 1] = y
    a[2, 2] = 1
    return a


def transform(points, T):
    points = [[p[0], p[1], 1] for p in list(points)]
    transformed = [[p[0] / p[2], p[1] / p[2]] for p in list(np.matmul(points, T))]
    return np.array(transformed)


if __name__ == "__main__":
    hyp = hyperbola(2, 1, 2.1, 6, 20)
    draw(hyp, 'Hyperbola')
'''
    el = ellipse(1, 4, 40)
    draw(el, 'Ellipse')

    T = rotate_matrix(30)
    draw(transform(el, T), 'Ellipse 1')

    par = parabola(1, 0, 6, 20)
    draw(par, 'Parabola')
'''