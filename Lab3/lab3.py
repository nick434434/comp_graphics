import numpy as np
import cv2
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


def draw(points):
    points = 40 * np.array(points)
    print(points)
    minimum = np.min(np.min(points))
    points -= minimum
    points = points.astype(np.int32)
    img = 255 * np.ones((400, 400))
    for i in range(0, len(points) - 1):
        cv2.line(img, (points[i][0], points[i][1]), (points[i+1][0], points[i+1][1]), (0, 255, 0))
    return img


if __name__ == "__main__":
    el = ellipse(4, 1, 40)
    img = draw(el)
    cv2.imshow('Ellipse', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
