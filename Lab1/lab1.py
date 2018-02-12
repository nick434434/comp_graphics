import numpy as np
import cv2
import math as m
from config import CONFIG


class Point(object):
    def __init__(self, tup, mul = False):
        self.x = tup[0]
        self.y = tup[1]
        if mul:
            self.x *= 10
            self.y *= 10
        print self

    def reflection(self, line):
        A1 = line[0]
        A2 = line[1]
        if A1.x == A2.x:
            return Point([])
        k1 = (A2.y - A1.y) * 1.0 / (A2.x - A1.x)
        b1 = A2.y - k1*A2.x
        print "y = " + str(k1) + "x + " + str(b1)
        k2 = -1.0 / k1
        b2 = self.y - k2*self.x
        x_int = (b2 - b1) * 1.0 / (k1 - k2)
        y_int = k2*x_int + b2
        print str(x_int) + " " + str(y_int)
        x_new = x_int + (x_int - self.x)
        y_new = y_int + (y_int - self.y)
        return Point([x_new, y_new])

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.x) + ' ' + str(self.y)


class Triangle:
    def __init__(self, p1, p2, p3):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.line = None

    def add_line(self, p1, p2):
        self.line = [p1, p2]

    def reflection(self, line=None):
        if line is None:
            if self.line is None:
                raise AttributeException
            else:
                line = self.line
        return Triangle(self.p1.reflection(line), self.p2.reflection(line), self.p3.reflection(line))

    def shift(self, x):
        for p in [self.p1, self.p2, self.p3]:
            p.x += x

    def min_x(self):
        min = self.p1.x
        for p in [self.p2, self.p3]:
            if p.x < min:
                min = p.x
        return min

    def draw(self, img, color):
        for (p1, p2) in [(self.p1, self.p2), (self.p2, self.p3), (self.p3, self.p1)]:
            cv2.line(img, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), color, 1)
        return img


class Solver1:
    def __init__(self, trngl1, trngl2, line):
        self.t1 = trngl1
        self.t2 = trngl2
        self.l = line

    def normalize(self):
        x_max = self.t1.p1.x
        x_min = self.t1.p1.x
        y_max = self.t1.p1.y
        y_min = self.t1.p1.y
        for p in [self.t1.p1, self.t1.p2, self.t1.p3, self.t2.p1, self.t2.p2, self.t2.p3]:
            if p.x > x_max:
                x_max = p.x
            elif p.x < x_min:
                x_min = p.x
            if p.y > y_max:
                y_max = p.y
            elif p.y < y_min:
                y_min = p.y

        for p in [self.t1.p1, self.t1.p2, self.t1.p3, self.t2.p1, self.t2.p2, self.t2.p3, self.l[0], self.l[1]]:
            p.x -= x_min - 5
            p.y -= y_min

        return (int(x_max+5), y_max)

    def draw(self, img):
        self.t1.draw(img, (0,0,255))
        self.t2.draw(img, (255,0,0))
        cv2.line(img, (int(self.l[0].x), int(self.l[0].y)), (int(self.l[1].x), int(self.l[1].y)), (0,255,0), 1)
        return img


class Point2(Point):
    def __init__(self, tup, mul=False):
        super(Point2, self).__init__(tup[0:2], mul)
        self.z = tup[2]

    def __repr__(self):
        return [self.x, self.y, self.z]

    def reflection(self, line):
        A1 = line[0]
        A2 = line[1]
        if A1.x == A2.x:

            return Point2([])
        k1 = (A2.y - A1.y) * 1.0 / (A2.x - A1.x)
        b1 = A2.y - k1*A2.x
        print "y = " + str(k1) + "x + " + str(b1)
        k2 = -1.0 / k1
        b2 = self.y - k2*self.x
        x_int = (b2 - b1) * 1.0 / (k1 - k2)
        y_int = k2*x_int + b2
        print str(x_int) + " " + str(y_int)
        x_new = x_int + (x_int - self.x)
        y_new = y_int + (y_int - self.y)
        return Point2([x_new, y_new, self.z])

    def transform(self, T):
        return np.matmul([self.x, self.y, self.z], T)


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


def simulate(t, line):
    if line[0].x == line[1].x:
        return
    else:
        A1 = line[0]
        A2 = line[1]

        k = (A2.y - A1.y) * 1.0 / (A2.x - A1.x)
        b = A2.y - k * A2.x

        T = shift_matrix(0, b)
        R = rotate_matrix(m.atan(k))
        P = reflect_matrix(1, -1)
        T1 = np.linalg.inv(T)
        R1 = np.linalg.inv(R)

        p1 = Point2(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(np.array(t.p1.__repr__()), T), R), P), R1), T1))
        p2 = Point2(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(np.array(t.p2.__repr__()), T), R), P), R1), T1))
        p3 = Point2(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(np.array(t.p3.__repr__()), T), R), P), R1), T1))
        return Triangle(p1, p2, p3)


if __name__ == "__main__":

    sz = int(max(max(abs(CONFIG['A1'][0]), abs(CONFIG['A1'][1])), max(abs(CONFIG['A2'][0]), abs(CONFIG['A2'][1]))))
    img = 255 * np.ones((10*sz,10*sz,3), np.uint8)

    trngl = Triangle(Point(CONFIG['p1'], True), Point(CONFIG['p2'], True), Point(CONFIG['p3'], True))
    trngl.add_line(Point(CONFIG['A1'], True), Point(CONFIG['A2'], True))
    reflected = trngl.reflection()

    solver = Solver1(trngl, reflected, [Point(CONFIG['A1'], True), Point(CONFIG['A2'], True)])
    x, y = solver.normalize()
    img = solver.draw(img)

    img = cv2.flip(img, 0)
    img = cv2.resize(img, (512, 512))
    cv2.imshow('Result1', img)

    p1 = CONFIG['p1']
    p2 = CONFIG['p2']
    p3 = CONFIG['p3']
    p1.append(1)
    p2.append(1)
    p3.append(1)
    A1 = CONFIG['A1']
    A2 = CONFIG['A2']
    A1.append(1)
    A2.append(1)
    img = 255 * np.ones((10 * sz, 10 * sz, 3), np.uint8)
    trngl = Triangle(Point2(p1, True), Point2(p2, True), Point2(p3, True))
    line = [Point2(A1, True), Point2(A2, True)]

    reflected = simulate(trngl, line)

    min1 = reflected.min_x()
    min2 = trngl.min_x()
    mm = min(min1, min2)
    if mm < 0:
        trngl.shift(-mm)
        reflected.shift(-mm)
        line[0].x -= mm
        line[1].x -= mm

    img = trngl.draw(img, (0,0,255))
    img = reflected.draw(img, (255,0,0))
    cv2.line(img, (line[0].x, line[0].y), (line[1].x, line[1].y), (0,255,0))

    img = cv2.flip(img, 0)
    img = cv2.resize(img, (512, 512))
    cv2.imshow("Result2", img)

    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()