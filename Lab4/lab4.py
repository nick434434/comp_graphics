import numpy as np
import math as m
from typing import List, Tuple
import matplotlib.pyplot as plt


def spline(points: List[Tuple[float, float]], cond_type: str, conditions: List[Tuple[float, float]]=None, cnt=100):
    n = len(points) - 1
    dp = np.zeros((n + 1, 2), dtype=np.float)
    for i, tp in enumerate(points):
        dp[i, :] = tp

    t = np.sqrt((dp[1:, 0] - dp[:-1, 0]) ** 2 + (dp[1:, 1] - dp[:-1, 1]) ** 2)

    a = np.zeros((n + 1, n + 1), dtype=np.float)
    for i in range(1, n):
        a[i, i - 1] = t[i]
        a[i, i + 1] = t[i - 1]
        a[i, i] = 2 * (t[i] + t[i - 1])

    b = np.zeros((n + 1, 2), dtype=np.float)
    for i in range(1, n):
        b[i, :] = 3.0 / (t[i - 1] * t[i]) * \
                  (t[i - 1] ** 2 * (dp[i + 1, :] - dp[i, :]) +
                   t[i] ** 2 * (dp[i, :] - dp[i - 1, :]))

    if cond_type == "fixed":
        b[0, :] = conditions[0]
        b[-1, :] = conditions[1]
        a[0, 0] = 1
        a[-1, -1] = 1
    elif cond_type == "weak":
        b[0, :] = 3.0 / 2 / t[1] * (dp[1, :] - dp[0, :])
        b[-1, :] = 6.0 / t[-1] * (dp[-1, :] - dp[-2, :])
        a[0, 0] = 1
        a[0, 1] = .5
        a[-1, -2] = 2
        a[-1, -1] = 4
    elif cond_type == "cyclic" or cond_type == "acyclic":
        a[0, 0] = 2 * (1 + t[-1] / t[1])
        a[0, 1] = t[-1] / t[1]
        a[0, -2] = 1 if cond_type == "cyclic" else -1
        b[0, :] = 3. * (t[-1] / t[1]**2 * (dp[1, :] - dp[0, :]) - (dp[-2, :] - dp[-1, :]) / t[-1])
        a = a[:-1, :-1]
        b = b[:-1, :]

    p_ = np.linalg.inv(a) @ b
    if cond_type == "cyclic" or cond_type == "acyclic":
        tmp = np.zeros((p_.shape[0] + 1, 2))
        tmp[:-1, :] = p_
        tmp[-1, :] = p_[0, :] * (1 if cond_type == "cyclic" else -1)
        p_ = tmp

    p_list = []
    n_per_iter = cnt // n + 1
    tau = np.linspace(0, 1, n_per_iter).reshape((n_per_iter, 1))
    tau_2 = tau ** 2
    tau_3 = tau_2 * tau
    for i in range(n):
        f_vec = np.hstack([2 * tau_3 - 3 * tau_2 + 1,
                           -2 * tau_3 + 3 * tau_2,
                           tau * (tau - 1) ** 2 * t[i],
                           tau * (tau_2 - tau) * t[i]])
        p_vec = np.vstack([dp[i, :], dp[i + 1, :], p_[i, :], p_[i + 1, :]])
        seg_points = f_vec @ p_vec
        seg_points = np.hstack([seg_points, np.ones((n_per_iter, 1))])
        p_list.append(seg_points)

    return np.vstack(p_list)


def draw_spline(p, t, c=None, transform=None):
    spn = spline(p, t, c)
    if transform is not None:
        spn = spn @ transform
        spn[:, :-1] = np.array([np.array(s[:-1]) / s[-1] for s in list(spn)]).reshape(spn.shape[0], spn.shape[1]-1)

    print(spn)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.grid(True, which='both')
    plt.plot(spn[:, 0], spn[:, 1])
    plt.show()


def rotate_matrix(phi):
    phi *= m.pi / 180
    return np.array([
        [m.cos(phi), m.sin(phi), 0],
        [-m.sin(phi), m.cos(phi), 0],
        [0, 0, 1]
    ])


def shift_matrix(x, y):
    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [x, y, 1]
    ])


if __name__ == "__main__":
    points = [(0, 0),
              (2, 1),
              (4, -3),
              (6, 7)]

    conditions = [(1, 1),
                  (1, 1)]

    draw_spline(points, "fixed", conditions)
    draw_spline(points, "weak")
    draw_spline(points, "cyclic")
    draw_spline(points, "acyclic")

    draw_spline(points, "acyclic", transform=shift_matrix(-2, -3))
    draw_spline(points, "acyclic", transform=rotate_matrix(30))
