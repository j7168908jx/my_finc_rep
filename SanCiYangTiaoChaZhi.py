# coding: utf-8
from typing import List
import numpy as np
from matplotlib import pyplot as plt


class SanCiYangTiaoChaZhi:

    def __init__(self, x: List[float], y: List[float], interpolation_type='3', bd_cond=None):
        """
        三次样条插值

        :param x: [x0, x1, ..., xn]
        :param y: [y0, y1, ..., yn]
        :param interpolation_type: '1'=1阶导数 '2'=2阶导数 '3'=周期
        :param bd_cond: when insert_type = '1' or '2', [左边界导数值，右边界导数值], else just leave it None
        """

        self.x0 = x
        self.n = len(x) - 1
        self.y0 = y
        self.interpolation_type = interpolation_type
        self.bd_cond = bd_cond

        if len(x) != len(y) or len(x) <= 3:
            raise IndexError('x y length bu yi yang or too short')

        self.h = [x[j+1] - x[j] for j in range(self.n)]  # 0...n-1

        self.M = []  # 0...n
        self.miu = [self.h[j-1]/(self.h[j-1]+self.h[j]) for j in range(1, self.n)]  # 1...n-1
        self.lbd = [self.h[j]/(self.h[j-1]+self.h[j]) for j in range(1, self.n)]  # 1...n-1
        self.d = [6 * ((y[j+1] - y[j]) / self.h[j] - (y[j] - y[j-1]) / self.h[j-1]) / (self.h[j] + self.h[j-1]) for j in range(1, self.n)] # 1...n-1

        if interpolation_type == '1' or interpolation_type == '2':
            if interpolation_type == '1':
                self.lbd = [1.0] + self.lbd
                self.d = [6 / self.h[0] * ((y[1] - y[0]) / (x[1] - x[0]) - bd_cond[0])] + self.d + [6 / self.h[self.n - 1] * (bd_cond[1] - (y[self.n] - y[self.n-1]) / (x[self.n] - x[self.n-1]))]
                self.miu = [None] + self.miu + [1.0]
            elif interpolation_type == '2':
                self.lbd = [0.0] + self.lbd
                self.miu = [None] + self.miu + [0.0]
                self.d = [2 * bd_cond[0]] + self.d + [2 * bd_cond[1]]
            A = [[2, self.lbd[0]] + (self.n-1) * [0]]
            for i in range(1, self.n):
                A.append((i-1) * [0] + [self.miu[i], 2, self.lbd[i]] + (self.n-1-i) * [0])
            A.append((self.n-1) * [0] + [self.miu[self.n], 2])
            A = np.array(A)
            b = np.array([[i] for i in self.d])
            print(A)
            self.M = np.linalg.solve(A, b)

        # not a good way to realize type 2
        elif interpolation_type == '20':
            self.lbd = [None] + self.lbd
            self.miu = [None] + self.miu + [None]
            self.d = [None] + self.d + [None]
            A = [[2, self.lbd[1]] + (self.n-3) * [0]]
            for i in range(2, self.n-1):
                A.append((i - 2) * [0] + [self.miu[i], 2, self.lbd[i]] + (self.n - 2 - i) * [0])
            A.append((self.n-3) * [0] + [self.miu[self.n-1], 2])
            A = np.array(A)
            b = [[self.d[1] - self.miu[1] * bd_cond[0]]] + [[self.d[i]] for i in range(2, self.n-1)] + [[self.d[self.n-1] - self.lbd[self.n-1] * bd_cond[1]]]
            b = np.array(b)
            self.M = np.array([[bd_cond[0]]] + list(np.linalg.solve(A, b)) + [[bd_cond[1]]])

        elif interpolation_type == '3':
            self.lbd = [None] + self.lbd + [self.h[0] / (self.h[0] + self.h[self.n-1])]
            self.miu = [None] + self.miu + [1 - self.lbd[self.n]]
            self.d = [None] + self.d + [6 * ((y[1] - y[0]) / self.h[0] - (y[self.n] - y[self.n-1]) / self.h[self.n-1]) / (self.h[0] + self.h[self.n-1])]

            A = [[2, self.lbd[1]] + (self.n-3) * [0] + [self.miu[1]]]
            for i in range(2, self.n):
                A.append((i-2) * [0] + [self.miu[i], 2, self.lbd[i]] + (self.n-1-i) * [0])
            A.append([self.lbd[self.n]] + (self.n-3) * [0] + [self.miu[self.n], 2])
            A = np.array(A)
            b = np.array([self.d[i] for i in range(1, self.n+1)])

            self.M = list(np.linalg.solve(A, b))

            last = self.M[self.n-1]
            self.M = [last] + self.M

    def get_value(self, x: float) -> float:
        """
        计算样条函数的对应函数值
        :param x: float value 计算插值结果
        :return: float value 样条函数的对应函数值
        """
        for j in range(self.n):
            if self.x0[j] < x < self.x0[j+1]:
                return float(self.M[j] * pow((self.x0[j+1] - x), 3) / 6 / self.h[j]
                             + self.M[j+1] * pow((x - self.x0[j]), 3) / 6 / self.h[j]
                             + (self.y0[j] - self.M[j] * pow(self.h[j], 2) / 6) * (self.x0[j+1] - x) / self.h[j]
                             + (self.y0[j+1] - self.M[j+1] * pow(self.h[j], 2) / 6) * (x - self.x0[j]) / self.h[j])
            if self.x0[j] == x:
                return self.y0[j]
        if self.x0[self.n] == x:
            return self.y0[self.n]
        raise RuntimeError

    def get_values(self, xs: List[float]) -> List[float]:
        results = [self.get_value(i) for i in xs]
        return results


if __name__ == "__main__":
    def real_f(x):
        return 1/(1+25*x*x)

    x_sample = list(np.linspace(-1, 1, 11))
    y_sample = [real_f(xi) for xi in x_sample]

    bd_cond1 = [50/26/26, -50/26/26]
    bd_cond2 = [(50*75-50)/pow(26, 3), (50*75-50)/pow(26, 3)]


    test1 = SanCiYangTiaoChaZhi(x_sample, y_sample, interpolation_type='1', bd_cond=bd_cond1)
    test2 = SanCiYangTiaoChaZhi(x_sample, y_sample, interpolation_type='2', bd_cond=bd_cond2)
    test3 = SanCiYangTiaoChaZhi(x_sample, y_sample, interpolation_type='3', bd_cond=None)

    t_v = list(np.linspace(-1, 1, 101))
    r_v1 = [round(test1.get_value(i), 6) for i in t_v]
    r_v2 = [round(test2.get_value(i), 6) for i in t_v]
    r_v3 = [round(test3.get_value(i), 6) for i in t_v]
    t2_v = list(np.linspace(-1, 1, 1001))

    print('type 1', r_v1)
    print('type 2', r_v2)
    print('type_3', r_v3)
    print('real  ', [round(1/(1+25*i*i), 6) for i in t_v])
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(t_v, r_v1, '-', t_v, r_v2, '--', t_v, r_v3, '--', t2_v, [real_f(i) for i in t2_v], ':')
    ax.legend(['type 1', 'type 2', 'type3', 'real'], loc='best')

    plt.show()


