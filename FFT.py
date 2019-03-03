from typing import List
import math
import cmath
from matplotlib import pyplot as plt
import numpy as np

pi = math.pi


class FFT:

    def __init__(self, y0: List[complex]):
        """
        an improved way to calc Sigma{0}{n-1} (x_k * omega^(kj)) for j = 0 ,..., N-1
        :param y0: the already known data list[complex]
        """
        self.a = None  # type: List[float]  # used in sanjiao duoxiangshi chazhi
        self.b = None  # type: List[float]  # used in sanjiao duoxiangshi chazhi

        if len(y0) == 0:
            raise IndexError

        p = round(math.log2(len(y0)))
        N = len(y0)
        if pow(2, p) != len(y0):
            raise NotImplementedError

        self.omega = [cmath.exp(1j*2*pi/N*k) for k in range(N//2)]  # type: List[complex]
        self.A = [None for _ in range(3)]  # type: List[List[complex]]
        self.A[0] = [y0[k] for k in range(len(y0))]
        self.A[1] = [0 for _ in range(N)]  
        self.A[2] = [0 for _ in range(N)]

        self.A[1] = y0.copy()
        for q in range(1, p+1):
            if q % 2:
                # step 5
                for k in range(pow(2, p-q)):
                    for j in range(pow(2, q-1)):
                        self.A[2][k*pow(2, q) + j] = self.A[1][k*pow(2, q-1)+j] + self.A[1][k*pow(2, q-1) + j + pow(2, p-1)]
                        self.A[2][k*pow(2, q) + j + pow(2, q-1)] = self.omega[k*pow(2, q-1)] * (self.A[1][k*pow(2, q-1)+j] -
                                                                                                self.A[1][k*pow(2, q-1) + j + pow(2, p-1)])
            else:
                # step 6
                for k in range(pow(2, p-q)):
                    for j in range(pow(2, q-1)):
                        self.A[1][k*pow(2, q) + j] = self.A[2][k*pow(2, q-1)+j] + self.A[2][k*pow(2, q-1) + j + pow(2, p-1)]
                        self.A[1][k*pow(2, q) + j + pow(2, q-1)] = self.omega[k*pow(2, q-1)] * (self.A[2][k*pow(2, q-1)+j] -
                                                                                                self.A[2][k*pow(2, q-1) + j + pow(2, p-1)])
        
        # step 8
        if p % 2 == 0:
            self.A[2] = self.A[1].copy()

        self.c = self.A[2].copy()

    def get_ck(self) -> List[complex]:
        """
        返回计算出来的c_k的值...从而想干啥就干啥
        :return: ck
        """
        return self.c

    def sanjiao_chazhi_get_value(self, x: float) -> float:
        """
        计算三角多项式的值， 当原来的给定区域是[-pi, pi]时...否则需要修改a, b的计算方法
        :param x: float, should be in [-pi, pi]
        :return: result
        """
        if not (-pi) <= x <= pi:
            raise ValueError("x should be in the range of [-pi, pi]")
        if not isinstance(self.a, list):
            c = self.c
            self.a = [(c[k] * cmath.exp(-1j * pi * k)).real / len(c) * 2 for k in range(len(c))][:len(c)//2+1]
            self.b = [(c[k] * cmath.exp(-1j * pi * k)).imag / len(c) * 2 for k in range(len(c))][:len(c)//2+1]
        a = self.a
        b = self.b

        s = a[0] / 2
        for i in range(1, len(a)):
            s += a[i] * math.cos(i * x)
            s += b[i] * math.sin(i * x)
        return s

    def sanjiao_chazhi_get_values(self, xs: List[float]) -> List[float]:
        """
        计算三角多项式的值， 当原来的给定区域是[-pi, pi]时...否则需要修改a, b的计算方法
        :param xs: should be in the range of [-pi, pi] for each x, else will raise ValueError
        :return: result
        """
        results = [self.sanjiao_chazhi_get_value(x) for x in xs]
        return results


if __name__ == "__main__":
    def real_f(x):
        # return x*x*x*x - 3*x*x*x + 2*x*x - math.tan((x) * (x-2))
        # return x + x*x + x*x*x + 2
        return x*x*math.cos(x)

    x_sample = list(np.linspace(-pi, pi, 32+1))
    x_sample = x_sample[:-1]
    y_sample = [real_f(xi) for xi in x_sample]
    # print(y_sample)
    test1 = FFT(y_sample)
    c = test1.get_ck()
    print('c', c)
    # exit(4)
    # r_v1 = [round(test1.get_value(i), 6) for i in t_v]
    real_v = list(np.linspace(-pi, pi, 1001))
    sanjiao = test1.sanjiao_chazhi_get_values(real_v)

    print('a', test1.a)
    print('b', test1.b)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    # ax.plot(t_v, r_v1, '-', t2_v, [real_f(i) for i in t2_v], ':')
    ax.plot(real_v, sanjiao, '-',
            real_v, [real_f(i) for i in real_v], ':')
    # ax.set_ylim([-30, 30])
    ax.legend(['sanjiao', 'real'], loc='best')

    # ax.legend(['type 1', 'real'], loc='best')

    plt.show()
