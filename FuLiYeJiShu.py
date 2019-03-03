from typing import List, Callable, Dict
from matplotlib import pyplot as plt
import numpy as np
import math

# 好像还有点问题 还没做好


class FuLiYeJiShu:

    def __init__(self, y0: List[float], n=0):
        """
        离散傅里叶级数展开，当x0给定为2*pi*j/N, j=0, 1, ..., N-1时，计算其傅立叶级数展开：
        Sn(x) = a0/2 + sigma{k=1}{n}(a_k * cos kx + b_k * sin kx), where n is not so big (less than N/2)
        :param n: int, 最大展开级数
        :param y0: List[float], the observed data
        """
        if n == 0:
            n = len(y0) // 2 - 1
        m = len(y0)
        self.n = n
        self.m = m
        self.a = [None for _ in range(n+1)]  # type: List[float]
        self.b = [None for _ in range(n+1)]  # type: List[float]

        self.a[0] = 2 / m * sum([y0[j] * math.cos(2*math.pi*j*0/m) for j in range(m)])

        for k in range(1, n+1):
            self.a[k] = 2 / m * sum([y0[j] * math.cos(2*math.pi*j*k/m) for j in range(m)])
            self.b[k] = 2 / m * sum([y0[j] * math.sin(2*math.pi*j*k/m) for j in range(m)])


    def get_value(self, x: float) -> float:
        """
        计算傅里叶级数函数的对应函数值
        :param x: float 计算插值结果
        :return: float 对应函数值
        """
        return self.a[0] / 2 + sum([self.a[k] * math.cos(k*x) + self.b[k] * math.sin(k*x) for k in range(1, self.n+1)])

    def get_values(self, xs: List[float]) -> List[float]:
        """
        计算一串点的函数的对应函数值
        :param xs: list, 要计算的点的列表
        :return: list, 结果
        """
        results = [self.get_value(x) for x in xs]
        return results


if __name__ == "__main__":
    def real_f(x):
        return x*x*math.cos(x)
        # return x + x*x + x*x*x + 2
    x_sample = list(np.linspace(0, 2*math.pi, 5))
    y_sample = [real_f(xi) for xi in x_sample]

    test1 = FuLiYeJiShu(y_sample[:-1])

    t_v = list(np.linspace(-1, 1, 101))
    r_v1 = [round(test1.get_value(i), 6) for i in t_v]
    t2_v = list(np.linspace(-1, 1, 1001))

    print('type 1', r_v1)
    print('real  ', [round(1/(1+25*i*i), 6) for i in t_v])
    # print('系数', test1.eval_equation_factor())
    # print('max_error', test1.estimate_maximum_error(real_f))
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(t_v, r_v1, '-', t2_v, [real_f(i) for i in t2_v], ':')
    ax.legend(['type 1', 'real'], loc='best')

    plt.show()
