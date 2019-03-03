from typing import List, Callable, Dict, Union
from matplotlib import pyplot as plt
import numpy as np
import math

class ShuZhiJiFen:

    def __init__(self, a: float, b: float, f: Callable[[float], float]):
        if a > b:
            self.a = b
            self.b = a
            self.f = lambda x: -f(x)
        else:
            self.a = a
            self.b = b
            self.f = f

    def fu_he_ti_xing_ji_fen(self, h=None) -> float:
        """
        :param h: step length
        :return:
        """
        if h is None:
            h = min(1e-5, (self.b-self.a)/1e5)
        result = 0
        if self.a != float('-inf') and self.b != float('inf'):
            start = self.a
            now = start
            end = self.b

            while now + h <= end:
                st = h * (self.f(now) + self.f(now + h)) / 2
                if st != float('nan'):
                    pass
                elif (h-1e-10) * (self.f(now+1e-10) + self.f(now + h)) / 2 != float('nan'):
                    st = (h-1e-10) * (self.f(now+1e-10) + self.f(now + h)) / 2
                else:
                    st = (h + 1e-10) * (self.f(now) + self.f(now + h + 1e-10)) / 2
                result += st
                now += h

            if now + h < end:
                result += (end - now) * (self.f(now) + self.f(end)) / 2

        else:
            raise NotImplementedError
        return result

    def fu_he_simpson_ji_fen(self, h=None) -> float:
        """

        :param h: step length
        :return:
        """
        if h is None:
            h = min(1e-5, (self.b-self.a)/1e5)
        result = 0
        if self.a != float('-inf') and self.b != float('inf'):
            start = self.a
            now = start
            end = self.b

            while now + h <= end:
                st = h * (self.f(now) + self.f(now + h) + self.f(now + h/2) * 4) / 6
                result += st
                now += h

            if now + h < end:
                result += (end - now) * (self.f(now) + self.f(end) + self.f((now+end) / 2) * 4) / 6

        else:
            raise NotImplementedError
        return result

    def long_bei_ge_ji_fen(self, epsilon=None) -> float:
        """

        :param epsilon: expected wu cha
        :return:
        """

        if epsilon is None:
            epsilon = 1e-10
        k = 0
        h = self.b - self.a
        f = self.f
        T = [[0.0 for _ in range(100)] for _ in range(100)]
        k += 1
        T[0][0] = (f(self.b) + f(self.a)) * h / 2
        T[0][1] = T[0][0] / 2 + h / 2 * sum([self.f(j / pow(2, 1) * (self.b - self.a) + self.a) for j in range(1, int(pow(2, 1)), 2)])
        
        T[1][0] = pow(4, 1) / (pow(4, 1) - 1) * T[0][1] - 1 / (pow(4, 1) - 1) * T[0][0]

        k = 1
        while abs(T[k][0] - T[k-1][0]) >= epsilon:
            k += 1
            h = (self.b - self.a) / pow(2, k-1)
            T[0][k] = T[0][k-1] / 2 + h / 2 * sum([self.f(j / pow(2, k) * (self.b - self.a) + self.a) for j in range(1, int(pow(2, k)), 2)])
            for m in range(1, k+1):
                T[m][k-m] = pow(4, m) / (pow(4, m) - 1) * T[m-1][k-m+1] - 1 / (pow(4, m) - 1) * T[m-1][k-m]

        return T[k][0]

    def zi_shi_ying_ji_fen(self, epsilon=None):
        """

        :param epsilon: maximum expected error
        :return:
        """
        if epsilon is None:
            epsilon = 1e-10

        def simpson(start: float, end: float, func: Callable[[float], float]) -> float:
            return (end-start) / 6 * (func(start) + 4 * func((start+end)/2) + func(end))


        a = self.a
        b = self.b
        f = self.f

        def calc_split(a: float, b: float, f: Callable[[float], float], epsi: float) -> float:

            S1 = simpson(a, b, f)
            S2 = simpson(a, (b+a)/2, f) + simpson((b+a)/2, b, f)
            if (S1 - S2) <= epsi:
                return S2 + (S2 - S1) / 15
            else:
                return calc_split(a, (a+b)/2, f, epsi/2) + calc_split((a+b)/2, b, f, epsi/2)

        return calc_split(a, b, f, epsilon * 15)


class ErWeiShuZhiJiFen:

    def __init__(self, a: float, b: float, c: Union[float, Callable[[float], float]], d: Union[float, Callable[[float], float]], f: Callable[[float, float], float]):
        """
        二维积分，积分形式: intergrate(a~b) intergrate(c~d) f(x, y) dy dx
        :param a:
        :param b:
        :param c:
        :param d:
        :param f:
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.f = f

    def fu_he_er_wei_simpson(self, h=None, k=None):
        if h is None:
            h = min(1e-5, (self.b-self.a)/1e5)

        if (not isinstance(self.c, (float, int))) or (not isinstance(self.d, (float, int))):
            raise NotImplementedError

        y0 = self.c
        ym = self.d
        a = self.a
        b = self.b
        y_half = [y0 + k * (i+0.5) for i in range(0, int((ym-y0) / k))]
        ys = [y0 + k * i for i in range(1, int((ym-y0) / k))]
        result = 0

        def g(_y):
            return lambda x: self.f(x, _y)
        result += ShuZhiJiFen(a, b, g(y0)).fu_he_simpson_ji_fen(h) + ShuZhiJiFen(a, b, g(ym)).fu_he_simpson_ji_fen(h)
        for y in y_half:
            result += 4 * ShuZhiJiFen(a, b, g(y)).fu_he_simpson_ji_fen(h)
        for y in ys:
            result += 2 * ShuZhiJiFen(a, b, g(y)).fu_he_simpson_ji_fen(h)

        result *= k / 6

        return result

    def gauss_er_wei_ji_fen(self, n=None):
        """
        Gauss-Legendre 求积公式，小数位精度1e-7
        :param n: 代数精度2n+1次
        :return:
        """
        if n is None:
            n = 5
        if n not in range(6):
            raise NotImplementedError
        if (not isinstance(self.c, (float, int))) or (not isinstance(self.d, (float, int))):
            raise NotImplementedError
        xs = [[0],
              [0.5773503, -0.5773503],
              [0.7745967, -0.7745967, 0],
              [0.8611363, -0.8611363, 0.3399810, -0.3399810],
              [0.9061798, -0.9061798, 0.5384693, -0.5384693, 0],
              [0.9324695, -0.9324695, 0.6612904, -0.6612904, 0.2386192, -0.2386192]
              ]
        As = [[2],
              [1, 1],
              [5/9, 5/9, 8/9, 8/9],
              [0.3478548, 0.3478548, 0.6521452, 0.6521452],
              [0.2369269, 0.2369269, 0.4786287, 0.4786287, 0.568889],
              [0.1713245, 0.1713245, 0.3607616, 0.3607616, 0.4679139, 0.4679139]
              ]
        a = self.a
        b = self.b
        c = self.c
        d = self.d

        def g(x, y):
            # 变换积分上下限
            return self.f((x+1)/2*(d-c)+c, (y+1)/2*(b-a)+a) * (b-a) * (d-c) / 4

        xk = xs[n]
        Ak = As[n]
        result = 0
        for i in range(len(xk)):
            for j in range(len(xk)):
                result += Ak[i] * Ak[j] * g(xk[i], xk[j])

        return result


    def fei_ju_xing_fu_he_simpson_ji_fen(self, h=None):
        """
        for integration like integrate(a~b) integrate(c(x)~d(x)) f(x, y) dy dx
        :param h: step length for integration between a~b using 复合辛普森公式
        :return:
        """
        c = self.c
        d = self.d
        if isinstance(c, (float, int)):
            c = lambda x: c
        if isinstance(d, (float, int)):
            d = lambda x: d
        if h is None:
            h = min(1e-5, (self.b-self.a)/1e5)
        c = self.c
        d = self.d
        f = self.f
        k = lambda x: (d(x) - c(x)) / 2
        simp_f = lambda x: k(x) / 3 * (f(x, c(x)) + 4 * f(x, c(x) + k(x)) + f(x, d(x)))
        calc = ShuZhiJiFen(self.a, self.b, simp_f)
        return calc.fu_he_simpson_ji_fen(h)


if __name__ == "__main__":
    def real_f(x):
        if x == 0:
            return 0
        return pow(x, 0.5) * math.log(x)
        # return x + x*x + x*x*x + 2
        # return pow(x, 1.5)

    def real_f2(x, y):
        return pow(math.e, -x*y)

    test1 = ShuZhiJiFen(0, 1, real_f)
    print('复合梯形  ', test1.fu_he_ti_xing_ji_fen(1e-5))
    print('复合辛普森', test1.fu_he_simpson_ji_fen(1e-4))
    print('龙贝格    ',  test1.long_bei_ge_ji_fen(1e-8))
    print('自适应    ', test1.zi_shi_ying_ji_fen(1e-12))

    test2 = ErWeiShuZhiJiFen(0, 1, 0, 1, real_f2)
    print('复合辛普森二维', test2.fu_he_er_wei_simpson(0.25, 0.25))
    print('高斯二维积分  ', test2.gauss_er_wei_ji_fen(4))

    def c(x):
        return 0.0

    def d(x):
        return pow(1 - pow(x, 2), 0.5)
    test3 = ErWeiShuZhiJiFen(0, 1, c, d, real_f2)
    print('复合辛普森圆内', test3.fei_ju_xing_fu_he_simpson_ji_fen(0.25))

