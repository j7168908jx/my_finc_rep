from typing import List, Callable, Dict
from matplotlib import pyplot as plt
import numpy as np


class NCiQuXianNiHe:

    def __init__(self, N: int, x0: List[float], y0: List[float], omega_x: List[float]):
        """
        N次多项式拟合（2-范数最小二乘拟合）（利用正交多项式）
        :param N: 多项式最高次数
        :param x0: list, 拟合点x值
        :param y0: list, 拟合点y值
        :param omega_x: list, 权重，按照拟合点给出，可以设置为[]从而使用默认值
        """
        self.start = x0[0]
        self.end = x0[-1]
        if len(omega_x) == 0:
            omega_x = [1 for _ in range(len(x0))]

        if len(x0) != len(y0):
            raise IndexError("x0 y0 length bu yi yang")

        if len(x0) <= 1:
            raise IndexError("x0 length too small")

        self.N = N
        self.p = [None for _ in range(N+1)]  # type: List[function]
        self.p[0] = lambda x: 1  # p0(x)
        alpha = [None for _ in range(N+1)]  # type: List[float]
        beta = [None for _ in range(N)]  # type: List[float]
        self.a_star = [None for _ in range(N+1)]  # type: List[float]
        
        alpha[1] = (sum([omega_x[i] * x0[i] * pow(self.p[0](x0[i]), 2) for i in range(len(x0))])
                    / sum([omega_x[i] * pow(self.p[0](x0[i]), 2) for i in range(len(x0))]))
        beta[0] = 0
        
        self.p[1] = lambda x: (x - alpha[1]) * self.p[0](x)
        
        self.a_star[0] = (sum([omega_x[i] * y0[i] * self.p[0](x0[i]) for i in range(len(x0))])
                          / sum([omega_x[i] * pow(self.p[0](x0[i]), 2) for i in range(len(x0))]))
        
        for k in range(1, N):

            alpha[k+1] = (sum([omega_x[i] * x0[i] * pow(self.p[k](x0[i]), 2) for i in range(len(x0))])
                          / sum([omega_x[i] * pow(self.p[k](x0[i]), 2) for i in range(len(x0))]))
            beta[k] = (sum([omega_x[i] * pow(self.p[k](x0[i]), 2) for i in range(len(x0))])
                       / sum([omega_x[i] * pow(self.p[k-1](x0[i]), 2) for i in range(len(x0))]))

            self.p[k+1] = lambda x, p_k=self.p[k], p_k_1=self.p[k-1], al = alpha[k+1], be = beta[k]:\
                (x - al) * p_k(x) - be * p_k_1(x)

            self.a_star[k] = (sum([omega_x[i] * y0[i] * self.p[k](x0[i]) for i in range(len(x0))])
                              / sum([omega_x[i] * pow(self.p[k](x0[i]), 2) for i in range(len(x0))]))
            
        self.a_star[N] = (sum([omega_x[i] * y0[i] * self.p[N](x0[i]) for i in range(len(x0))])
                          / sum([omega_x[i] * pow(self.p[N](x0[i]), 2) for i in range(len(x0))]))

    def get_value(self, x: float) -> float:
        """
        计算正交多项式函数的对应函数值
        :param x: float value 计算插值结果
        :return: float value 多项式的对应函数值
        """
        y = 0
        for i in range(self.N+1):
            y += self.a_star[i] * self.p[i](x)
        return y

    def get_values(self, xs: List[float]) -> List[float]:
        """
        计算一串点的正交多项式函数的对应函数值
        :param xs: list, 要计算的点的列表
        :return: list, 结果
        """
        results = [self.get_value(x) for x in xs]
        return results

    def eval_equation_factor(self) -> Dict[str, float]:
        """
        利用线性代数估计多项式各次系数的值
        f(x) = a0 + a1*x + a2*x^2 + ... + an*x^n
        X = [[1^0, 1^1, ..., 1^N],
             [2^0, 2^1, ..., 2^N],
             ...,
             [(N+1)^0, (N+1)^1, ..., (N+1)^N]]
        b = [P(1), P(2), ..., P(N+1)]'
        :return: X \ b => [a0, a1, a2, ..., an]
        """
        ys = self.get_values([i for i in range(1, self.N+2)])
        ys = [[i] for i in ys]
        ys = np.array(ys)
        X = [[pow(k, n) for n in range(self.N+1)] for k in range(1, self.N+2)]
        X = np.array(X)
        a = np.linalg.solve(X, ys)

        return dict([('a' + str(i), round(a[i][0], 3)) for i in range(len(a))])

    def estimate_maximum_error(self, real_function: Callable[[float], float], start=None, end=None, step=1e-5) -> float:
        """
        估计（无穷范数下的）最大误差，需要指定函数
        :param real_function: function, the real function
        :param start: float, the start of the closed interval
        :param end: float, the end of the closed interval
        :param step: float, step, default 1e-5
        :return: float, the maximum error of the two functions in the given interval
        """
        if start is None:
            start = self.start
        if end is None:
            end = self.end
        all_test = [start + i * step for i in range(1+int((end-start) / step))]
        max_error = max([abs(real_function(x) - self.get_value(x)) for x in all_test])
        return max_error


if __name__ == "__main__":
    def real_f(x):
        return 1/(1+25*x*x)
        # return x + x*x + x*x*x + 2
    x_sample = list(np.linspace(-1, 1, 11))
    y_sample = [real_f(xi) for xi in x_sample]

    test1 = NCiQuXianNiHe(3, x_sample, y_sample, [])

    t_v = list(np.linspace(-1, 1, 101))
    r_v1 = [round(test1.get_value(i), 6) for i in t_v]
    t2_v = list(np.linspace(-1, 1, 1001))

    print('type 1', r_v1)
    print('real  ', [round(1/(1+25*i*i), 6) for i in t_v])
    print('系数', test1.eval_equation_factor())
    print('max_error', test1.estimate_maximum_error(real_f))
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(t_v, r_v1, '-', t2_v, [real_f(i) for i in t2_v], ':')
    ax.legend(['type 1', 'real'], loc='best')

    plt.show()
