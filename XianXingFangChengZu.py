from typing import List, Callable, Dict, Union, Iterable, Tuple, Sized
import numpy as np
import math
import random

class XianXingFangChengZu:

    def __init__(self, A: Union[Iterable, List, np.ndarray], b: Union[Iterable, List, np.ndarray]):

        self.A = np.mat(A, dtype='float64')
        self.b = np.mat(b, dtype='float64')
        self.det = None
        if len(self.A.shape) != 2 or len(self.b.shape) != 2:
            raise NotImplementedError('too high dim')

        if self.b.shape[1] != 1:
            self.b = self.b.T

        if self.b.shape[1] != 1:
            raise ValueError('b is not an vector')

        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError('dimension of A and b are not matched')

        if self.A.shape[0] != self.A.shape[1]:
            raise NotImplementedError

    def lie_zhu_yuan_gao_si_xiao_qu(self) -> np.mat:

        A = self.A.copy()
        b = self.b.copy()
        b = [b[i, 0] for i in range(b.shape[0])]

        n = len(b)

        det = 1
        i_k = [k for k in range(n)]

        for k in range(0, n-1):
            for i in range(k, n):
                if A[i, k] >= A[i_k[k], k]:
                    i_k[k] = i
            if A[i_k[k], k] == 0:
                raise ZeroDeterminantError

            if i_k[k] != k:
                for j in range(k, n):
                    temp = A[k, j]
                    A[k, j] = A[i_k[k], j]
                    A[i_k[k], j] = temp
                temp = b[k]
                b[k] = b[i_k[k]]
                b[i_k[k]] = temp

                det *= -1
            for i in range(k+1, n):
                m = A[i, k] / A[k, k]
                A[i, k] = m
                for j in range(k+1, n):
                    A[i, j] -= m * A[k, j]
                b[i] -= m * b[k]
            det *= A[k, k]

        if A[n-1, n-1] == 0:
            raise ZeroDeterminantError
        b[n-1] = b[n-1] / A[n-1, n-1]
        for i in range(n-2, -1, -1):
            b[i] = (b[i] - sum([A[i, j] * b[j] for j in range(i+1, n)])) / A[i, i]

        det *= A[n-1, n-1]

        self.det = det
        return b


    def lie_zhu_yuan_gao_si_xiao_qu(self) -> np.mat:

        A = self.A.copy()
        b = self.b.copy()
        b = [b[i, 0] for i in range(b.shape[0])]

        n = len(b)

        det = 1
        i_k = [k for k in range(n)]

        for k in range(0, n-1):
            for i in range(k, n):
                if A[i, k] >= A[i_k[k], k]:
                    i_k[k] = i
            if A[i_k[k], k] == 0:
                raise ZeroDeterminantError

            if i_k[k] != k:
                for j in range(k, n):
                    temp = A[k, j]
                    A[k, j] = A[i_k[k], j]
                    A[i_k[k], j] = temp
                temp = b[k]
                b[k] = b[i_k[k]]
                b[i_k[k]] = temp

                det *= -1
            for i in range(k+1, n):
                m = A[i, k] / A[k, k]
                A[i, k] = m
                for j in range(k+1, n):
                    A[i, j] -= m * A[k, j]
                b[i] -= m * b[k]
            det *= A[k, k]

        if A[n-1, n-1] == 0:
            raise ZeroDeterminantError
        b[n-1] = b[n-1] / A[n-1, n-1]
        for i in range(n-2, -1, -1):
            b[i] = (b[i] - sum([A[i, j] * b[j] for j in range(i+1, n)])) / A[i, i]

        det *= A[n-1, n-1]

        self.det = det
        return b


    def quan_xuan_zhu_yuan_gao_si_xiao_qu(self) -> List:

        A = self.A.copy()
        b = self.b.copy()
        b = [b[i, 0] for i in range(b.shape[0])]

        n = len(b)
        j_record = [i for i in range(n)]
        for k in range(n):

            i_mx = k
            j_mx = k
            for i in range(k, n):
                for j in range(k, n):
                    if abs(A[i, j]) > abs(A[i_mx, j_mx]):
                        i_mx = i
                        j_mx = j
            if A[k, k] == 0:
                raise ZeroDeterminantError
            if i_mx != k:
                for i in range(n):
                    tmp = A[k, i]
                    A[k, i] = A[i_mx, i]
                    A[i_mx, i] = tmp

                tmp = b[k]
                b[k] = b[i_mx]
                b[i_mx] = tmp

            if j_mx != k:
                tmp = j_record[k]
                j_record[k] = j_record[j_mx]
                j_record[j_mx] = tmp

                for j in range(n):
                    tmp = A[j, k]
                    A[j, k] = A[j, j_mx]
                    A[j, j_mx] = tmp

            for i in range(k + 1, n):
                for j in range(k + 1, n):
                    A[i, j] -= A[i, k] / A[k, k] * A[k, j]
                b[i] -= A[i, k] / A[k, k] * b[k]
                A[i, k] = 0

        b[n - 1] = b[n - 1] / A[n - 1, n - 1]

        for i in range(n - 2, -1, -1):
            b[i] = (b[i] - sum([A[i, j] * b[j] for j in range(i + 1, n)])) / A[i, i]
        j = 0

        while j < n:
            if j_record[j] != j:
                tmp = b[j]
                b[j] = b[j_record[j]]
                b[j_record[j]] = tmp

                tmp = j_record[j_record[j]]
                j_record[j_record[j]] = j_record[j]
                j_record[j] = tmp

            if j_record[j] == j:
                j += 1
        return b

    def die_dai(self, x0: Union[Iterable, List, np.ndarray], method: str, max_error=1e-10, omega=1):
        """
        die dai fa to solve linear equations
        :param x0: initial value of x
        :param method: in ['jaccobi', 'SOR', 'CG']
        :param max_error: when ||x_k - x_k-1||_inf < max_error, iteration stops.
        :param omega: to be used in SOR
        :return: x_cpt, iteration times
        """
        xs = [x0]
        A = self.A.copy()
        b = self.b.copy()
        n = len(b)

        if method not in ['jaccobi', 'SOR', 'CG']:
            raise NotImplementedError("method must be in ['jaccobi', 'SOR', 'CG']")

        if method == 'CG':
            r = [(self.b - self.A * np.mat(x0).T)]  # type: List[np.matrixlib.defmatrix.matrix]
            p = [(self.b - self.A * np.mat(x0).T)]  # type: List[np.matrixlib.defmatrix.matrix]
            k = 0
            xs = [np.mat(x0).T]
            alpha = []
            beta = []
            while True:
                if np.linalg.norm(r[k], np.inf) <= max_error or float(p[k].T * A * p[k]) <= max_error:
                    break

                alpha.append(np.mat(float(r[k].T * r[k]) / float(p[k].T * self.A * p[k])))
                xs.append(xs[k] + p[k] * alpha[k])
                r.append(r[k] - self.A * p[k] * alpha[k])
                beta.append(np.mat(float(r[k+1].T * r[k+1]) / float(r[k].T * r[k])))
                p.append(r[k+1] + p[k] * beta[k])
                k += 1

            return xs[-1].T.tolist()[0], k

        else:

            if method == 'SOR' and not (0 < omega < 2):
                print('omega not in (0, 2), x might diverge')

            def jaccobi(x):
                x_new = [0 for i in range(n)]
                for i in range(n):
                    x_new[i] = float((b[i] - sum([A[i, j] * x[j] if j != i else 0 for j in range(n)])) / A[i, i])
                return x_new

            def SOR(x, omega=omega):
                x_new = [0 for i in range(n)]
                for i in range(n):
                    x_new[i] = float(x[i] + omega * (b[i] - sum([A[i, j] * x_new[j] for j in range(i)]) - sum(
                        [A[i, j] * x[j] for j in range(i, n)])) / A[i, i])
                return x_new

            xs.append(eval(method)(xs[-1]))

            def error_inf(x1, x2):
                return max([abs(x1[i] - x2[i]) for i in range(len(x1))])

            while error_inf(xs[-1], xs[-2]) >= max_error:
                xs.append(eval(method)(xs[-1]))

            return xs[-1], len(xs)-1



class ZeroDeterminantError(ValueError):

    def __init__(self, err=None):
        if err is None:
            err = "the Determinant is Zero"
        ValueError.__init__(self, err)


if __name__ == "__main__":

    def create_A(n: int) -> List[List[float]]:
        a = [[0 for j in range(n)] for i in range(n)]
        for i in range(len(a)):
            for j in range(len(a)):
                if i > j:
                    a[i][j] = -1
                if i == j:
                    a[i][j] = 1
                if j == len(a) - 1:
                    a[i][j] = 1
        return a

    def create_x(n: int):
        return [random.random() for i in range(n)]

    def Hilbert_matrix(n: int) -> List[List[float]]:
        A = [[1/(i+j-1) for j in range(1, n+1)] for i in range(1, n+1)]
        return A

    def ptest0():

        A = [[10, 7, 8, 7],
             [7, 5, 6, 5],
             [8, 6, 10, 9],
             [7, 5, 9, 10]]

        b = [32, 23, 33, 31]

        dA = [[10, 7, 8.1, 7.2],
              [7.08, 5.04, 6, 5],
              [8, 5.98, 9.89, 9],
              [6.99, 5, 9, 9.98]]

        test1 = XianXingFangChengZu(dA, b)
        print('det(A)', np.linalg.det(A))
        print('A的特征值', np.linalg.eigvals(A))
        print('cond(A)_2', np.linalg.cond(A, 2))
        dx = list(map(lambda x: x - 1, test1.lie_zhu_yuan_gao_si_xiao_qu()))
        print('delta_x', dx)
        print('||delta_x||', np.linalg.norm(dx, 2))
        print('||delta_x|| / ||x||', np.linalg.norm(dx, 2) / np.linalg.norm([1, 1, 1, 1], 2))
        print('||delta_A|| / ||A||', np.linalg.norm(np.mat(dA) - np.mat(A), 2) / np.linalg.norm(np.mat(A), 2))

        # -------------第4题----------------

        def Hilbert_matrix(n: int) -> List[List[float]]:
            A = [[1 / (i + j - 1) for j in range(1, n + 1)] for i in range(1, n + 1)]
            return A

        print('利用高斯消去法求解:')
        for n in range(2, 15):
            H = Hilbert_matrix(n)
            # b = [sum([H[i][j] for j in range(n)]) for i in range(n)]
            x = [1 for i in range(n)]
            b = np.matmul(H, x)
            x_ = XianXingFangChengZu(H, b).lie_zhu_yuan_gao_si_xiao_qu()

            dx = [xx - 1 for xx in x_]
            # print(list(map(lambda x: round(x, 5), dx)))
            max_ = 1 + int(np.log10(0.5 / max([abs(x) for x in dx])))
            print(dx)
            print('cond(H%d)_inf' % n, np.linalg.cond(H, np.inf),
                  '最大剩余误差||r||_inf',
                  max([abs(b[i] - sum([H[i][j] * x_[j] for j in range(len(x_))])) for i in range(len(b))]),
                  '近似解x~正确的位数', max_)
            if max([abs(x) for x in dx]) >= 0.5:
                print('误差大于0.5, n = ', n)
                break

    def ptest1():

        A = create_A(60)
        x = create_x(60)
        b = np.matmul(A, x)

        test = XianXingFangChengZu(A, b)
        x_p = test.lie_zhu_yuan_gao_si_xiao_qu()
        x_c = test.quan_xuan_zhu_yuan_gao_si_xiao_qu()

        nm1 = np.linalg.norm(b - np.matmul(A, x_p), 2)
        nm2 = np.linalg.norm([x[i] - x_p[i] for i in range(len(x))], 2)

        print('1) i)', nm1)
        print('1)ii)', nm2)
        nm3 = np.linalg.norm(b - np.matmul(A, x_c), 2)
        nm4 = np.linalg.norm([x[i] - x_c[i] for i in range(len(x))], 2)

        print('2) i)', nm3)
        print('2)ii)', nm4)

        A = Hilbert_matrix(12)
        x = create_x(12)
        b = np.matmul(A, x)
        test2 = XianXingFangChengZu(A, b)
        x_p = test2.lie_zhu_yuan_gao_si_xiao_qu()
        x_c = test2.quan_xuan_zhu_yuan_gao_si_xiao_qu()

        nm1 = np.linalg.norm(b - np.matmul(A, x_p), 2)
        nm2 = np.linalg.norm([x[i] - x_p[i] for i in range(len(x))], 2)

        print('3)1) i)', nm1)
        print('3)1)ii)', nm2)

        nm3 = np.linalg.norm(b - np.matmul(A, x_c), 2)
        nm4 = np.linalg.norm([x[i] - x_c[i] for i in range(len(x))], 2)

        print('3)2) i)', nm3)
        print('3)2)ii)', nm4)

    def ptest2():
        avg_error = []
        for times in range(1):
            for n in [250, 300, 350]:
                A = create_A(n)
                x = create_x(n)
                b = np.matmul(A, x)

                test = XianXingFangChengZu(A, b)
                x_p = test.lie_zhu_yuan_gao_si_xiao_qu()
                x_c = test.quan_xuan_zhu_yuan_gao_si_xiao_qu()
                print('x_p', x_p)
                print('x_c', x_c)
                nm1 = np.linalg.norm(b - np.matmul(A, x_p), 2)
                nm2 = np.linalg.norm([x[i] - x_p[i] for i in range(len(x))], 2)

                print('1) i)', nm1)
                print('1)ii)', nm2)

                nm3 = np.linalg.norm(b - np.matmul(A, x_c), 2)
                nm4 = np.linalg.norm([x[i] - x_c[i] for i in range(len(x))], 2)

                print('2) i)', nm3)
                print('2)ii)', nm4)

                print(n, '\t', nm1, '\t', nm2, '\t', nm3, '\t', nm4)
                avg_error.append([nm1, nm2, nm3, nm4])
        avg_error = [sum([avg_error[i][j] for i in range(len(avg_error))])/10 for j in range(4)]

        print(avg_error)

    def ptest3():
        avg_error = []
        for times in range(10):
            for n in [12]:
                A = Hilbert_matrix(n)
                x = create_x(n)
                b = np.matmul(A, x)

                test = XianXingFangChengZu(A, b)
                x_p = test.lie_zhu_yuan_gao_si_xiao_qu()
                x_c = test.quan_xuan_zhu_yuan_gao_si_xiao_qu()
                print('x_p', x_p)
                print('x_c', x_c)
                nm1 = np.linalg.norm(b - np.matmul(A, x_p), 2)
                nm2 = np.linalg.norm([x[i] - x_p[i] for i in range(len(x))], 2)

                print('1) i)', nm1)
                print('1)ii)', nm2)

                nm3 = np.linalg.norm(b - np.matmul(A, x_c), 2)
                nm4 = np.linalg.norm([x[i] - x_c[i] for i in range(len(x))], 2)

                print('2) i)', nm3)
                print('2)ii)', nm4)

                print(n, '\t', nm1, '\t', nm2, '\t', nm3, '\t', nm4)
                avg_error.append([nm1, nm2, nm3, nm4])
        avg_error = [sum([avg_error[i][j] for i in range(len(avg_error))])/10 for j in range(4)]

        print(avg_error)

    def ptest4():
        A = [[3, 1], [1, 2]]

        b = [5, 5]

        test = XianXingFangChengZu(A, b)

        print(test.die_dai([0, 0], 'CG'))

    ptest4()