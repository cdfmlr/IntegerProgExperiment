import numpy as np
from .solver import LpSolver


class LpProblem(object):
    """
    LpProblem 描述一个线性规划问题

    :attribute c: 目标函数系数,  <math> c = (c_1, c_2, ..., c_n)^T </math>
    :attribute a: 系数矩阵,     <math> a = (a_{ij})_{m \times n} = (p_1, p_2, ..., p_n) </math>
    :attribute b: 右端常数,     <math> b = (b_1, b_2, ..., b_m)^T, b > 0 </math>
    :attribute base_idx: 基变量的下标集合
    """

    def __init__(self, c, a, b):
        """
        :param c: 目标函数系数,  <math> c = (c_1, c_2, ..., c_n)^T </math>
        :param a: 系数矩阵,     <math> a = (a_{ij})_{m \times n} = (p_1, p_2, ..., p_n) </math>
        :param b: 右端常数,     <math> b = (b_1, b_2, ..., b_m)^T, b > 0 </math>
        """
        self.c = np.array(c, 'float64')
        self.a = np.array(a, 'float64')
        self.b = np.array(b, 'float64')

    def solve(self, solver: type, **kwargs):
        """
        调用指定算法，求解线性规划问题

        :param solver: 指定算法(LpSolver实例)
        """
        assert issubclass(solver, LpSolver)
        s = solver(self)
        return s.solve(**kwargs)
