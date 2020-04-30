import copy

import numpy as np

from .problem import LpProblem
from .solve import LpSolve
from .solver import LpSolver


class SimplexMethod(LpSolver):
    """ 单纯形(表)法

    待解决应符合标准型，即：
    <math>
        max z = c^T * x
        s.t. a*x = b, x >= 0, b > 0
    </math>

    单纯形算法参考：https://zh.wikipedia.org/zh-hans/单纯形法
    """

    class Problem(LpProblem):
        """
        单纯形(表)法内部的线性规划问题表示
        """
        def __init__(self, c, a, b):
            super().__init__(c, a, b)
            self.base_idx = np.ones(len(b), 'int') * -1
            self.entering_idx = -1
            self.leaving_idx = -1
            self.theta = []
            self.tab = []
            self.cc = copy.deepcopy(c)

    def __init__(self, problem: LpProblem):
        super().__init__(problem)
        self.problem = self.Problem(problem.c, problem.a, problem.b)
        self.pb = None

    def find_idt_base(self):
        """
        尝试找单位阵作初始基

        :return: True if success False else
        """
        base_idx = np.ones(len(self.problem.b), 'int') * -1
        aT = self.problem.a.T
        for i in range(len(self.problem.b)):
            e = np.zeros(len(self.problem.b))
            e[i] = 1
            for j in range(len(aT)):
                if np.all(aT[j] == e):
                    base_idx[i] = j

        self.problem.base_idx = base_idx
        return np.all(base_idx >= 0)

    def big_m(self, **kwargs):
        """
        用大M法得到初始基

        :param kwargs: show_tab=True (default False): 打印运算过程
        :return: None
        """
        M = ((max(abs(self.problem.c)) + 1) ** 2) * 10 + 10
        if kwargs.get("show_tab", False):
            print(f"大M法\n\nM = {M}\n")
        for i in range(len(self.problem.base_idx)):
            if self.problem.base_idx[i] < 0:
                self.problem.c = np.insert(self.problem.c, len(self.problem.c), np.array([-M]))

                ap = np.zeros(len(self.problem.b))
                ap[i] = 1
                self.problem.a = np.c_[self.problem.a, ap]

                self.problem.base_idx[i] = len(self.problem.c) - 1

    def two_step(self, **kwargs):
        """
        用两阶段法得到初始基，第一阶段在此计算

        :param kwargs: show_tab=True (default False): 打印单纯形表
        :return: 第一阶段的解
        """
        p = copy.deepcopy(self.problem)
        p.c = np.zeros(len(p.c))
        for i in range(len(self.problem.base_idx)):
            if self.problem.base_idx[i] < 0:
                p.c = np.insert(p.c, len(p.c), np.array([-1]))

                ap = np.zeros(len(p.b))
                ap[i] = 1
                p.a = np.c_[p.a, ap]

                self.problem.base_idx[i] = len(p.c) - 1
        p.base_idx = self.problem.base_idx
        s1 = _simplex_solve(p, tab=kwargs.get("show_tab", False))
        if kwargs.get("show_tab", False):
            print("两阶段法\n\nStep1:\n")
            print(_simplex_tab_tostring(p))
            print("Step2:\n")
        self.problem.c = copy.deepcopy(self.problem.cc)
        self.problem.base_idx = p.base_idx
        self.problem.b = p.b
        self.problem.a = (p.a.T[0: len(self.problem.c)]).T
        return s1

    def solve(self, **kwargs) -> LpSolve:
        """
        单纯形算法入口

        :param kwargs:
            base_idx=[...] (default []): 指定初始基，缺省则由算法自行确定
            show_tab=True  (default False): 打印单纯形表
            two_step=True  (default False): 使用两阶段法
            big_m=True     (default True):  使用大 M 法
        :return: 问题的解
        """
        base_idx = kwargs.get("base_idx", [])
        if base_idx:    # 用户指定了基
            self.problem.base_idx = base_idx
        else:
            if not self.find_idt_base():    # 没有找到单位阵作初始基，用人工变量法（大M法 / 两阶段法）
                if kwargs.get("two_step", False):
                    s1 = self.two_step(**kwargs)
                    if not s1.success:  # 第一阶段确定无可行解
                        return s1
                else:
                    self.big_m(**kwargs)

        s = None
        self.pb = copy.deepcopy(self.problem)
        if kwargs.get("show_tab", False):
            if s is None:  # For nothing, just referent outer s
                s = _simplex_solve(self.pb, tab=True)
            print(_simplex_tab_tostring(self.pb))
        else:
            if s is None:
                s = _simplex_solve(self.pb, tab=False)
        return s


def _simplex_solve(p: SimplexMethod.Problem, tab=False) -> LpSolve:
    """ simplex_solve 对给定了初始基的标准型使用<em>单纯形表</em>进行求解

    可选参数：
        tab=True (default False): 计算单纯形表

    :return: 问题的解
    """

    # 初始单纯形表的检验数计算
    for i in range(len(p.base_idx)):
        p.c -= p.c[p.base_idx[i]] * p.a[i]

    if tab:
        _current_simplex_tab(p)

    p.entering_idx = np.argmax(p.c)  # 确定入基变量
    while p.c[p.entering_idx] > 0:
        p.theta = []
        for i in range(len(p.b)):
            if p.a[i][p.entering_idx] > 0:
                p.theta.append(p.b[i] / p.a[i][p.entering_idx])
            else:
                p.theta.append(float("inf"))

        p.leaving_idx = np.argmin(np.array(p.theta))  # 确定出基变量

        if p.theta[p.leaving_idx] == float("inf"):  # 出基变量 == inf
            return LpSolve(False, "无界解", [None], None)

        _pivot(p)

        if tab:
            _current_simplex_tab(p)
        p.entering_idx = np.argmax(p.c)     # Next 入基变量

    # 迭代结束，分析解的情况
    x = np.zeros(len(p.c))
    x[p.base_idx] = p.b

    x_real = x[0: len(p.cc)]
    x_presonal = x[len(p.cc):]    # 人工变量

    if np.any(x_presonal != 0):
        return LpSolve(False, "无可行解", None, None)

    z = np.dot(x_real, p.cc)

    for i in range(len(p.c)):
        if (i not in p.base_idx) and (abs(p.c[i]) < 1e-8):  # 非基变量检验数为0
            return LpSolve(True, "无穷多最优解", x_real, z)

    return LpSolve(True, "唯一最优解", x_real, z)


def _pivot(p: SimplexMethod.Problem) -> None:
    """
    对给定问题原址执行转轴操作（基变换）
    """
    main_element = p.a[p.leaving_idx][p.entering_idx]

    p.a[p.leaving_idx] /= main_element
    p.b[p.leaving_idx] /= main_element

    p.base_idx[p.leaving_idx] = p.entering_idx

    for i in range(len(p.b)):
        if i != p.leaving_idx and p.a[i][p.entering_idx] != 0:
            p.b[i] -= p.a[i][p.entering_idx] * p.b[p.leaving_idx]
            p.a[i] -= p.a[i][p.entering_idx] * p.a[p.leaving_idx]

    p.c -= p.c[p.entering_idx] * p.a[p.leaving_idx]


def _current_simplex_tab(p: SimplexMethod.Problem) -> None:
    """
    计算当前单纯形表
    :return: None
    """
    if len(p.tab) > 0:
        main_element = '%.2f' % _float_round4print(p.tab[-1][p.leaving_idx][p.entering_idx + 2])
        p.tab[-1][p.leaving_idx][p.entering_idx + 2] = f'[{main_element}]'

    tab = []
    for i in range(len(p.b)):
        if len(p.theta) > 0:
            p.tab[-1][i][-1] = p.theta[i]

        tab += [
            [f'x_{p.base_idx[i]}', p.b[i]] + list(p.a[i]) + [" ", ]
        ]

    tab += [
        (" ", " ") + tuple(p.c) + (" ",),
    ]

    p.tab.append(tab)


def _simplex_tab_tostring(p: SimplexMethod.Problem, step=None):
    """
    将给定 SimplexMethod.Problem 中的单纯形表转化为字符串

    :param p: SimplexMethod.Problem
    :param step: None 则转化全部，或 int 只转化第几步
    :return: 转化后的字符串
    """
    s = ''
    if step is None:
        for step in p.tab:
            for row in step:
                for i in row:
                    s += '%6.6s\t' % _float_round4print(i)
                s += '\n'
            s += '-' * 16 + '\n'
    else:
        for row in p.tab[step]:
            for i in row:
                s += '%6.6s\t' % _float_round4print(i)
                s += '\n'
            s += '-' * 16 + '\n'
    return s


def _float_round4print(f):
    """
    为方便打印，格式化绝对值趋于0或无穷的浮点值
    :param f: 任意值，但只有 float 才会被操作
    :return: int(0): 若 f 是 float 且趋于 0；
             格式化的指数表示字符串: 若 f 是 float 且 f >= 1E6；
             f: 其他情况
    """
    if isinstance(f, float):
        if abs(f) <= 1E-6:
            return 0
        elif abs(f) >= 1E6:
            return '%2.2e' % f
    return f
