import math
import numpy as np
from scipy import optimize

from CuttingPlane.linprog.problem import LpProblem
from CuttingPlane.linprog.simplex_method import SimplexMethod

# epsilon 是算法中判断「零」的临界值，绝对值小于该值的数认为是0
_epsilon = 1e-8


def _is_int(n) -> bool:
    """
    is_int 是判断给定数字 n 是否为整数，
    在判断中 n 小于epsilon的小数部分将被忽略，
    是则返回 True，否则 False

    :param n: 待判断的数字
    :return: True if n is A_ub integer, False else
    """
    return (n - math.floor(n) < _epsilon) or (math.ceil(n) - n < _epsilon)


def cutting_plane(c, A, b):
    """
    cutting_plane 对整数规划问题使用「割平面法」进行*递归*求解。

    Returns
    -------
    :return: {"success": True|False, "x": array([...]), "fun": ...}
                - success: 若求解成功则返回 True，否则 False
                - x: 最优解
                - fun: 最优目标函数值
    """

    # 对松弛问题求解
    s = SimplexMethod(LpProblem(c, A, b))
    r = s.solve(big_m=True)

    # print(s)
    # print(r)
    # print(s.pb.a)

    if not r.success:
        return {"success": False, "x": None, "fun": None}

    x = r.solve
    z = r.target

    if all([_is_int(i) for i in x]):  # 最优解是整数解
        return {"success": True, "x": x, "fun": z}

    # 有非整数变量
    # 找出一个非整数变量的索引
    opt_idx = [i for i, v in enumerate(x) if not _is_int(v)][-1]
    opt_tab_inx = s.pb.base_idx[opt_idx]

    # opt 在最终单纯形表中的一行
    row = s.pb.a[opt_idx]
    bow = x[opt_idx]

    # print(opt_idx, row, bow)

    # 拆分整数和小数
    row_int = []
    row_dec = []
    for r in row:
        row_int.append(math.floor(r))
        row_dec.append(r - math.floor(r))

    # print(row_int, row_dec)

    bow_int = math.floor(bow)
    bow_dec = bow - math.floor(bow)

    # 构造新的条件、问题
    new_con = row_dec + [-1]
    new_A = list(s.problem.a.copy())
    for i in range(len(new_A)):
        new_A[i] = list(new_A[i]) + [0]
    new_A.append(new_con)
    new_B = list(s.problem.b.copy())
    new_B.append(bow_dec)
    new_C = list(s.pb.c) + [0]

    # print(new_A, new_B, new_C)

    # 递归求解新问题
    return cutting_plane(new_C, new_A, new_B)


def _test1():
    c = [3, 4, 1]
    A_ub = [[-1, -6, -2], [-2, 0, 0]]
    b_ub = [-5, -3]
    r = cutting_plane(c, A_ub, b_ub)
    print(r)


def _test2():
    c = [40, 90]
    A_ub = [[9, 7], [7, 20]]
    b_ub = [56, 70]
    r = cutting_plane(c, A_ub, b_ub)
    print(r)


def _test3():
    c = [1, 1]
    A_ub = [[2, 1], [4, 5]]
    b_ub = [6, 20]
    r = cutting_plane(c, A_ub, b_ub)
    print(r)


def _test4():
    c = [1, 1]
    a = [[-1, 1], [3, 1]]
    b = [1, 4]
    r = cutting_plane(c, a, b)
    print(r)


if __name__ == "__main__":
    _test1()  # Failed: WA
    _test2()  # Failed: WA
    _test3()  # Pass
    _test4()  # Pass
