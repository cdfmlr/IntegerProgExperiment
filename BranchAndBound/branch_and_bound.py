import math
import numpy as np
from scipy import optimize

from BranchAndBound.bnb_Tree import BnBTree, BnBTreeNode

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


def branch_and_bound(c, A_ub, b_ub, A_eq, b_eq, bounds, bnbTreeNode=None):
    """
    branch_and_bound 对整数规划问题使用「分支定界法」进行*递归*求解。

    底层对松弛问题求解使用 scipy.optimize.linprog 完成，
    该算法只是在 scipy.optimize.linprog 求解的基础上加以整数约束，
    所以求解问题的模型、参数中的 c, A_ub, b_ub, A_eq, b_eq, bounds
    与 scipy.optimize.linprog 的完全相同。

    问题模型：
        Minimize:     c^T * x
    
        Subject to:   A_ub * x <= b_ub
                      A_eq * x == b_eq
                      (x are integers)

    你可以提供一个 BnBTreeNode 实例作为根节点来记录求解过程，得到一个求解过程的树形图。
    如果需要这样的求解过程的树形图，你可以这样调用 branch_and_bound：
        c = [-40, -90]
        A_ub = [[9, 7], [7, 20]]
        b_ub = [56, 70]
        bounds = [(0, None), (0, None)]
        tree = BnBTree()
        r = branch_and_bound(c, A_ub, b_ub, None, None, bounds, tree.root)
        print(r)    # 打印求解结果
        print(tree) # 打印求解过程的树形图

    Parameters
    ----------
    :param c: 系数矩阵。array_like
        Coefficients of the linear objective function to be minimized.
    :param A_ub: 不等式约束条件矩阵，array_like, 若无则需要传入 None
        2-D array which, when matrix-multiplied by ``x``, gives the values of
        the upper-bound inequality constraints at ``x``.
    :param b_ub: 不等式约束条件右端常数，array_like, 若无则需要传入 None
        1-D array of values representing the upper-bound of each inequality
        constraint (row) in ``A_ub``.
    :param A_eq: 等式约束条件矩阵，array_like, 若无则需要传入 None
        2-D array which, when matrix-multiplied by ``x``, gives the values of
        the equality constraints at ``x``.
    :param b_eq: 等式约束条件右端常数，array_like, 若无则需要传入 None
        1-D array of values representing the RHS of each equality constraint
        (row) in ``A_eq``.
    :param bounds: 变量取值范围，sequence
        ``(min, max)`` pairs for each element in ``x``, defining
        the bounds on that parameter. Use None for one of ``min`` or
        ``max`` when there is no bound in that direction. By default
        bounds are ``(0, None)`` (non-negative)
        If a sequence containing a single tuple is provided, then ``min`` and
        ``max`` will be applied to all variables in the problem.
    :param bnbTreeNode: 该步的 bnbTreeNode
        提供一个 BnBTreeNode 实例作为根节点来记录求解过程，得到一个求解过程的树形图。

    Returns
    -------
    :return: {"success": True|False, "x": array([...]), "fun": ...}
                - success: 若求解成功则返回 True，否则 False
                - x: 最优解
                - fun: 最优目标函数值
    """

    # 对松弛问题求解
    r = optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)

    if bnbTreeNode:
        bnbTreeNode.res_x = r.x
        bnbTreeNode.res_fun = r.fun

    if not r.success:
        return {"success": False, "x": None, "fun": None}

    x = r.x
    z = sum(np.array(x) * np.array(c))

    if all([_is_int(i) for i in x]):  # 最优解是整数解
        return {"success": True, "x": x, "fun": z}

    # 有非整数变量
    # 找出第一个非整数变量的索引
    opt_idx = [i for i, v in enumerate(x) if not _is_int(v)][0]

    # 构造新的条件、问题
    # con1: <=
    new_con1 = [1 if i == opt_idx else 0 for i in range(len(A_ub[0]))]
    new_A1 = A_ub.copy()
    new_A1.append(new_con1)
    new_B1 = b_ub.copy()
    new_B1.append(math.floor(x[opt_idx]))

    # 构造新问题的 BnBTreeNode
    if bnbTreeNode:
        bnbTreeNode.left = BnBTreeNode()
        bnbTreeNode.left.x_idx = opt_idx
        bnbTreeNode.left.x_c = "<="
        bnbTreeNode.left.x_b = math.floor(x[opt_idx])

    # 递归求解新问题
    r1 = branch_and_bound(c, new_A1, new_B1, A_eq, b_eq, bounds, bnbTreeNode.left if bnbTreeNode else None)

    # 构造新的条件
    # con2: >
    new_con2 = [-1 if i == opt_idx else 0 for i in range(len(A_ub[0]))]
    new_A2 = A_ub.copy()
    new_A2.append(new_con2)
    new_B2 = b_ub.copy()
    new_B2.append(-math.ceil(x[opt_idx]))

    # 构造新问题的 BnBTreeNode
    if bnbTreeNode:
        bnbTreeNode.right = BnBTreeNode()
        bnbTreeNode.right.x_idx = opt_idx
        bnbTreeNode.right.x_c = ">="
        bnbTreeNode.right.x_b = math.ceil(x[opt_idx])

    # 递归求解新问题
    r2 = branch_and_bound(c, new_A2, new_B2, A_eq, b_eq, bounds, bnbTreeNode.right if bnbTreeNode else None)

    # 子问题返回了，找出其中的最优可行继续向上一层返回
    if r1["success"] and r2["success"]:
        return min((r1, r2), key=lambda A_ub: A_ub["fun"])
    elif r1["success"]:
        return r1
    elif r2["success"]:
        return r2
    else:
        return None


def _test1():
    c = [3, 4, 1]
    A_ub = [[-1, -6, -2], [-2, 0, 0]]
    b_ub = [-5, -3]
    A_eq = None
    b_eq = None
    bounds = [(0, None), (0, None), (0, None)]
    tree = BnBTree()
    r = branch_and_bound(c, A_ub, b_ub, A_eq, b_eq, bounds, tree.root)
    print(r)
    print(tree)


def _test2():
    c = [-40, -90]
    A_ub = [[9, 7], [7, 20]]
    b_ub = [56, 70]
    A_eq = None
    b_eq = None
    bounds = [(0, None), (0, None)]
    tree = BnBTree()
    r = branch_and_bound(c, A_ub, b_ub, A_eq, b_eq, bounds, tree.root)
    print(r)
    print(tree)


def _test3():
    c = [-1, -1]
    A_ub = [[2, 1], [4, 5]]
    b_ub = [6, 20]
    bounds = [(0, None), (0, None)]
    r = branch_and_bound(c, A_ub, b_ub, None, None, bounds)
    print(r)


if __name__ == "__main__":
    _test1()
    _test2()
    _test3()
