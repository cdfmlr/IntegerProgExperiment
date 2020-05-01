import time
import random


def monte_carlo(x_nums, fun, cons, bounds, random_times=10 ** 5):
    """
    monte_carlo 对整数规划问题使用「蒙特卡洛法」求满意解

    对于线性、非线性的整数规划，在一定计算量下可以考虑用 蒙特卡洛法 得到一个满意解。

    注意：蒙特卡洛法只能在一定次数的模拟中求一个满意解（通常不是最优的），而且对于每个变量必须给出有明确上下界的取值范围。

    问题模型：
        Minimize:   fun(x)

        Subject to: cons(x) <= 0
                    (x are integers)

    Parameters
    ----------
    :param x_nums: `int`, 未知数向量 x 的元素个数
    :param fun:  `(x: list) -> float`, 要最小化的目标函数
    :param cons: `(x: list) -> list`, 小于等于 0 的约束条件
    :param bounds: `list`, 各个 x 的取值范围
    :param random_times: `int`, 随机模拟次数

    Returns
    -------
    :return: {"x": array([...]), "fun": ...}
                - x: 最优解
                - fun: 最优目标函数值

    Examples
    --------
    试求得如下整数规划问题的一个满意解：
        Min  x_0 + x_1
        s.t. 2 * x_0 + x_1 <= 6
             4 * x_0 + 5 * x_1 <= 20
             (x_0、x_1 为整数)
    编写目标函数：
        >>> fun = lambda x: x[0] + x[1]
    编写约束条件：
        >>> cons = lambda x: [2 * x[0] + x[1] - 6, 4 * x[0] + 5 * x[1] - 20]
    指定取值范围：
        >>> bounds = [(0, 100), (0, 100)]
    调用蒙特卡洛法求解：
        >>> monte_carlo(2, fun, cons, bounds)
        {'fun': 4, 'x': [1, 3]}
    可以看的 monte_carlo 返回了一个满意解（事实上，这是个最优解，但一般情况下不是）。
    """
    random.seed(time.time)
    pb = 0
    xb = []
    for i in range(random_times):
        x = [random.randint(bounds[i][0], bounds[i][1]) for i in range(x_nums)]  # 产生一行x_nums列的区间[0, 99] 上的随机整数
        rf = fun(x)
        rg = cons(x)
        if all((a < 0 for a in rg)):  # 若 rg 中所有元素都小于 0，即符合约束条件
            if pb < rf:
                xb = x
                pb = rf
    return {"fun": pb, "x": xb}


def _test1():
    def fun(x):
        return x[0] + x[1]

    def cons(x):
        return [
            2 * x[0] + x[1] - 6,
            4 * x[0] + 5 * x[1] - 20,
        ]

    bounds = [(0, 100), (0, 100)]
    r = monte_carlo(2, fun, cons, bounds)
    print(r)


def _test2():
    def fun(x):
        return 40 * x[0] + 90 * x[1]

    def cons(x):
        return [
            9 * x[0] + 7 * x[1] - 56,
            7 * x[0] + 20 * x[1] - 70,
        ]

    bounds = [(0, 100), (0, 100)]
    r = monte_carlo(2, fun, cons, bounds)
    print(r)


def _test3():
    def f(x: list) -> int:
        return x[0] ** 2 + x[1] ** 2 + 3 * x[2] ** 2 + \
               4 * x[3] ** 2 + 2 * x[4] ** 2 - 8 * x[0] - 2 * x[1] - \
               3 * x[2] - x[3] - 2 * x[4]

    def g(x: list) -> list:
        return [
            sum(x) - 400,
            x[0] + 2 * x[1] + 2 * x[2] + x[3] + 6 * x[4] - 800,
            2 * x[0] + x[1] + 6 * x[2] - 200,
            x[2] + x[3] + 5 * x[4] - 200
        ]

    bounds = [(0, 99)] * 5
    r = monte_carlo(5, f, g, bounds)
    print(r)


if __name__ == "__main__":
    _test1()
    _test2()
    _test3()
