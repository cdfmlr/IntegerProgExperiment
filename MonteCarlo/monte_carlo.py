import time
import random


def monte_carlo(x_nums, fun, cons, bounds, random_times=10 ** 5):
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
