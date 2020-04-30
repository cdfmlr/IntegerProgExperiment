class LpSolver(object):
    """线性规划问题的解法抽象类

    该项目中的所有解法实现都继承于此类。
    """
    def __init__(self, problem):
        """
        :param problem: 待解决的问题(LpProblem实例)
        """
        pass

    def solve(self, **kwargs):
        """
        求解算法入口，调用此方法开始对该LpSolver实例化时传入的 LpProblem 进行求解

        :return: 求解结果(LpSolve实例)
        """
        pass
