class LpSolve(object):
    """
    LpSolve 描述一个线性规划问题的解

    :attributes success:     是否得到了最优解
    :attributes description: 解的描述
    :attributes solve:       最优解
    :attributes target:      最优目标函数值
    """

    def __init__(self, success: bool, description: str, solve: list, target: float):
        self.success = success
        self.description = description
        self.solve = solve
        self.target = target

    def __str__(self):
        return f'最优化成功\t: {self.success}\n解的描述\t: {self.description}\n最优解\t: {self.solve}\n最优目标函数值\t: {self.target}'
