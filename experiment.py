import numpy as np

import BranchAndBound
import CuttingPlane
import MonteCarlo
import HungarianAssignment


def branch_and_bound_test():
    print("\nBranch and Bound Test:\n" + "-" * 22)
    c = [3, 4, 1]
    A_ub = [[-1, -6, -2], [-2, 0, 0]]
    b_ub = [-5, -3]
    bounds = [(0, None), (0, None), (0, None)]
    tree = BranchAndBound.BnBTree()
    r = BranchAndBound.branch_and_bound(c, A_ub, b_ub, None, None, bounds, tree.root)
    print(r)
    print(tree)


def cutting_plane_test():
    print("\nCutting Plane Test:\n" + "-" * 22)
    c = [1, 1, 0, 0]
    a = [[2, 1, 1, 0], [4, 5, 0, 1]]
    b = [6, 20]
    r = CuttingPlane.cutting_plane(c, a, b)
    print(r)


def monte_carlo_test():
    print("\nMonte Carlo Test:\n" + "-" * 22)

    def fun(x):
        return x[0] + x[1]

    def cons(x):
        return [
            2 * x[0] + x[1] - 6,
            4 * x[0] + 5 * x[1] - 20,
        ]

    bounds = [(0, 100), (0, 100)]
    r = MonteCarlo.monte_carlo(2, fun, cons, bounds)
    print(r)


def hungarian_assignment_test():
    print("\nHungarian Assignment Test:\n" + "-" * 22)
    c = [[2, 15, 13, 4], [10, 4, 14, 15], [9, 14, 16, 13], [7, 8, 11, 9]]
    r = HungarianAssignment.hungarian_assignment(c)
    print(r)
    m = np.zeros(np.array(c).shape, dtype=int)
    m[r] = 1
    print(m)


if __name__ == "__main__":
    branch_and_bound_test()
    cutting_plane_test()
    monte_carlo_test()
    hungarian_assignment_test()
