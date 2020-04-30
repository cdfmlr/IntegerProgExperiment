"""
BranchAndBound 提供求解整数规划问题的「分支定界法」

Public
------
func branch_and_bound(c, A_ub, b_ub, A_eq, b_eq, bounds, bnbTreeNode=None):
    对整数规划问题使用「分支定界法」进行*递归*求解。
class BnBTree: 表示分枝定界法求整数规划问题过程的树
class BnBTreeNode: BnBTree 的节点
"""

from BranchAndBound.bnb_Tree import BnBTreeNode, BnBTree
from BranchAndBound.branch_and_bound import branch_and_bound
