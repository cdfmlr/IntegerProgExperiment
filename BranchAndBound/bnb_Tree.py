class BnBTreeNode(object):
    """
    BnBTreeNode 是 BnBTree 的节点

    Fields
    ------
    left    : BnBTreeNode, 左子节点，分支定界法里的左枝
    right   : BnBTreeNode, 右子节点，分支定界法里的右枝

    x_idx   : int,   分支定界法里新增条件的变量索引
    x_c     : str,   分支定界法里新增条件的比较运算符 "<=" 或 ">="
    x_b     : float, 分支定界法里新增条件的右端常数

    res_x   : numpy array, 分支定界法里这一步的松弛解
    res_fun : float，      分支定界法里这一步的目标函数值

    sub_flag: bool, 若节点为*整颗BnBTree*的根节点则为 False，否则 True
    """

    def __init__(self):
        super().__init__()
        self.left = None
        self.right = None

        self.x_idx = None
        self.x_c = None
        self.x_b = None

        self.res_x = None
        self.res_fun = None

        self.sub_flag = True

    def __str__(self):
        if self.sub_flag:
            return f'x[{self.x_idx}] {self.x_c} {self.x_b}: {self.res_x} -> {self.res_fun}'
        else:
            return f'Root: {self.res_x} -> {self.res_fun}'


class BnBTree(object):
    """
    BnBTree 是表示分枝定界法求整数规划问题过程的树

    Fields
    ------
    root: 树根节点
    """

    def __init__(self):
        super().__init__()
        self.root = BnBTreeNode()
        self.root.sub_flag = False
        self.__str_tree = ""

    def __str__(self):
        if self.__str_tree == "":
            def walk(node, indentation):
                self.__str_tree += "\t|" * indentation + "-- "
                # print("\t|" * indentation + "--", end=" ")
                self.__str_tree += str(node) + "\n"
                # print(node)
                if node.left:
                    walk(node.left, indentation + 1)
                if node.right:
                    walk(node.right, indentation + 1)

            walk(self.root, 0)
        return self.__str_tree


def _bnb_tree_test():
    tree = BnBTree()

    tree.root.left = BnBTreeNode()
    tree.root.left.res_fun = "left"

    tree.root.right = BnBTreeNode()
    tree.root.right.res_fun = "right"

    tree.root.right.left = BnBTreeNode()
    tree.root.right.left.res_fun = "rl"

    tree.root.right.right = BnBTreeNode()
    tree.root.right.right.res_fun = "rr"

    tree.root.right.left.left = BnBTreeNode()
    tree.root.right.left.left.res_fun = "rll"

    print(tree)


if __name__ == "__main__":
    _bnb_tree_test()
